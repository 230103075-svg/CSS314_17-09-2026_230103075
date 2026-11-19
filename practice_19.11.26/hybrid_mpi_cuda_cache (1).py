import os
import re
import sys
import json
import math
import time
import hashlib
import platform
import subprocess
import numpy as np
from mpi4py import MPI
from numba import cuda, njit, float64, uint64  # noqa: F401  (float64 used in shared array)

TILE_DIM = 16
SHARED_DIM = TILE_DIM + 2  # 18x18 shared tile
GLOBAL_NX = 512
GLOBAL_NY = 512
STENCIL_STEPS = 300
SHM_ITERS = 40_000_000
LAMBDA_DAMP = 0.15

LOCAL_N = GLOBAL_NX // 2          # 256 interior cells per rank per direction
PADDED_N = LOCAL_N + 2            # 258 with ghost layer
DX = 1.0 / (GLOBAL_NX + 1)        # 1/513
CFL_SAFETY = 0.9                  # dt = 0.9 * dx^2 / (4 kappa_max)
CHECKPOINTS = (0, 100, 200, 300)
SHM_SLOTS = 64
CTRL_IDX = 63                     # read-only control slot (never written)
STRIDES = (1, 8, 16)
SHM_REPS = 3

# message tags: direction the data travels
TAG_UP, TAG_DOWN, TAG_LEFT, TAG_RIGHT = 101, 102, 103, 104


def get_gpu_uuid() -> str:
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=uuid", "--format=csv,noheader"],
            encoding="utf-8",
        )
        return out.strip().splitlines()[0].strip()
    except Exception:  # noqa: BLE001
        return "GPU-UUID-UNAVAILABLE"


def get_gpu_name() -> str:
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=name,compute_cap,memory.total", "--format=csv,noheader"],
            encoding="utf-8",
        )
        return out.strip().splitlines()[0].strip()
    except Exception:  # noqa: BLE001
        return "unavailable"


def get_cpu_model() -> str:
    try:
        with open("/proc/cpuinfo", encoding="utf-8") as f:
            for line in f:
                if line.startswith("model name"):
                    return line.split(":", 1)[1].strip()
    except Exception:  # noqa: BLE001
        pass
    return platform.processor() or "unknown"


def detect_cpu_line_size() -> int:
    path = "/sys/devices/system/cpu/cpu0/cache/index0/coherency_line_size"
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            return int(f.read().strip())
    return 64


# ==============================================================================
# TASK 1: Hybrid MPI Cartesian Grid + Node-Local MPI-3 Shared-Memory Window
# ==============================================================================
def init_hybrid_context(github_user: str) -> dict:
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    if size != 4:
        raise RuntimeError(f"Requires exactly 4 MPI ranks (2x2 grid), got {size}.")
    cuda.select_device(0)
    gpu_uuid = comm.bcast(get_gpu_uuid() if rank == 0 else None, root=0)
    cpu_sig = f"{platform.machine()}::{os.cpu_count()}"
    line_bytes = detect_cpu_line_size()
    digest = hashlib.sha256(
        f"{github_user.strip()}::{gpu_uuid}::{cpu_sig}::{line_bytes}".encode("utf-8")
    ).hexdigest()
    instance_seed = int(digest[:16], 16)  # deterministic 64-bit seed

    # non-periodic 2x2 Cartesian communicator
    cart_comm = comm.Create_cart(dims=[2, 2], periods=[False, False], reorder=False)
    cart_rank = cart_comm.Get_rank()
    coords = cart_comm.Get_coords(cart_rank)
    up, down = cart_comm.Shift(0, 1)      # dim 0 = rows
    left, right = cart_comm.Shift(1, 1)   # dim 1 = columns

    # node-local shared-memory communicator + 64-byte-aligned window of 64 uint64 slots
    node_comm = comm.Split_type(MPI.COMM_TYPE_SHARED)
    node_rank = node_comm.Get_rank()
    slack = 128 // 8                                # room to align to 64 (and report 128)
    nbytes = (SHM_SLOTS + slack) * 8 if node_rank == 0 else 0
    win = MPI.Win.Allocate_shared(nbytes, 8, comm=node_comm)
    buf, _ = win.Shared_query(0)
    raw = np.ndarray(buffer=buf, dtype=np.uint64, shape=(SHM_SLOTS + slack,))
    raw_addr = raw.ctypes.data
    offset = ((-raw_addr) % 64) // 8
    shm_view = raw[offset:offset + SHM_SLOTS]
    base_addr = shm_view.ctypes.data
    if node_rank == 0:
        shm_view[:] = 0
    node_comm.Barrier()

    return {
        "comm": comm, "rank": rank, "size": size,
        "cart_comm": cart_comm, "cart_rank": cart_rank, "coords": tuple(coords),
        "up": up, "down": down, "left": left, "right": right,
        "node_comm": node_comm, "node_rank": node_rank, "win": win,
        "shm_raw": raw, "shm_view": shm_view,
        "shm_base_mod64": int(base_addr % 64), "shm_base_mod128": int(base_addr % 128),
        "github_user": github_user.strip(), "gpu_uuid": gpu_uuid, "cpu_sig": cpu_sig,
        "line_bytes": line_bytes, "instance_seed": instance_seed, "digest": digest,
    }


# ==============================================================================
# TASK 2: Rank-local GPU subdomain + halo pack / unpack kernels
# ==============================================================================
def initialize_fields(ctx: dict) -> dict:
    """Builds rank-local (258, 258) u, kappa, S from GLOBAL coordinates (no global grid)."""
    r, c = ctx["coords"]
    gi = r * LOCAL_N + np.arange(PADDED_N, dtype=np.float64)   # global padded row index 0..513
    gj = c * LOCAL_N + np.arange(PADDED_N, dtype=np.float64)   # global padded col index 0..513
    Y, X = np.meshgrid(gi * DX, gj * DX, indexing="ij")

    rng = np.random.default_rng(ctx["instance_seed"])          # seed-bound, identical on all ranks
    cx, cy = 0.30 + 0.10 * rng.random(), 0.55 + 0.10 * rng.random()

    kappa = 0.1 + 0.9 * np.exp(-((X - 0.6) ** 2 + (Y - 0.4) ** 2) / 0.05) \
        + 0.05 * np.sin(4 * np.pi * X) * np.sin(4 * np.pi * Y)
    src = 0.5 * np.sin(2 * np.pi * X) * np.sin(2 * np.pi * Y)
    u = np.sin(np.pi * X) * np.sin(np.pi * Y) + np.exp(-((X - cx) ** 2 + (Y - cy) ** 2) / 0.002)
    u[0, :] = u[-1, :] = u[:, 0] = u[:, -1] = 0.0               # ghosts filled by exchange / Dirichlet 0

    local_kmax = float(kappa[1:-1, 1:-1].max())
    kappa_max = ctx["cart_comm"].allreduce(local_kmax, op=MPI.MAX)
    dt_max = DX * DX / (4.0 * kappa_max)
    dt = CFL_SAFETY * dt_max
    return {"u0": np.ascontiguousarray(u), "kappa": np.ascontiguousarray(kappa),
            "src": np.ascontiguousarray(src), "kappa_max": kappa_max,
            "dt_max": dt_max, "dt": dt, "gauss_center": (cx, cy)}


@cuda.jit
def pack_halos_kernel(u, send):
    """Gathers top/bottom rows (contiguous) and left/right columns (strided) into 1D buffers."""
    t = cuda.grid(1)
    n = send.shape[1]
    if t < n:
        send[0, t] = u[1, t + 1]        # top row     -> up neighbour
        send[1, t] = u[n, t + 1]        # bottom row  -> down neighbour
        send[2, t] = u[t + 1, 1]        # left col    -> left neighbour  (stride 2064 B)
        send[3, t] = u[t + 1, n]        # right col   -> right neighbour (stride 2064 B)


@cuda.jit
def unpack_halos_kernel(u, recv):
    """Scatters received 1D buffers into ghost indices 0 and 257."""
    t = cuda.grid(1)
    n = recv.shape[1]
    if t < n:
        u[0, t + 1] = recv[0, t]        # from up    (its bottom row)
        u[n + 1, t + 1] = recv[1, t]    # from down  (its top row)
        u[t + 1, 0] = recv[2, t]        # from left  (its right col)
        u[t + 1, n + 1] = recv[3, t]    # from right (its left col)


# ==============================================================================
# TASK 3: Shared-memory tiled stencil + dual-stream pipeline
# ==============================================================================
@cuda.jit(device=True, inline=True)
def stencil_point(uc, un_, us, uw, ue, kc, kn, ks, kw, ke, s, dt, inv_dx2, lam):
    flux = (0.5 * (kc + ke) * (ue - uc) - 0.5 * (kc + kw) * (uc - uw)
            + 0.5 * (kc + ks) * (us - uc) - 0.5 * (kc + kn) * (uc - un_)) * inv_dx2
    return uc + dt * (flux - lam * uc + s)


@cuda.jit
def tiled_stencil_kernel(u, unew, kap, src, i0, i1, j0, j1, dt, inv_dx2, lam):
    tile = cuda.shared.array(shape=(SHARED_DIM, SHARED_DIM), dtype=float64)
    tx = cuda.threadIdx.x
    ty = cuda.threadIdx.y
    i = i0 + cuda.blockIdx.y * TILE_DIM + ty
    j = j0 + cuda.blockIdx.x * TILE_DIM + tx
    n0 = u.shape[0]
    n1 = u.shape[1]
    if i < n0 and j < n1:
        tile[ty + 1, tx + 1] = u[i, j]
        if ty == 0:
            tile[0, tx + 1] = u[i - 1, j]
        if ty == TILE_DIM - 1 and i + 1 < n0:
            tile[SHARED_DIM - 1, tx + 1] = u[i + 1, j]
        if tx == 0:
            tile[ty + 1, 0] = u[i, j - 1]
        if tx == TILE_DIM - 1 and j + 1 < n1:
            tile[ty + 1, SHARED_DIM - 1] = u[i, j + 1]
    cuda.syncthreads()
    if i <= i1 and j <= j1:
        unew[i, j] = stencil_point(
            tile[ty + 1, tx + 1], tile[ty, tx + 1], tile[ty + 2, tx + 1],
            tile[ty + 1, tx], tile[ty + 1, tx + 2],
            kap[i, j], kap[i - 1, j], kap[i + 1, j], kap[i, j - 1], kap[i, j + 1],
            src[i, j], dt, inv_dx2, lam)


@cuda.jit
def border_strip_kernel(u, unew, kap, src, dt, inv_dx2, lam):
    """Updates the 4 outer strips: rows 1 and N, columns 1 and N (rows 2..N-1)."""
    t = cuda.grid(1)
    n = u.shape[0] - 2
    total = 2 * n + 2 * (n - 2)
    if t < total:
        if t < n:
            i = 1
            j = 1 + t
        elif t < 2 * n:
            i = n
            j = 1 + (t - n)
        elif t < 2 * n + (n - 2):
            i = 2 + (t - 2 * n)
            j = 1
        else:
            i = 2 + (t - 2 * n - (n - 2))
            j = n
        unew[i, j] = stencil_point(
            u[i, j], u[i - 1, j], u[i + 1, j], u[i, j - 1], u[i, j + 1],
            kap[i, j], kap[i - 1, j], kap[i + 1, j], kap[i, j - 1], kap[i, j + 1],
            src[i, j], dt, inv_dx2, lam)


def make_state(ctx: dict, fields: dict) -> dict:
    st = {
        "d_u": cuda.to_device(fields["u0"]),
        "d_un": cuda.to_device(fields["u0"]),
        "d_k": cuda.to_device(fields["kappa"]),
        "d_s": cuda.to_device(fields["src"]),
        "d_send": cuda.device_array((4, LOCAL_N), dtype=np.float64),
        "d_recv": cuda.device_array((4, LOCAL_N), dtype=np.float64),
        "h_send": cuda.pinned_array((4, LOCAL_N), dtype=np.float64),
        "h_recv": cuda.pinned_array((4, LOCAL_N), dtype=np.float64),
        "stream_interior": cuda.stream(),
        "stream_boundary": cuda.stream(),
        "dt": fields["dt"], "inv_dx2": 1.0 / (DX * DX), "lam": LAMBDA_DAMP,
    }
    st["h_send"][:] = 0.0
    st["h_recv"][:] = 0.0   # PROC_NULL receives leave zeros = Dirichlet boundary
    return st


def post_irecvs(ctx, st):
    cc, h = ctx["cart_comm"], st["h_recv"]
    return [
        cc.Irecv([h[0], MPI.DOUBLE], source=ctx["up"], tag=TAG_DOWN),
        cc.Irecv([h[1], MPI.DOUBLE], source=ctx["down"], tag=TAG_UP),
        cc.Irecv([h[2], MPI.DOUBLE], source=ctx["left"], tag=TAG_RIGHT),
        cc.Irecv([h[3], MPI.DOUBLE], source=ctx["right"], tag=TAG_LEFT),
    ]


def post_isends(ctx, st):
    cc, h = ctx["cart_comm"], st["h_send"]
    return [
        cc.Isend([h[0], MPI.DOUBLE], dest=ctx["up"], tag=TAG_UP),
        cc.Isend([h[1], MPI.DOUBLE], dest=ctx["down"], tag=TAG_DOWN),
        cc.Isend([h[2], MPI.DOUBLE], dest=ctx["left"], tag=TAG_LEFT),
        cc.Isend([h[3], MPI.DOUBLE], dest=ctx["right"], tag=TAG_RIGHT),
    ]


HALO_THREADS = 128
HALO_BLOCKS = (LOCAL_N + HALO_THREADS - 1) // HALO_THREADS
BORDER_CELLS = 2 * LOCAL_N + 2 * (LOCAL_N - 2)
BORDER_BLOCKS = (BORDER_CELLS + HALO_THREADS - 1) // HALO_THREADS


def tile_grid(lo, hi):
    nb = (hi - lo + 1 + TILE_DIM - 1) // TILE_DIM
    return (nb, nb)


def step_serialized(ctx, st):
    s = st["stream_boundary"]
    reqs = post_irecvs(ctx, st)
    pack_halos_kernel[HALO_BLOCKS, HALO_THREADS, s](st["d_u"], st["d_send"])
    st["d_send"].copy_to_host(st["h_send"], stream=s)
    s.synchronize()
    reqs += post_isends(ctx, st)
    MPI.Request.Waitall(reqs)
    st["d_recv"].copy_to_device(st["h_recv"], stream=s)
    unpack_halos_kernel[HALO_BLOCKS, HALO_THREADS, s](st["d_u"], st["d_recv"])
    tiled_stencil_kernel[tile_grid(1, LOCAL_N), (TILE_DIM, TILE_DIM), s](
        st["d_u"], st["d_un"], st["d_k"], st["d_s"], 1, LOCAL_N, 1, LOCAL_N,
        st["dt"], st["inv_dx2"], st["lam"])
    s.synchronize()
    st["d_u"], st["d_un"] = st["d_un"], st["d_u"]


def step_overlapped(ctx, st):
    sb, si = st["stream_boundary"], st["stream_interior"]
    reqs = post_irecvs(ctx, st)
    pack_halos_kernel[HALO_BLOCKS, HALO_THREADS, sb](st["d_u"], st["d_send"])
    st["d_send"].copy_to_host(st["h_send"], stream=sb)
    # deep interior [2..255, 2..255] runs concurrently on its own stream
    tiled_stencil_kernel[tile_grid(2, LOCAL_N - 1), (TILE_DIM, TILE_DIM), si](
        st["d_u"], st["d_un"], st["d_k"], st["d_s"], 2, LOCAL_N - 1, 2, LOCAL_N - 1,
        st["dt"], st["inv_dx2"], st["lam"])
    sb.synchronize()                                  # boundary data is on the host
    reqs += post_isends(ctx, st)
    MPI.Request.Waitall(reqs)                         # MPI while the interior kernel runs
    st["d_recv"].copy_to_device(st["h_recv"], stream=sb)
    unpack_halos_kernel[HALO_BLOCKS, HALO_THREADS, sb](st["d_u"], st["d_recv"])
    border_strip_kernel[BORDER_BLOCKS, HALO_THREADS, sb](
        st["d_u"], st["d_un"], st["d_k"], st["d_s"], st["dt"], st["inv_dx2"], st["lam"])
    sb.synchronize()
    si.synchronize()
    st["d_u"], st["d_un"] = st["d_un"], st["d_u"]


def global_energy(ctx, st):
    u = st["d_u"].copy_to_host()
    local = float(np.sum(u[1:-1, 1:-1] ** 2) * DX * DX)
    return ctx["cart_comm"].allreduce(local, op=MPI.SUM)


def run_mode(ctx, fields, step_fn, steps):
    st = make_state(ctx, fields)
    energies = {0: global_energy(ctx, st)}
    ctx["cart_comm"].Barrier()
    elapsed = 0.0
    done = 0
    for cp in CHECKPOINTS[1:]:
        t0 = time.perf_counter()
        while done < min(cp, steps):
            step_fn(ctx, st)
            done += 1
        cuda.synchronize()
        elapsed += time.perf_counter() - t0
        energies[cp] = global_energy(ctx, st)
    return st["d_u"].copy_to_host(), energies, elapsed * 1e3


# ==============================================================================
# TASK 4: Multi-rank host cache-line false-sharing sweep (MPI-3 shared window)
# ==============================================================================
@njit(nogil=True)
def host_shm_accumulate(shm_view, idx, ctrl_idx, iters):
    """In-place read-modify-write on shm_view[idx], iters times.

    The index is re-derived each iteration from a load of the (never written, always 0)
    control slot in the SAME buffer. LLVM cannot prove the target address is loop
    invariant or that it does not alias the control slot, so the value cannot be hoisted
    into a register: every iteration performs a real load and a real store on the line.
    """
    one = np.uint64(1)
    for _ in range(iters):
        j = idx + np.int64(shm_view[ctrl_idx] >> np.uint64(63))
        shm_view[j] = shm_view[j] + one
    return shm_view[idx]


def benchmark_shm_strides(ctx):
    view, nc, rank = ctx["shm_view"], ctx["node_comm"], ctx["node_rank"]
    host_shm_accumulate(view, rank, CTRL_IDX, 1000)   # JIT warm-up
    nc.Barrier()                                      # nobody may still be warming up
    results = {}
    for stride in STRIDES:
        idx = rank * stride
        per_rep_max = []
        ok = True
        for _ in range(SHM_REPS):
            if rank == 0:
                view[:CTRL_IDX] = 0
            nc.Barrier()
            t0 = time.perf_counter()
            val = host_shm_accumulate(view, idx, CTRL_IDX, SHM_ITERS)
            t1 = time.perf_counter()
            nc.Barrier()
            ok &= int(val) == SHM_ITERS
            per_rep_max.append(nc.allreduce((t1 - t0) * 1e3, op=MPI.MAX))
        all_ok = nc.allreduce(ok, op=MPI.LAND)
        best = float(min(per_rep_max))
        results[stride] = {
            "stride_slots": stride, "spacing_bytes": stride * 8,
            "slot_indices": [r * stride for r in range(nc.Get_size())],
            "cpu_line_index": [((ctx["shm_base_mod64"] + 8 * r * stride) // 64)
                               for r in range(nc.Get_size())],
            "gpu_sector_index": [((ctx["shm_base_mod128"] + 8 * r * stride) // 32)
                                 for r in range(nc.Get_size())],
            "wall_ms_per_rep_max_over_ranks": per_rep_max,
            "wall_ms_best": best,
            "ns_per_update": best * 1e6 / SHM_ITERS,
            "counter_check_ok": bool(all_ok),
        }
    return results


# ==============================================================================
# Audits
# ==============================================================================
def _first_text(obj):
    if isinstance(obj, dict):
        return "\n".join(str(v) for v in obj.values())
    return str(obj)


def ptx_audit():
    try:
        ptx = _first_text(tiled_stencil_kernel.inspect_asm())
    except Exception as exc:  # noqa: BLE001
        return {"error": repr(exc)}
    return {
        "st.shared.f64": len(re.findall(r"\bst\.shared\.f64\b", ptx)),
        "ld.shared.f64": len(re.findall(r"\bld\.shared\.f64\b", ptx)),
        "bar.sync": len(re.findall(r"\b(?:bar|barrier)\.sync\b", ptx)),
        "ld.global.f64": len(re.findall(r"\bld\.global(?:\.nc)?\.f64\b", ptx)),
        "st.global.f64": len(re.findall(r"\bst\.global\.f64\b", ptx)),
        "ptx_lines": ptx.count("\n"),
    }


def llvm_audit():
    try:
        ir = _first_text(host_shm_accumulate.inspect_llvm())
    except Exception as exc:  # noqa: BLE001
        return {"error": repr(exc)}
    # isolate the compiled function body (not the CPython wrapper)
    m = re.search(r"define[^\n]*host_shm_accumulate[^\n]*\{(.*?)\n\}", ir, re.S)
    body = m.group(1) if m else ir
    blocks = re.split(r"\n(?=[\w.\-]+:)", body)
    loop_rmw_blocks = [b for b in blocks
                       if "phi " in b and re.search(r"load i64", b) and re.search(r"store i64", b)]
    return {
        "load_i64_total": len(re.findall(r"load i64", body)),
        "store_i64_total": len(re.findall(r"store i64", body)),
        "volatile_ops": body.count("volatile"),
        "atomicrmw_ops": body.count("atomicrmw"),
        "loop_blocks_with_load_and_store": len(loop_rmw_blocks),
        "register_hoisting_prevented": len(loop_rmw_blocks) > 0,
    }


# ==============================================================================
# TASK 5: Cross-tier verification + manifest export
# ==============================================================================
def main():
    github_user = sys.argv[1] if len(sys.argv) > 1 else os.environ.get("GITHUB_USERNAME", "")
    if not github_user.strip():
        raise SystemExit("Pass your GitHub username as the first argument.")
    ctx = init_hybrid_context(github_user)
    rank, cc = ctx["cart_rank"], ctx["cart_comm"]
    fields = initialize_fields(ctx)

    # JIT warm-up of all kernels (1 step each mode on a throw-away state)
    for fn in (step_serialized, step_overlapped):
        warm = make_state(ctx, fields)
        fn(ctx, warm)
    cuda.synchronize()
    cc.Barrier()

    u_ser, e_ser, ms_ser = run_mode(ctx, fields, step_serialized, STENCIL_STEPS)
    u_ovl, e_ovl, ms_ovl = run_mode(ctx, fields, step_overlapped, STENCIL_STEPS)
    local_diff = float(np.max(np.abs(u_ser[1:-1, 1:-1] - u_ovl[1:-1, 1:-1])))
    max_diff = cc.allreduce(local_diff, op=MPI.MAX)
    rank_diffs = cc.gather(local_diff, root=0)
    t_ser = cc.allreduce(ms_ser, op=MPI.MAX)
    t_ovl = cc.allreduce(ms_ovl, op=MPI.MAX)
    energy_rel = max(abs(e_ser[k] - e_ovl[k]) / max(abs(e_ser[k]), 1e-300) for k in e_ser)
    energy_ok = all(math.isfinite(e_ser[k]) and math.isfinite(e_ovl[k]) for k in e_ser) \
        and energy_rel < 1e-12

    topo = cc.gather({"rank": rank, "coords": list(ctx["coords"]),
                      "up": ctx["up"], "down": ctx["down"],
                      "left": ctx["left"], "right": ctx["right"]}, root=0)

    shm = benchmark_shm_strides(ctx)
    ptx = ptx_audit() if rank == 0 else None
    llvm = llvm_audit() if rank == 0 else None

    if rank == 0:
        s1, s8, s16 = (shm[s]["wall_ms_best"] for s in STRIDES)
        dt, dt_max = fields["dt"], fields["dt_max"]
        print("=" * 78)
        print("HYBRID MPI-CUDA STENCIL & CACHE-LINE COHERENCY LAB")
        print("=" * 78)
        print(f"GitHub user      : {ctx['github_user']}")
        print(f"INSTANCE_SEED    : {ctx['instance_seed']}  (sha256 {ctx['digest'][:16]}...)")
        print(f"GPU              : {get_gpu_name()}  UUID {ctx['gpu_uuid']}")
        print(f"CPU              : {get_cpu_model()}  sig {ctx['cpu_sig']}  line {ctx['line_bytes']} B")
        print(f"MPI ranks        : {ctx['size']} (2x2 non-periodic Cartesian)")
        for t in topo:
            print(f"  rank {t['rank']} coords {t['coords']}  up {t['up']:>2} down {t['down']:>2} "
                  f"left {t['left']:>2} right {t['right']:>2}")
        print("-" * 78)
        print(f"Grid             : {GLOBAL_NX}x{GLOBAL_NY} interior, {LOCAL_N}x{LOCAL_N} per rank "
              f"(padded {PADDED_N}x{PADDED_N}), dx = 1/{GLOBAL_NX + 1} = {DX!r}")
        print(f"kappa_max        : {fields['kappa_max']!r}")
        print(f"dt_max (CFL)     : {dt_max!r}")
        print(f"dt used          : {dt!r}  (dt/dt_max = {dt / dt_max!r})  CFL ok = {dt <= dt_max}")
        print("-" * 78)
        print("Global energy E = dx^2 * sum(u^2) via cart_comm.allreduce")
        for k in CHECKPOINTS:
            print(f"  step {k:>3}: serialized {e_ser[k]!r}   overlapped {e_ovl[k]!r}")
        print(f"max |u_ser - u_ovl| per rank : {rank_diffs}")
        print(f"max_diff (all ranks)         : {max_diff!r}   parity ok = {max_diff < 1e-12}")
        print(f"energy parity (rel)          : {energy_rel!r}   ok = {energy_ok}")
        print(f"gpu_serialized_ms            : {t_ser!r}")
        print(f"gpu_overlapped_ms            : {t_ovl!r}")
        print(f"overlap speedup              : {t_ser / t_ovl!r}")
        print("-" * 78)
        print(f"PTX audit (tiled_stencil_kernel): {ptx}")
        print(f"LLVM audit (host_shm_accumulate): {llvm}")
        print("-" * 78)
        print(f"MPI-3 shared window: base mod 64 = {ctx['shm_base_mod64']}, "
              f"base mod 128 = {ctx['shm_base_mod128']}, {SHM_ITERS:,} RMW per rank")
        for s in STRIDES:
            r = shm[s]
            print(f"  stride {s:>2} ({r['spacing_bytes']:>3} B): best {r['wall_ms_best']!r} ms  "
                  f"({r['ns_per_update']!r} ns/update)  CPU lines {r['cpu_line_index']}  "
                  f"GPU sectors {r['gpu_sector_index']}  check {r['counter_check_ok']}")
        print(f"false-sharing slowdown stride1/stride8  : {s1 / s8!r}")
        print(f"stride16/stride8 ratio                   : {s16 / s8!r}")
        print("=" * 78)

        manifest = {
            "github_username": ctx["github_user"],
            "instance_seed": ctx["instance_seed"], "seed_digest_sha256": ctx["digest"],
            "gpu_uuid": ctx["gpu_uuid"], "gpu": get_gpu_name(),
            "cpu_model": get_cpu_model(), "cpu_signature": ctx["cpu_sig"],
            "cpu_line_bytes": ctx["line_bytes"],
            "mpi": {"size": ctx["size"], "dims": [2, 2], "periods": [False, False],
                    "topology": topo, "library": MPI.Get_library_version().splitlines()[0]},
            "grid": {"global_nx": GLOBAL_NX, "global_ny": GLOBAL_NY, "local_n": LOCAL_N,
                     "padded_n": PADDED_N, "dx": DX, "tile": TILE_DIM, "shared_tile": SHARED_DIM},
            "physics": {"kappa_max": fields["kappa_max"], "dt_max": dt_max, "dt": dt,
                        "cfl_ratio": dt / dt_max, "cfl_ok": dt <= dt_max,
                        "lambda_damp": LAMBDA_DAMP, "steps": STENCIL_STEPS,
                        "gauss_center": list(fields["gauss_center"])},
            "energy": {"serialized": {str(k): v for k, v in e_ser.items()},
                       "overlapped": {str(k): v for k, v in e_ovl.items()},
                       "max_relative_difference": energy_rel, "ok": energy_ok},
            "parity": {"max_diff": max_diff, "per_rank": rank_diffs, "ok": max_diff < 1e-12},
            "timing": {"gpu_serialized_ms": t_ser, "gpu_overlapped_ms": t_ovl,
                       "overlap_speedup": t_ser / t_ovl},
            "ptx_audit": ptx, "llvm_audit": llvm,
            "shm": {"iters_per_rank": SHM_ITERS, "reps": SHM_REPS,
                    "base_mod64": ctx["shm_base_mod64"], "base_mod128": ctx["shm_base_mod128"],
                    "strides": {str(k): v for k, v in shm.items()},
                    "false_sharing_slowdown_s1_over_s8": s1 / s8,
                    "s16_over_s8": s16 / s8},
            "versions": {"python": platform.python_version(), "numpy": np.__version__,
                         "numba": __import__("numba").__version__,
                         "mpi4py": __import__("mpi4py").__version__},
        }
        with open("hybrid_lab_manifest.json", "w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2)
        print("Written hybrid_lab_manifest.json")

    cc.Barrier()
    ctx["win"].Free()


if __name__ == "__main__":
    main()
