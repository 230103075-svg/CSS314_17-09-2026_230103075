# Task 07 — Maximum Tolerable Serial Overhead

**Given:** P = 8,192 cores, required S(P) ≥ 7,500.

## Derivation

```
S(P) = P − α(P − 1) ≥ S_req
        α(P − 1) ≤ P − S_req
               α ≤ (P − S_req) / (P − 1)
```

S(P) decreases as α increases (dS/dα = −(P − 1) < 0), so the largest α that meets the requirement is the boundary value:

```
α_max = (8192 − 7500) / (8192 − 1) = 692 / 8191
```

## Exact numerical tolerance

* exact fraction: **α_max = 692/8191** (already in lowest terms)
* decimal: **α_max = 0.0844829691** ≈ **8.448297 %**

## Check

```
S = 8192 − (692/8191)·8191 = 8192 − 692 = 7500   ✓
```

Any α ≤ 0.0844829691 (8.448 % of the parallel run spent serially) meets the target; at α_max the efficiency is exactly 7500/8192 = 0.915527.
