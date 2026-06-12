"""
Mosek-free reference implementation (cvxpy) of the two SDPs used by ppt_julia /
quantum-correlations, for cross-checking the Julia results.

- ppt_mixture_visibility(rho, dims): white-noise robustness t* such that
  t*rho + (1-t*) I/d is a PPT mixture.  t* < 1  <=>  rho is genuinely
  multipartite entangled.   (== Julia method="PPT_mixer")

- gmn(rho, dims): renormalized genuine multipartite negativity of Eq.(8),
  Hofmann-Moroder-Guhne 2014.  Returned in the same N_std convention as
  ppt_julia.gmn_hmg (i.e. raw HMG value / 2).   (== Julia method="HMG")

Both enumerate every bipartition m|m̄ of the system (subsets of size
1..floor(N/2)), exactly like RobustnessToPPTMixture.jl.

Requires: pip install cvxpy   (uses the bundled SCS/Clarabel solvers).
"""
import itertools
import numpy as np
import cvxpy as cp

def _bipartitions(N):
    """Inequivalent bipartitions m|m̄ as subsets of {0..N-1}, sizes 1..floor(N/2)."""
    parts = []
    for m in range(1, N // 2 + 1):
        for s in itertools.combinations(range(N), m):
            parts.append(list(s))
    # for even N, size-N/2 subsets double-count m|m̄; drop the duplicates
    if N % 2 == 0:
        half = [p for p in parts if len(p) == N // 2]
        keep, seen = [], set()
        for p in parts:
            if len(p) != N // 2:
                keep.append(p); continue
            comp = tuple(sorted(set(range(N)) - set(p)))
            if comp in seen:
                continue
            seen.add(tuple(p)); keep.append(p)
        parts = keep
    return parts


def _pt(expr, dims, axes):
    for ax in axes:
        expr = cp.partial_transpose(expr, dims, ax)
    return expr


def _check(rho, dims):
    rho = np.asarray(rho, dtype=complex)
    d = int(np.prod(dims))
    if rho.shape != (d, d):
        raise ValueError(f"rho must be {d}x{d} for dims={dims}, got {rho.shape}")
    return (rho + rho.conj().T) / 2, d


def ppt_mixture_visibility(rho, dims, solver=cp.SCS):
    rho, d = _check(rho, dims)
    bips = _bipartitions(len(dims))
    X = [cp.Variable((d, d), hermitian=True) for _ in bips]
    t = cp.Variable()
    cons = [Xi >> 0 for Xi in X]
    cons += [_pt(Xi, dims, s) >> 0 for Xi, s in zip(X, bips)]
    cons.append(t * rho + (1 - t) / d * np.eye(d) == cp.sum(X))
    prob = cp.Problem(cp.Maximize(t), cons)
    prob.solve(solver=solver)
    if prob.status not in ("optimal", "optimal_inaccurate"):
        raise RuntimeError(f"PPT-mixture SDP not solved: status={prob.status}")
    return float(t.value)


def get_GMN_QC(rho, dims, solver=cp.SCS):
    # the code approximating GMN in Eq.(8) of arXiv:1401.2424
    rho, d = _check(rho, dims)
    bips = _bipartitions(len(dims))
    X = [cp.Variable((d, d), hermitian=True) for _ in bips]
    Y = [cp.Variable((d, d), hermitian=True) for _ in bips]
    cons = [Xi >> 0 for Xi in X] + [Yi >> 0 for Yi in Y]
    W = X[0] + _pt(Y[0], dims, bips[0])
    for Xi, Yi, s in zip(X, Y, bips):
        cons.append(W == Xi + _pt(Yi, dims, s))
    cons += [np.eye(d) - W >> 0, np.eye(d) + W >> 0]
    prob = cp.Problem(cp.Minimize(cp.real(cp.trace(rho @ W))), cons)
    prob.solve(solver=solver)
    if prob.status not in ("optimal", "optimal_inaccurate"):
        raise RuntimeError(f"GMN SDP not solved: status={prob.status}")
    return max(0.0, -float(prob.value)) / 2.0   # /2 -> N_std, matches ppt_julia


def get_GMN(rho, dims, solver=cp.CLARABEL):
    """
    The *exact* renormalized genuine multipartite negativity of Eq.(8) of
    Hofmann-Moroder-Guhne 2014 (arXiv:1401.2424), in the standard negativity
    convention N_std = sum of |negative eigenvalues| (so pure GHZ -> 0.5,
    matching the paper and the project's N_A/N_B/N_G columns).

    Instead of the witness form of Eq.(8) -- which has the *unbounded* positive
    operators P_m and is numerically ill-conditioned (it converges only to
    'optimal_inaccurate') -- this solves the mathematically equivalent mixed
    convex-roof / dual SDP of Theorem 2 / Eq.(9) and Appendix A (Eq. A.15-A.17):

        N_g(rho) = min  sum_m tr(Z_m^-)
        s.t.  sum_m Z_m = rho,                         (a decomposition of rho)
              Z_m >= 0,                                (each component a state)
              Z_m^{T_m} + Z_m^- >= 0,  Z_m^- >= 0,     (negative part of Z_m^{T_m})

    where m runs over the inequivalent bipartitions and tr(Z_m^-) = p_m N_m(rho_m)
    is the bipartite-negativity contribution of component m.  All variables are
    bounded (the Z_m sum to a trace-1 state), so the problem is well conditioned
    and converges cleanly to 'optimal' (use the default CLARABEL solver).

    This is the quantity to use when you want the literal Eq.(8) value rather
    than the stabilized -I<=W<=I variant returned by ppt_julia.gmn_hmg / gmn():
    the two agree on pure states but differ by a few percent on mixed states
    (e.g. 0.3645 vs 0.337 on the S->ttg idm=25 state).

    Parameters
    ----------
    rho : (D, D) array_like, complex   (D = prod(dims))
    dims : sequence of int             (e.g. [2, 2, 2] or [2, 2, 3])
    solver : cvxpy solver, optional    (default CLARABEL; SCS may report
             'optimal_inaccurate' but gives the same value to ~3 decimals)

    Returns
    -------
    float
        N_g(rho) >= 0; 0 iff rho is a PPT mixture.  Satisfies
        N_g(rho) <= min_m N_m(rho) (the min-cut negativity).
    """
    rho, d = _check(rho, dims)
    bips = _bipartitions(len(dims))
    Z = [cp.Variable((d, d), hermitian=True) for _ in bips]   # components Z_m
    Zneg = [cp.Variable((d, d), hermitian=True) for _ in bips]  # negative parts
    cons = []
    for Zi, Zni, s in zip(Z, Zneg, bips):
        cons += [Zi >> 0, Zni >> 0, _pt(Zi, dims, s) + Zni >> 0]
    cons.append(cp.sum(Z) == rho)
    obj = cp.Minimize(cp.sum([cp.real(cp.trace(Zni)) for Zni in Zneg]))
    prob = cp.Problem(obj, cons)
    prob.solve(solver=solver)
    if prob.status not in ("optimal", "optimal_inaccurate"):
        raise RuntimeError(f"exact-GMN SDP not solved: status={prob.status}")
    return max(0.0, float(prob.value))
