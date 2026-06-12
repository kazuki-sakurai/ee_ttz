"""
CP-symmetry diagnostic for the integrated spin-density-matrix calculation
in psinteg_all.py.

Why each polarised state is *individually* CP-invariant.
-----------------------------------------------------------
In the CM frame the initial state (e_R^- at +z, e_L^+ at -z) is mapped by
CP to (e_L^+ at -z, e_R^- at +z) which, after the usual relabelling so the
electron sits at +z, is the SAME polarised state.  So CP does NOT swap R1
and R2; it acts within each.  P alone is what relates R1 to R2, and P is
broken in the SM by the V-A coupling, so R1 != R2 is expected.

For SM CP invariance:
    R1 = CP_map(R1),
    R2 = CP_map(R2),
    R3 = CP_map(R3)    where R3 = R1 + R2 (unpolarised),
with CP_map = (swap t<->tbar subsystems) x (flip every helicity index).

Two tests:

  [A] Per-polarisation CP invariance.
        Check R = CP_map(R) for R in {R1, R2, R3}.
        Expected rel.err ~ 1e-12 for SM-correct amplitudes integrated
        over the full phase space.

  [B] [optional, --amplitude] Amplitude-level CP check at one phase-space
        point.  For SM-correct amplitudes (same beam polarisation on
        both sides),
            M_R(lT, lTB, lZ; m12,  cth3, cth1, ph)
              ~  M_R(-lTB, -lT, -lZ; m12, -cth3, cth1, ph + pi),
        with a single constant overall phase across all 12 helicity
        assignments.  The "ratio" column should be a pure phase to
        within numerical noise.

Interpretation:
  - [A] PASS                -> integrated R-matrices respect CP.
  - [A] FAIL on R1 or R2    -> bug in how the amplitudes are evaluated
                               (most likely a helicity-convention / sign
                               issue in pyHELAS or in the helicity routing
                               in get_amphel).
  - [A] PASS but [B] FAIL   -> CP is restored only after integration; the
                               per-event CP relation in pyHELAS is broken.
                               (Should not happen for a CP-symmetric theory.)

Usage:
    python3 cp_check.py
        Run [A] and [B] on data/psinteg_all_res.npz.

    python3 cp_check.py path/to/file.npz
        Run [A] and [B] on a different file.

    python3 cp_check.py --amplitude
        Also run [C].  Requires pyHELAS / get_amphel / get_momenta
        (i.e. all the imports that psinteg_all.py uses).
"""

import os
import sys
import numpy as np


# ----------------------------------------------------------------------
# CP map on the integrated R-matrix (12 x 12 matrix on  H_t (x) H_tbar (x) H_Z
# with subsystem dimensions [2, 2, 3]).
# ----------------------------------------------------------------------

def cp_map(R12):
    """Apply the CP relabeling to a 12 x 12 spin-density matrix.

    The matrix is laid out with the helicity ordering used by
    psinteg_all.py:

        for lT  in (+1, -1):
            for lTB in (+1, -1):
                for lZ  in (+1, 0, -1):

    so the underlying tensor shape is (2, 2, 3, 2, 2, 3) and the row index
    runs in row-major order over (lT, lTB, lZ).  CP_map is

        (1) flip every helicity sign:
                +1 <-> -1   (qubit axes T, TB)
                +1 <-> -1, 0 -> 0   (qutrit axis Z)
            implemented by ::-1 along each axis;
        (2) swap the t and tbar subsystems:
            transpose (T <-> TB) on both row and column sides,
            i.e. axes (0, 1) and (3, 4).
    """
    R = R12.reshape(2, 2, 3, 2, 2, 3)
    # (1) helicity flips
    R = R[::-1, ::-1, ::-1, ::-1, ::-1, ::-1]
    # (2) t <-> tbar swap  (on row indices: axes 0,1; on column indices: 3,4)
    R = np.transpose(R, (1, 0, 2, 4, 3, 5))
    return R.reshape(12, 12)


def _rel_err(A, B):
    """max abs deviation of A-B, normalised by max abs entry of B."""
    return np.max(np.abs(A - B)) / max(np.max(np.abs(B)), 1e-30)


# ----------------------------------------------------------------------
# Tests on integrated R-matrices.
# ----------------------------------------------------------------------

def diagnose(npz_path):
    d   = np.load(npz_path)
    R1  = d['R1']                 # (Nrs, 12, 12)
    R2  = d['R2']
    rs  = d['rs']
    Nrs = R1.shape[0]

    print(f"Loaded {npz_path}")
    print(f"  Nrs = {Nrs}, R shape per point = {R1.shape[1:]}")
    print()

    # Test [A]: each polarisation should be CP-invariant separately.
    print("Test [A]: CP invariance of each polarised R")
    print("  R = CP_map(R) for R in {R1, R2, R3=R1+R2}")
    print("  expected rel.err ~ 1e-12 (machine precision)")
    print(f"  {'i':>3s}  {'sqrt(s) [GeV]':>14s}   "
          f"{'|R1 - CP(R1)|/|R1|':>20s}   "
          f"{'|R2 - CP(R2)|/|R2|':>20s}   "
          f"{'|R3 - CP(R3)|/|R3|':>20s}")
    print("  " + "-" * 100)

    pass_R1 = pass_R2 = pass_R3 = True
    for i in range(Nrs):
        r1 = _rel_err(cp_map(R1[i]),         R1[i])
        r2 = _rel_err(cp_map(R2[i]),         R2[i])
        R3 = R1[i] + R2[i]
        r3 = _rel_err(cp_map(R3),            R3)

        if r1 > 1e-8: pass_R1 = False
        if r2 > 1e-8: pass_R2 = False
        if r3 > 1e-8: pass_R3 = False

        flag1 = "" if r1 < 1e-8 else " *"
        flag2 = "" if r2 < 1e-8 else " *"
        flag3 = "" if r3 < 1e-8 else " *"

        print(f"  {i:>3d}  {rs[i]:>14.2f}   "
              f"{r1:>20.3e}{flag1:<2}   "
              f"{r2:>20.3e}{flag2:<2}   "
              f"{r3:>20.3e}{flag3:<2}")
    print()
    print("  (rows marked with '*' exceed the 1e-8 tolerance)")
    print()
    print("=" * 60)
    print(f"  R1 CP-invariant:  {'PASS' if pass_R1 else 'FAIL'}")
    print(f"  R2 CP-invariant:  {'PASS' if pass_R2 else 'FAIL'}")
    print(f"  R3 CP-invariant:  {'PASS' if pass_R3 else 'FAIL'}")
    print("=" * 60)


# ----------------------------------------------------------------------
# Optional pointwise amplitude test.
# ----------------------------------------------------------------------

# Different recipes for the CP-image of (m12, cth3, cth1, ph).  The first
# one is my original guess; #2 also flips cth1 (the most likely fix if the
# pyHELAS angle convention differs from mine); the rest are sanity bracket
# options.
CP_KINEMATICS = {
    'orig (-cth3, cth1, ph+pi)':       lambda c3, c1, p: (-c3,  c1, p + np.pi),
    'flip cth1 (-cth3, -cth1, ph+pi)': lambda c3, c1, p: (-c3, -c1, p + np.pi),
    '-cth3, cth1, -ph':                lambda c3, c1, p: (-c3,  c1, -p),
    '-cth3, -cth1, -ph':               lambda c3, c1, p: (-c3, -c1, -p),
}


def _phase_stats(amps_orig, amps_cp, hel_final):
    """Return (mean phase, std of phase, std of |ratio|) for the amplitude
    ratio M(orig)/M(CP) under the CP relabeling (lT, lTB, lZ) ->
    (-lTB, -lT, -lZ)."""
    ratios = []
    for (lT, lTB, lZ) in hel_final:
        a_o = amps_orig[(lT, lTB, lZ)]
        a_c = amps_cp[(-lTB, -lT, -lZ)]
        if abs(a_c) < 1e-30 or abs(a_o) < 1e-30:
            continue
        ratios.append(a_o / a_c)
    ratios = np.array(ratios)
    if not ratios.size:
        return None
    return (float(np.mean(np.angle(ratios))),
            float(np.std (np.angle(ratios))),
            float(np.std (np.abs  (ratios))))


def _cp2_residuals(amps_orig, amps_cp, hel_final):
    """For each helicity, return f(lambda; p) + f(lambda_tilde; CP(p)),
    which should be 0 mod 2*pi if the kinematic CP recipe is correct.
    We only have the original (p) and the CP'd (CP(p)) ratios; the CP^2
    consistency uses the CP'd ratios as f at CP(p)."""
    out = []
    for (lT, lTB, lZ) in hel_final:
        a_o = amps_orig[( lT,  lTB,  lZ)]
        a_c = amps_cp  [(-lTB, -lT, -lZ)]
        if abs(a_c) < 1e-30 or abs(a_o) < 1e-30:
            continue
        f_p     = np.angle(a_o / a_c)
        # At CP(p), the corresponding ratio uses the same hel pair
        # tilde'd both ways, which gives:
        a_o_cp  = amps_cp  [( lT,  lTB,  lZ)]
        a_c_cp  = amps_orig[(-lTB, -lT, -lZ)]
        if abs(a_c_cp) < 1e-30 or abs(a_o_cp) < 1e-30:
            continue
        f_cp_p  = np.angle(a_o_cp / a_c_cp)
        out.append((f_p + f_cp_p))
    return np.array(out)


def amplitude_check(rs=1000.0, m12=600.0, cth1=0.3, cth3=-0.4, ph=1.7,
                    pol='R'):
    """Pointwise amplitude-level CP check at one phase-space point.

    For each polarised initial state (R = e_R e_L, or L = e_L e_R)
    individually, SM CP invariance implies
        M(lT, lTB, lZ; m12,  cth3, cth1, ph)
          = phase * M(-lTB, -lT, -lZ; m12, -cth3, cth1, ph + pi),
    where the SAME beam polarisation appears on both sides (because CP
    of the initial state, after relabelling so the e- stays at +z, is
    the same polarised state).

    The CP-flipped kinematics in our parameterisation:
      - m_tt unchanged (Lorentz scalar).
      - cos theta_Z flips sign (p_Z reverses).
      - cos theta_t unchanged (new t direction in new tt-rest frame
        equals old t direction; reference axis also CP-invariant after
        relabelling).
      - phi shifts by pi (parity flip of the x, y plane).

    For SM-correct amplitudes the ratio M(orig) / M(CP'd) is a single
    pure phase across all 12 (lT, lTB, lZ) values.
    """
    sys.path.append('/Users/kazuki/Projects/pyHELAS')
    # get_amphel lives in ee_ttz.py, get_momenta in ee_ttz_func.py.
    from ee_ttz      import mt, mZ, get_amphel
    from ee_ttz_func import get_momenta

    lE, lEB = (+1, -1) if pol == 'R' else (-1, +1)
    pol_lbl = "e_R^- e_L^+" if pol == 'R' else "e_L^- e_R^+"

    print()
    print("=" * 78)
    print(f"Amplitude-level CP check at one phase-space point ({pol_lbl})")
    print(f"  point (m12, cth3, cth1, ph) = ({m12}, {cth3}, {cth1}, {ph})")
    print()
    print("For each candidate CP recipe (m12, cth3, cth1, ph) -> CP-image,")
    print("we report:")
    print("  std(|ratio|)        : should be ~ 0   (|M|^2 CP-invariance)")
    print("  std(arg(ratio))     : ~ 0  iff the recipe absorbs the phases")
    print("                        into a constant overall factor")
    print("  CP^2 mean residual  : sum f(lambda;p)+f(lambda_tilde;CP(p))")
    print("                        should be 0 mod 2*pi if recipe is right")
    print()

    pp_base = {'E':  np.array([rs/2, 0, 0,  rs/2]),
               'EB': np.array([rs/2, 0, 0, -rs/2])}

    HEL_FINAL = [(lT, lTB, lZ)
                 for lT  in (1, -1)
                 for lTB in (1, -1)
                 for lZ  in (1, 0, -1)]

    def _amps_at(c3, c1, p):
        pp = dict(pp_base)
        p1, p2, p3 = get_momenta(rs, m12, mt, mZ, c3, c1, p)
        pp['T']  = np.array([p1.E, p1.px, p1.py, p1.pz])
        pp['TB'] = np.array([p2.E, p2.px, p2.py, p2.pz])
        pp['Z']  = np.array([p3.E, p3.px, p3.py, p3.pz])
        return {h: get_amphel(pp,
                              {'E': lE, 'EB': lEB,
                               'T': h[0], 'TB': h[1], 'Z': h[2]}).sum()
                for h in HEL_FINAL}

    amp_orig = _amps_at(cth3, cth1, ph)

    header = (f"  {'recipe':<38s}   {'std(|ratio|)':>13s}   "
              f"{'std(arg)':>10s}   {'mean(f(p)+f(CP(p)))':>22s}")
    print(header)
    print("  " + "-" * (len(header) - 2))

    for name, recipe in CP_KINEMATICS.items():
        c3_cp, c1_cp, p_cp = recipe(cth3, cth1, ph)
        amp_cp = _amps_at(c3_cp, c1_cp, p_cp)
        stats = _phase_stats(amp_orig, amp_cp, HEL_FINAL)
        if stats is None:
            print(f"  {name:<38s}   (no usable ratios)")
            continue
        mean_phase, std_phase, std_mag = stats
        cp2_res = _cp2_residuals(amp_orig, amp_cp, HEL_FINAL)
        # Wrap residuals to (-pi, pi] and take the worst (max abs).
        cp2_wrapped = ((cp2_res + np.pi) % (2*np.pi)) - np.pi
        cp2_summary = float(np.max(np.abs(cp2_wrapped)))
        print(f"  {name:<38s}   {std_mag:>13.3e}   "
              f"{std_phase:>10.3e}   {cp2_summary:>22.3e}")


# ----------------------------------------------------------------------
def main(argv):
    here = os.path.dirname(os.path.abspath(__file__))

    # Find the input file -- either positional, or default location.
    npz = None
    for a in argv[1:]:
        if a.startswith('-'):
            continue
        npz = a
        break
    if npz is None:
        npz = os.path.join(here, 'data', 'psinteg_all_res.npz')

    if os.path.exists(npz):
        diagnose(npz)
    else:
        print(f"Input file not found: {npz}")
        print("(Run psinteg_all.py first to produce data/psinteg_all_res.npz.)")

    if '--amplitude' in argv:
        # Run for both polarisations so we can see whether the failure
        # (if any) is symmetric in R <-> L.
        amplitude_check(pol='R')
        amplitude_check(pol='L')


if __name__ == '__main__':
    main(sys.argv)
