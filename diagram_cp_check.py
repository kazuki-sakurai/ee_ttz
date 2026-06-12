"""
Per-diagram CP localization for e+e- -> t tbar Z.

Reuses pyHELAS.get_amphel() to obtain all 9 diagram amplitudes at a single
phase-space point and at its CP image (m12, -cth3, cth1, ph + pi), then
forms various subset sums and runs the same CP check as cp_check.py on each.

Tree-level diagram structure (1-indexed, matching ee_ttz.py):

    1 : Higgs s-channel    (Z* -> H Z; H -> tt)              [self-CP]
    2 : FSR-t   via gamma  (e+e- > gamma* > tt; t  -> tZ)    } CP-pair
    3 : FSR-t   via Z      (e+e- > Z*     > tt; t  -> tZ)    } 4
    4 : FSR-tb  via gamma  (e+e- > gamma* > tt; tb -> tbZ)   } 2
    5 : FSR-tb  via Z      (e+e- > Z*     > tt; tb -> tbZ)   } 3
    6 : ISR-e-  via gamma  (e- -> e-Z; gamma* > tt)          } CP-pair
    7 : ISR-e-  via Z      (e- -> e-Z; Z*     > tt)          } 8
    8 : ISR-e+  via gamma  (e+ -> e+Z; gamma* > tt)          } 6
    9 : ISR-e+  via Z      (e+ -> e+Z; Z*     > tt)          } 7

Under CP, the partners are:
    {1}          <-> {1}             (Higgs is self-CP)
    {2, 3}       <-> {4, 5}          (FSR-t  <-> FSR-tb)
    {6, 7}       <-> {8, 9}          (ISR-e- <-> ISR-e+)

So the CP-invariant subsets of the SM amplitude are exactly
    {1}, {2,3,4,5}, {6,7,8,9}, {1..9}
and any unions thereof.

What each row of the report means:

    For each subset D, we compute
        amp_p [lambda]  = sum_{i in D} amp_i (lambda; p)
        amp_cp[lambda]  = sum_{i in D} amp_i (lambda; CP(p))
    and the ratio
        r(lambda) = amp_p[lambda] / amp_cp[lambda_tilde]
    where lambda_tilde = (-lTB, -lT, -lZ).

      std(|r|)        : 0 iff |M|^2 is CP-invariant for this subset.
      std(arg)        : 0 iff the CP relation closes for this subset
                        (the ratio is a constant overall phase).
      CP^2 max res    : worst-case wrap of  f(lambda;p) + f(lambda~;CP(p))
                        which is 0 mod 2pi for SM-correct subsets.

CP-invariant subsets that FAIL std(arg) ~ 0 or CP^2 ~ 0 contain the bug.

Usage:
    python3 diagram_cp_check.py             # pol = R
    python3 diagram_cp_check.py L           # pol = L
"""

import os
import sys
sys.path.append('/Users/kazuki/Projects/pyHELAS')
import numpy as np

from ee_ttz      import mt, mZ, get_amphel
from ee_ttz_func import get_momenta


# Reference kinematic point (same as cp_check.py).
RS   = 1000.0
M12  = 600.0
CTH1 = 0.3
CTH3 = -0.4
PH   = 1.7

HEL_FINAL = [(lT, lTB, lZ)
             for lT  in (1, -1)
             for lTB in (1, -1)
             for lZ  in (1, 0, -1)]

# Subsets to test.  Tuple: (label, set of 1-indexed diagrams, expected_CP_invariant)
SUBSETS = [
    # --- baselines ------------------------------------------------------
    ("Full sum {1..9}",          set(range(1, 10)),         True),
    ("Higgs only  {1}",          {1},                       True),
    # --- CP-invariant unions of CP-pairs --------------------------------
    ("FSR all     {2,3,4,5}",    {2, 3, 4, 5},              True),
    ("ISR all     {6,7,8,9}",    {6, 7, 8, 9},              True),
    # --- single CP-pair halves: NOT CP-invariant by themselves ----------
    ("FSR-t  half {2,3}",        {2, 3},                    False),
    ("FSR-tb half {4,5}",        {4, 5},                    False),
    ("ISR-e- half {6,7}",        {6, 7},                    False),
    ("ISR-e+ half {8,9}",        {8, 9},                    False),
    # --- knock-outs: drop one CP-pair and re-test the rest --------------
    # If the remainder becomes CP-symmetric, the bug lived in the
    # dropped pair.
    ("Full - {1}",               set(range(1,10)) - {1},       True),
    ("Full - FSR-t  {2,3}",      set(range(1,10)) - {2,3},     False),
    ("Full - FSR-tb {4,5}",      set(range(1,10)) - {4,5},     False),
    ("Full - ISR-e- {6,7}",      set(range(1,10)) - {6,7},     False),
    ("Full - ISR-e+ {8,9}",      set(range(1,10)) - {8,9},     False),
    # --- enable only one CP-pair (lots of "expected fail" but useful) ----
    ("FSR-t  pair {2,3,4,5}\\{4,5}", {2,3},                  False),
]


def _amps_at(rs, m12, c3, c1, ph, lE, lEB):
    """Return a dict: hel -> 9-element complex array of per-diagram amps."""
    pp = {'E':  np.array([rs/2.0, 0.0, 0.0,  rs/2.0]),
          'EB': np.array([rs/2.0, 0.0, 0.0, -rs/2.0])}
    p1, p2, p3 = get_momenta(rs, m12, mt, mZ, c3, c1, ph)
    pp['T']  = np.array([p1.E, p1.px, p1.py, p1.pz])
    pp['TB'] = np.array([p2.E, p2.px, p2.py, p2.pz])
    pp['Z']  = np.array([p3.E, p3.px, p3.py, p3.pz])
    return {h: np.asarray(
                  get_amphel(pp,
                             {'E': lE, 'EB': lEB,
                              'T': h[0], 'TB': h[1], 'Z': h[2]}),
                  dtype=np.complex128)
            for h in HEL_FINAL}


def _subset_sum(amp_dict, subset, hel):
    """Sum the amplitudes for diagrams in `subset` (1-indexed) at one helicity."""
    arr = amp_dict[hel]
    return sum(arr[i - 1] for i in subset)


def _check_subset(amp_p, amp_cp, subset):
    """Compute (std|ratio|, std arg, CP^2 max residual) for a given subset."""
    ratios = []
    for hel in HEL_FINAL:
        lT, lTB, lZ = hel
        a_p  = _subset_sum(amp_p,  subset, ( lT,  lTB,  lZ))
        a_cp = _subset_sum(amp_cp, subset, (-lTB, -lT, -lZ))
        if abs(a_cp) < 1e-30 or abs(a_p) < 1e-30:
            continue
        ratios.append(a_p / a_cp)
    if not ratios:
        return None, None, None

    ratios    = np.array(ratios)
    std_mag   = float(np.std(np.abs(ratios)))
    std_phase = float(np.std(np.angle(ratios)))

    # CP^2 residual: f(lambda; p) + f(lambda_tilde; CP(p)) should be 0 mod 2*pi.
    cp2 = []
    for hel in HEL_FINAL:
        lT, lTB, lZ = hel
        a_p     = _subset_sum(amp_p,  subset, ( lT,  lTB,  lZ))
        a_cp_t  = _subset_sum(amp_cp, subset, (-lTB, -lT, -lZ))
        a_p_t   = _subset_sum(amp_p,  subset, (-lTB, -lT, -lZ))
        a_cp    = _subset_sum(amp_cp, subset, ( lT,  lTB,  lZ))
        if (abs(a_p) < 1e-30 or abs(a_cp_t) < 1e-30 or
            abs(a_p_t) < 1e-30 or abs(a_cp) < 1e-30):
            continue
        f_at_p    = np.angle(a_p / a_cp_t)
        f_at_cp_p = np.angle(a_p_t / a_cp)
        cp2.append(f_at_p + f_at_cp_p)
    cp2 = np.array(cp2) if cp2 else np.array([0.0])
    cp2_wrapped = ((cp2 + np.pi) % (2*np.pi)) - np.pi
    cp2_residual = float(np.max(np.abs(cp2_wrapped)))

    return std_mag, std_phase, cp2_residual


def _cross_pair_check(amp_p, amp_cp, label, sub_p, sub_cp):
    """For SM CP-invariance,
           M_{sub_p}(lambda; p) ~ phase * M_{sub_cp}(lambda_tilde; CP(p))
    with a single overall phase across all helicities.
    """
    ratios = []
    for hel in HEL_FINAL:
        lT, lTB, lZ = hel
        a_p  = _subset_sum(amp_p,  sub_p,  ( lT,  lTB,  lZ))
        a_cp = _subset_sum(amp_cp, sub_cp, (-lTB, -lT, -lZ))
        if abs(a_p) < 1e-30 or abs(a_cp) < 1e-30:
            continue
        ratios.append(a_p / a_cp)
    if not ratios:
        return None, None
    ratios    = np.array(ratios)
    std_mag   = float(np.std(np.abs(ratios)))
    std_phase = float(np.std(np.angle(ratios)))
    return std_mag, std_phase


def main(argv):
    pol = (argv[1] if len(argv) > 1 else 'R').upper()
    if pol not in ('R', 'L'):
        raise SystemExit("pol must be 'R' or 'L'")
    lE, lEB = (+1, -1) if pol == 'R' else (-1, +1)
    pol_lbl = "e_R^- e_L^+" if pol == 'R' else "e_L^- e_R^+"

    print()
    print(f"Per-diagram CP localization at one phase-space point  ({pol_lbl})")
    print(f"  point      (m12, cth3, cth1, ph) = "
          f"({M12}, {CTH3}, {CTH1}, {PH})")
    print(f"  CP'd point                       = "
          f"({M12}, {-CTH3}, {CTH1}, {PH + np.pi:.4f})")
    print()

    # All per-diagram amplitudes at p and at CP(p), for every helicity.
    amp_p  = _amps_at(RS, M12, CTH3,  CTH1, PH,           lE, lEB)
    amp_cp = _amps_at(RS, M12, -CTH3, CTH1, PH + np.pi,   lE, lEB)

    # Per-subset report.
    print("[A] Per-subset CP check")
    print(f"  {'subset':<32s}   {'expect':>6s}   "
          f"{'std(|r|)':>10s}   {'std(arg)':>10s}   "
          f"{'CP^2 max res':>14s}")
    print("  " + "-" * 84)
    for (name, subset, expected_cp) in SUBSETS:
        res = _check_subset(amp_p, amp_cp, subset)
        if res[0] is None:
            print(f"  {name:<32s}   (no usable ratios)")
            continue
        std_mag, std_phase, cp2 = res
        tag  = "PASS" if expected_cp else "n/a"
        flag = ""
        if expected_cp and (std_phase > 1e-6 or cp2 > 1e-6):
            flag = "   <-- BUG INSIDE"
        if (not expected_cp) and std_phase < 1e-6 and cp2 < 1e-6:
            flag = "   (unexpectedly CP)"
        print(f"  {name:<32s}   {tag:>6s}   "
              f"{std_mag:>10.3e}   {std_phase:>10.3e}   "
              f"{cp2:>14.3e}{flag}")
    print()

    # Cross-pair CP relations: directly test "subset A CP-maps to subset B".
    print("[B] Cross-pair CP relations")
    print("  For SM, e.g. M_{2,3}(lambda; p) ~ const * M_{4,5}(lambda~; CP(p)),")
    print("  so std(|r|) and std(arg) should both be ~ 0.")
    print()
    print(f"  {'pair test':<32s}   {'std(|r|)':>14s}   {'std(arg)':>14s}")
    print("  " + "-" * 72)
    pairs = [
        ("M_{2,3}(p) vs M_{4,5}(CP)",    {2, 3}, {4, 5}),
        ("M_{4,5}(p) vs M_{2,3}(CP)",    {4, 5}, {2, 3}),
        ("M_{6,7}(p) vs M_{8,9}(CP)",    {6, 7}, {8, 9}),
        ("M_{8,9}(p) vs M_{6,7}(CP)",    {8, 9}, {6, 7}),
        ("M_{1}(p) vs M_{1}(CP)",        {1},    {1}),
    ]
    for name, sub_p, sub_cp in pairs:
        res = _cross_pair_check(amp_p, amp_cp, name, sub_p, sub_cp)
        if res[0] is None:
            print(f"  {name:<32s}   (no usable ratios)")
            continue
        std_mag, std_phase = res
        flag = ""
        if std_phase > 1e-6:
            flag = "   <-- NOT CP-PAIR"
        elif std_mag > 1e-8:
            flag = "   (|r| varies)"
        print(f"  {name:<32s}   {std_mag:>14.3e}   {std_phase:>14.3e}{flag}")
    print()
    print("Interpretation:")
    print("  [A] : any 'BUG INSIDE' row contains the bad diagram(s).")
    print("  [B] : any 'NOT CP-PAIR' row means that pair is not actually")
    print("        CP-related under our (m12, -cth3, cth1, ph+pi) recipe.")
    print("        For SM, only diagrams known to be CP partners should pair.")


if __name__ == '__main__':
    main(sys.argv)
