"""
Phase-space integration of the e+ e- -> t tbar Z spin density matrix over the
(cos theta_Z, phi) angles at fixed (m_tt, cos theta_t), keeping the two
pure-helicity beam polarisations (e_R e_L and e_L e_R) separate.

This is the (m_tt, cos theta_t)-plane companion of `psinteg.py`, which scans
the (m_tt, cos theta_Z) plane with cos theta_t / phi integrated instead.

Integration: tensor product of Gauss-Legendre on cos theta_Z in
[-cth3_cut, +cth3_cut] and the composite trapezoid rule on phi in
[0, 2*pi).  Both rules converge much faster than uniform Monte Carlo for
smooth integrands; the trapezoid rule on a periodic interval is exponentially
convergent.  `cth3_cut = 1` reproduces the full kinematic range, `cth3_cut <
1` restricts to the central polar-angle window |cos theta_Z| < cth3_cut.

Output file:
    data/psinteg_cth1_res_<rs_str>.npz                  # cth3_cut = 1 (full)
    data/psinteg_cth1_res_cut<cth3_cut>_<rs_str>.npz    # otherwise

        R1                  : (N, 12, 12) complex   integrated R for (lE, lEB) = (+1, -1)
        R2                  : (N, 12, 12) complex   integrated R for (lE, lEB) = (-1, +1)
        rs                  : (N,)        float     sqrt(s)                           [GeV]
        m12                 : (N,)        float     m_{t tbar}                        [GeV]
        cth1                : (N,)        float     cos theta_t                       (dimensionless)
        cth3_cut            : scalar      float     cos theta_Z upper-cut applied     (dimensionless)
        dsig_eR_pbperGeV    : (N,)        float     d^2 sigma^{+-} / (d m_tt d cos theta_t)  [pb/GeV]
        dsig_eL_pbperGeV    : (N,)        float     d^2 sigma^{-+} / (d m_tt d cos theta_t)  [pb/GeV]
        dsig_unpol_pbperGeV : (N,)        float     (eR + eL) / 4 unpolarised average        [pb/GeV]

Differential cross section
--------------------------
At each (m_tt, cos theta_t) scan point we evaluate

    d^2 sigma^{lE lEB} / (d m_tt d cos theta_t)  |_{|cos theta_Z| < cth3_cut}
        = N_c * m_tt * beta_1(m_tt^2) * beta_2(m_tt^2)
                / (1024 pi^4 * s)
                * Tr(R_int^{lE lEB})                              [GeV^-3]

with R_int the inner-angle integral (over cos theta_Z and phi).  For
`cth3_cut = 1` this is the fully-integrated differential cross section
in the (m_tt, cos theta_t) plane; for `cth3_cut < 1` it is the fiducial
differential cross section restricted to |cos theta_Z| < cth3_cut.  N_c
= 3 is the QCD colour multiplicity of the t-tbar pair; the unpolarised
average uses (sigma^{+-} + sigma^{-+}) / 4.  Output is converted from
GeV^-3 to pb/GeV via 1 GeV^-2 = 3.8937937e8 pb.

Usage
-----
    python3 psinteg_cth1.py                  # full range, cth3_cut = 1
    python3 psinteg_cth1.py 0.75             # restricted, cth3_cut = 0.75
    python3 psinteg_cth1.py 1 0.75           # both back-to-back (one file each)

Multiple positional values are processed in sequence, so the typical "give
me both files" invocation is `python3 psinteg_cth1.py 1 0.75`.
"""

import os, sys
import time
sys.path.append('/Users/kazuki/Projects/pyHELAS')
import numpy as np
from math import acos, pi, sqrt
from numpy.polynomial.legendre import leggauss

from QI_functions import *
from ee_ttz import *
from ee_ttz_func import *


######################################################
# Kinematic and unit-conversion helpers
######################################################

def get_beta(M, m1, m2):
    """Two-body phase-space factor: sqrt(lambda(M^2, m1^2, m2^2)) / M^2.

    Returns 0 (rather than NaN) for kinematics infinitesimally below
    threshold so that boundary points don't poison the cross-section
    evaluation.
    """
    beta_sq = (1.0
               - 2.0 * (m1**2 + m2**2) / M**2
               +       (m1**2 - m2**2)**2 / M**4)
    return sqrt(max(0.0, beta_sq))


# 1 GeV^-2 = 3.8937937e8 pb  (PDG (hbar c)^2).
GEV_INV2_TO_PB = 3.8937937e8

# QCD colour multiplicity of the t-tbar pair.
NC = 3


######################################################
# Setup (rs, kinematic edges, scan resolution)
######################################################

rs     = 1000
eps    = 10**-4
mttmin = 2 * mt * (1 + eps)
mttmax = (rs - mZ) * (1 - eps)

# Outer scan over (m12, cos theta_t).
nx, ny = 60, 60

# Inner-integral quadrature parameters.  Same density as psinteg.py: 16 GL
# nodes in cos theta_Z and 16 trapezoid nodes in phi gives 6-7 digits on a
# smooth integrand.  Bump these (especially `ncth3`) if you want more
# precision on the cth3_cut = 0.75 variant where the integrand has steeper
# variation per unit cos theta_Z.
ncth3 = 16        # Gauss-Legendre nodes in cos theta_Z
nph   = 16        # Trapezoid nodes in phi in [0, 2*pi)

# Beam four-momenta at fixed sqrt(s).
pp = {'E':  np.array([rs/2.0, 0.0, 0.0,  rs/2.0]),
      'EB': np.array([rs/2.0, 0.0, 0.0, -rs/2.0])}

# Trap nodes on the periodic phi axis are independent of any other knob.
ph_nodes = np.linspace(0.0, 2*pi, nph, endpoint=False)
ph_w     = np.full(nph, 2*pi / nph)

# Helicity bookkeeping (same as psinteg.py).
HEL_FINAL = [(lT, lTB, lZ)
             for lT  in (1, -1)
             for lTB in (1, -1)
             for lZ  in (1, 0, -1)]
INI_KEEP  = [(+1, -1), (-1, +1)]

# Columns we record per outer scan point.  The outer scan now varies
# (m12, cth1), so cth1 takes the place that th3 had in psinteg.py.
input_keys = ['rs', 'm12', 'cth1']
xsec_keys  = ['dsig_eR_pbperGeV',
              'dsig_eL_pbperGeV',
              'dsig_unpol_pbperGeV']


######################################################
# Quadrature helpers and per-run filename
######################################################

def _cth3_quadrature(cth3_cut, n_nodes=ncth3):
    """Gauss-Legendre nodes & weights for the integral
        int_{-cth3_cut}^{+cth3_cut} f(cos theta_Z) d(cos theta_Z)
    For cth3_cut = 1 these reduce to the standard leggauss output.
    Implemented as a linear rescaling of the [-1, 1] nodes/weights.
    """
    nodes_std, w_std = leggauss(n_nodes)
    nodes = float(cth3_cut) * nodes_std       # in [-cth3_cut, +cth3_cut]
    weights = float(cth3_cut) * w_std         # Jacobian d cth3 = cth3_cut * d t
    return nodes, weights


def _out_path(here, rs_str, cth3_cut):
    """`data/psinteg_cth1_res_<rs>.npz` for cth3_cut == 1, otherwise
    `data/psinteg_cth1_res_cut<cth3_cut>_<rs>.npz`.  The :g format keeps
    sensible round-trip filenames (e.g. 0.75 stays 0.75, 1 stays 1).
    """
    if abs(float(cth3_cut) - 1.0) < 1e-12:
        base = f'psinteg_cth1_res_{rs_str}.npz'
    else:
        base = f'psinteg_cth1_res_cut{cth3_cut:g}_{rs_str}.npz'
    return os.path.join(here, 'data', base)


def _rs_str(rs):
    """600 -> '0.6', 1000 -> '1', 500 -> '0.5'.  Matches psinteg.py's
    filename convention so the two scripts produce companion files in
    `data/` for the same sqrt(s)."""
    if rs == 1000:   return 1
    if rs == 500:    return 0.5
    if rs == 600:    return 0.6
    return f'{rs/1000:g}'


######################################################
# Run a single (m_tt, cos theta_t) scan at given cth3_cut
######################################################

def run_scan(cth3_cut):
    print(f"\n==> cth3_cut = {cth3_cut:g}  "
          f"({'full kinematic range' if cth3_cut >= 1.0 else 'restricted |cos theta_Z| < ' + format(cth3_cut, 'g')})",
          flush=True)

    # Per-run quadrature: cos theta_Z on [-cth3_cut, +cth3_cut], phi on
    # [0, 2 pi).  Flatten into a single 1-D loop the same way psinteg.py
    # does for cos theta_t and phi.
    cth3_nodes, cth3_w = _cth3_quadrature(cth3_cut)
    quad_w   = np.outer(cth3_w, ph_w).ravel()    # length ncth3 * nph
    cth3_grid = np.repeat(cth3_nodes, nph)
    ph_grid   = np.tile(ph_nodes, ncth3)

    # Outer scan axes: x = m12, y = cth1.
    xar = np.linspace(mttmin, mttmax, nx)
    yar = np.linspace(-1 + eps, 1 - eps, ny)

    records = {k: [] for k in input_keys + xsec_keys}
    Rpol1   = []   # integrated R per outer point for (+1, -1)
    Rpol2   = []   # ...                              for (-1, +1)

    n_xdm    = len(xar)
    t_total0 = time.perf_counter()

    for ixdm, m12 in enumerate(xar):
        t_step0 = time.perf_counter()

        for cth1 in yar:

            # Accumulate the integrated R for each kept (lE, lEB).
            R_int = {ini: np.zeros((12, 12), dtype=np.complex128)
                     for ini in INI_KEEP}

            # Single pass over the flattened (cth3, ph) node grid.  The
            # kinematics depend ONLY on (m12, cth3, cth1, ph), so we build
            # the four-momenta once per (cth3, ph) node and reuse them
            # for both polarisations.
            for k in range(quad_w.size):
                cth3 = cth3_grid[k]
                ph   = ph_grid[k]
                w    = quad_w[k]

                p1, p2, p3 = get_momenta(rs, m12, mt, mZ, cth3, cth1, ph)
                pp['T']  = np.array([p1.E, p1.px, p1.py, p1.pz])
                pp['TB'] = np.array([p2.E, p2.px, p2.py, p2.pz])
                pp['Z']  = np.array([p3.E, p3.px, p3.py, p3.pz])

                for (lE, lEB) in INI_KEEP:
                    amp = np.array([
                        get_amphel(pp,
                                   {'E': lE, 'EB': lEB,
                                    'T':  lT, 'TB': lTB, 'Z': lZ}).sum()
                        for (lT, lTB, lZ) in HEL_FINAL
                    ])
                    R_int[(lE, lEB)] += w * np.outer(amp, amp.conj())

            # Differential cross section at this (m12, cth1) point.
            #   d^2 sigma / (d m_tt d cos theta_t)
            #     = N_c * m_tt * beta_1 * beta_2 / (1024 pi^4 s) * Tr(R_int)
            # in GeV^-3; multiply by GEV_INV2_TO_PB to get pb/GeV (cos
            # theta_t is dimensionless).  For cth3_cut < 1 this is the
            # fiducial differential cross section restricted to
            # |cos theta_Z| < cth3_cut.
            beta1     = get_beta(rs,  m12, mZ)        # sqrt(s) -> ttbar + Z
            beta2     = get_beta(m12, mt,  mt)        # ttbar   -> t + tbar
            prefactor = (NC * m12 * beta1 * beta2
                         / (1024.0 * pi**4 * rs**2)
                         * GEV_INV2_TO_PB)
            tr_eR      = np.trace(R_int[(+1, -1)]).real
            tr_eL      = np.trace(R_int[(-1, +1)]).real
            dsig_eR    = prefactor * tr_eR
            dsig_eL    = prefactor * tr_eL
            dsig_unpol = 0.25 * (dsig_eR + dsig_eL)

            records['rs'  ].append(rs)
            records['m12' ].append(m12)
            records['cth1'].append(cth1)
            records['dsig_eR_pbperGeV'   ].append(dsig_eR)
            records['dsig_eL_pbperGeV'   ].append(dsig_eL)
            records['dsig_unpol_pbperGeV'].append(dsig_unpol)
            Rpol1.append(R_int[(+1, -1)].copy())
            Rpol2.append(R_int[(-1, +1)].copy())

        t_step  = time.perf_counter() - t_step0
        t_total = time.perf_counter() - t_total0
        eta     = t_total / (ixdm + 1) * (n_xdm - ixdm - 1)
        print(f"  IXDM={ixdm+1:>2d}/{n_xdm}   m12={m12:8.2f}   "
              f"step = {t_step:8.2f} s   total = {t_total:8.2f} s   "
              f"ETA = {eta:8.2f} s",
              flush=True)

    # Save.
    here     = os.path.dirname(os.path.abspath(__file__))
    rs_str   = _rs_str(rs)
    out_path = _out_path(here, rs_str, cth3_cut)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    np.savez_compressed(
        out_path,
        R1=np.array(Rpol1, dtype=np.complex128),
        R2=np.array(Rpol2, dtype=np.complex128),
        cth3_cut=np.float64(cth3_cut),
        **{k: np.array(v) for k, v in records.items()},
    )
    print(f"Saved {len(Rpol1)} (m12, cth1) points to {out_path}")
    print(f"  Quadrature: {ncth3} GL (cth3 in [-{cth3_cut:g}, +{cth3_cut:g}]) "
          f"x {nph} trap (ph) = {ncth3*nph} nodes per outer point")


######################################################
# Driver
######################################################

if __name__ == '__main__':
    if len(sys.argv) > 1:
        cuts = [float(a) for a in sys.argv[1:]]
    else:
        # Default: just the full-range variant.  Pass "1 0.75" on the
        # command line to also produce the restricted output.
        cuts = [1.0]

    for c in cuts:
        if not (0.0 < c <= 1.0):
            raise SystemExit(
                f"cth3_cut must be in (0, 1]; got {c}")
        run_scan(c)
