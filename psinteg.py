"""
Phase-space integration of the e+ e- -> t tbar Z spin density matrix over the
(cos theta_t, phi) angles at fixed (m_tt, cos theta_Z), keeping the two
pure-helicity beam polarisations (e_R e_L and e_L e_R) separate.  Stores the
two integrated 12x12 R-matrices per outer scan point in a .npz file for later
QI analysis.

Integration: tensor product of Gauss-Legendre on cos theta_t in [-1, 1] and
the composite trapezoid rule on phi in [0, 2*pi).  Both rules converge much
faster than uniform Monte Carlo for smooth integrands; the trapezoid rule on
a periodic interval is exponentially convergent.

Output file (one per `mode`):
    data/psinteg_res_<mode>.npz
        R1                  : (N, 12, 12) complex   integrated R for (lE, lEB) = (+1, -1)
        R2                  : (N, 12, 12) complex   integrated R for (lE, lEB) = (-1, +1)
        rs                  : (N,)        float     sqrt(s)
        m12                 : (N,)        float     m_{t tbar}                 [GeV]
        th3                 : (N,)        float     acos(cos theta_Z)          [rad]
        dsig_eR_pbperGeV    : (N,)        float     d^2 sigma^{+-} /(dm_{tt} d cos theta_Z) [pb/GeV]
        dsig_eL_pbperGeV    : (N,)        float     d^2 sigma^{-+} /(dm_{tt} d cos theta_Z) [pb/GeV]
        dsig_unpol_pbperGeV : (N,)        float     (eR+eL)/4 unpolarised average           [pb/GeV]

Differential cross section
--------------------------
At each (m_{tt}, cos theta_Z) scan point we evaluate

    d^2 sigma^{lE lEB} / (d m_{tt} d cos theta_Z)
        = N_c * m_{tt} * beta_1(m_{tt}^2) * beta_2(m_{tt}^2)
                / (1024 pi^4 * s)
                * Tr(R_int^{lE lEB})                            [GeV^-3]

with R_int the inner-angle integral that this script already accumulates,
and N_c = 3 the t-tbar QCD colour multiplicity (pyHELAS returns the
colour-stripped amplitude M_0; sum_{ij}|M^{ij}|^2 = N_c |M_0|^2; the spin
density matrix rho = R/Tr R is colour-independent, but the cross section
is not).  The unpolarised average uses (sigma^{+-} + sigma^{-+}) / 4.
Output is converted from GeV^-3 to pb/GeV with 1 GeV^-2 = 3.8937937e8 pb.

Usage:
    python3 psinteg.py
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

    Same form as in psinteg_all.py.  Returns 0 (rather than NaN) for
    kinematics infinitesimally below threshold so that boundary points
    don't poison the cross-section evaluation.
    """
    beta_sq = (1.0
               - 2.0 * (m1**2 + m2**2) / M**2
               +       (m1**2 - m2**2)**2 / M**4)
    return sqrt(max(0.0, beta_sq))


# 1 GeV^-2 = 3.8937937e8 pb  (PDG (hbar c)^2).  Used to convert the
# differential cross section from GeV^-3 to pb/GeV (cos theta_Z is
# dimensionless, so per-cos-theta_Z carries no units).
GEV_INV2_TO_PB = 3.8937937e8

# QCD colour multiplicity of the t-tbar pair.  See module docstring.
NC = 3


######################################################
# Setup
######################################################

rs     = 1000
eps = 10**-4
mttmin = 2*mt*(1 + eps)
mttmax = (rs - mZ)*(1 - eps)

# Outer scan over (m12, cos theta_Z).
nx, ny = 60, 60

# Inner-integral quadrature parameters.  16 x 32 ~ 500 nodes typically gives
# 6-7 digits on a smooth (cth1, ph) integrand.  Bump these if you want more
# precision; both rules converge much faster than uniform MC.
ncth1 = 16        # Gauss-Legendre nodes in cth1 in [-1, 1]
nph   = 16        # Trapezoid nodes in ph    in [0, 2*pi)

# Beam four-momenta at fixed sqrt(s).
pp = {'E':  np.array([rs/2.0, 0.0, 0.0,  rs/2.0]),
      'EB': np.array([rs/2.0, 0.0, 0.0, -rs/2.0])}

# Quadrature nodes and weights.  Both axes are flattened into a single 1-D
# loop later via np.repeat / np.tile + np.outer for the weights.
cth1_nodes, cth1_w = leggauss(ncth1)                            # [-1, 1]
ph_nodes           = np.linspace(0.0, 2*pi, nph, endpoint=False)
ph_w               = np.full(nph, 2*pi / nph)
quad_w             = np.outer(cth1_w, ph_w).ravel()             # length ncth1*nph
cth1_grid          = np.repeat(cth1_nodes, nph)
ph_grid            = np.tile(ph_nodes, ncth1)

# Bookkeeping for the helicity sums.
#   12 final-state helicities (lT, lTB, lZ) = 2 * 2 * 3.
#   Only the two Drell-Yan-allowed initial helicities survive in the
#   massless-electron limit; we keep both pure polarisations separately.
HEL_FINAL = [(lT, lTB, lZ)
             for lT  in (1, -1)
             for lTB in (1, -1)
             for lZ  in (1, 0, -1)]
INI_KEEP  = [(+1, -1), (-1, +1)]

# cth1 and ph have been integrated out, so they are no longer per-point
# variables; we only record the OUTER scan inputs.
input_keys = ['rs', 'm12', 'th3']

# Differential-cross-section columns (computed from R_int at each scan
# point; see the module docstring for the formula).  Stored in pb/GeV.
xsec_keys  = ['dsig_eR_pbperGeV',
              'dsig_eL_pbperGeV',
              'dsig_unpol_pbperGeV']


######################################################
# Scan
######################################################

for mode in [0]:

    print(f"mode = {mode}", flush=True)

    records = {k: [] for k in input_keys + xsec_keys}
    Rpol1   = []   # one 12x12 integrated R per (m12, cth3) for (+1, -1)
    Rpol2   = []   # ...                                          (-1, +1)

    if mode == 0:                            # outer scan: x = m12, y = cth3
        xar = np.linspace(mttmin, mttmax, nx)
        yar = np.linspace(-1+eps, 1-eps, ny)

    n_xdm    = len(xar)
    t_total0 = time.perf_counter()

    for ixdm, xdm in enumerate(xar):

        t_step0 = time.perf_counter()

        for ydm in yar:

            if mode == 0:
                m12  = xdm
                cth3 = ydm

            # Accumulate the integrated R for each kept (lE, lEB).
            R_int = {ini: np.zeros((12, 12), dtype=np.complex128)
                     for ini in INI_KEEP}

            # Single pass over the flattened (cth1, ph) node grid.  The
            # kinematics depend ONLY on (cth1, ph), not on (lE, lEB), so we
            # build the four-momenta once per node and reuse them for both
            # polarisations.
            for k in range(quad_w.size):
                cth1 = cth1_grid[k]
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

            # Differential cross section at this (m12, cth3) point.
            #   d^2 sigma / (d m_tt d cos theta_Z)
            #       = N_c * m_tt * beta_1 * beta_2 / (1024 pi^4 s) * Tr(R_int)
            # in GeV^-3; multiply by GEV_INV2_TO_PB to get pb/GeV (cos
            # theta_Z is dimensionless).  N_c = 3 is the t-tbar QCD colour
            # multiplicity, NOT included in the colour-stripped R-matrix.
            beta1     = get_beta(rs,  m12, mZ)        # sqrt(s) -> ttbar + Z
            beta2     = get_beta(m12, mt,  mt)        # ttbar   -> t + tbar
            prefactor = (NC * m12 * beta1 * beta2
                         / (1024.0 * pi**4 * rs**2)
                         * GEV_INV2_TO_PB)
            tr_eR  = np.trace(R_int[(+1, -1)]).real
            tr_eL  = np.trace(R_int[(-1, +1)]).real
            dsig_eR    = prefactor * tr_eR
            dsig_eL    = prefactor * tr_eL
            dsig_unpol = 0.25 * (dsig_eR + dsig_eL)

            # Store this outer scan point.
            if mode == 0:
                records['rs'].append(rs)
                records['m12'].append(m12)
                records['th3'].append(acos(cth3))
                records['dsig_eR_pbperGeV'].append(dsig_eR)
                records['dsig_eL_pbperGeV'].append(dsig_eL)
                records['dsig_unpol_pbperGeV'].append(dsig_unpol)
                Rpol1.append(R_int[(+1, -1)].copy())
                Rpol2.append(R_int[(-1, +1)].copy())

        t_step  = time.perf_counter() - t_step0
        t_total = time.perf_counter() - t_total0
        eta     = t_total / (ixdm + 1) * (n_xdm - ixdm - 1)
        print(f"  IXDM={ixdm+1:>2d}/{n_xdm}   XDM={xdm:8.2f}   "
              f"step = {t_step:8.2f} s   total = {t_total:8.2f} s   "
              f"ETA = {eta:8.2f} s",
              flush=True)

    ######################################################
    # Save
    ######################################################
    if rs == 1000:
        rs_str = 1
    if rs == 500:
        rs_str = 0.5
    if rs == 600:
        rs_str = 0.6

    out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            f'psinteg_res_{rs_str}.npz')
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    np.savez_compressed(
        out_path,
        R1=np.array(Rpol1, dtype=np.complex128),
        R2=np.array(Rpol2, dtype=np.complex128),
        **{k: np.array(v) for k, v in records.items()},
    )
    print(f"Saved {len(Rpol1)} (m12, cth3) points to {out_path}")
    print(f"  Quadrature: {ncth1} GL (cth1) x {nph} trap (ph) "
          f"= {ncth1*nph} nodes per outer point")
