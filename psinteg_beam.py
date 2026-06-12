"""
Beam-basis variant of psinteg.py.

Same 2-D outer scan over (m_tt, cos theta_Z) with inner integration over
(cos theta_t, phi), keeping the two pure-helicity beam polarisations
(e_R e_L and e_L e_R) separate.

Difference from psinteg.py
--------------------------
At each (cth1, phi) quadrature node, after pyHELAS produces the 12
helicity-basis amplitudes M_hel(lT, lTB, lZ), we apply a per-event spin
rotation

      M_beam = (U_t  x  U_tbar  x  U_Z) @ M_hel

where U_i is the unitary that takes pyHELAS's helicity eigenstate at the
particle's direction p_hat_i to the spin eigenstate along the lab z-axis
(the beam direction).  The resulting integrated R-matrix per outer
(m_tt, cos theta_Z) point is therefore in the BEAM BASIS.

In the beam basis the (cth1, phi)-integrated R is expected to be CP-symmetric
under the simple cp_map of cp_check.py, so the t/tbar QI asymmetries seen in
the helicity-basis plots are expected to vanish here.

Output file (one per `mode`):
    data/psinteg_beam_res_<mode>.npz
        R1   : (N, 12, 12) complex   integrated R for (lE, lEB) = (+1, -1)
        R2   : (N, 12, 12) complex   integrated R for (lE, lEB) = (-1, +1)
        rs   : (N,)        float     sqrt(s)
        m12  : (N,)        float     m_{t tbar}
        th3  : (N,)        float     acos(cos theta_Z)

Usage:
    python3 psinteg_beam.py
"""

import os, sys
import time
sys.path.append('/Users/kazuki/Projects/pyHELAS')
import numpy as np
from math import acos, sqrt, pi
from numpy.polynomial.legendre import leggauss

from QI_functions import *
from ee_ttz import *
from ee_ttz_func import *


# ----------------------------------------------------------------------
# Helicity -> beam-basis spin rotations, derived from pyHELAS conventions.
# Identical to psinteg_all_beam.py; kept inline so this script is
# self-contained.
# ----------------------------------------------------------------------

def _angles_from_p(p):
    """Return (theta, phi) of the 3-momentum (lab frame) given a 4-vector
    p = (E, px, py, pz). pmag == 0 -> returns (0, 0)."""
    px, py, pz = p[1], p[2], p[3]
    pmag = sqrt(px*px + py*py + pz*pz)
    if pmag == 0.0:
        return 0.0, 0.0
    theta = np.arccos(np.clip(pz / pmag, -1.0, 1.0))
    phi   = np.arctan2(py, px)
    return theta, phi


def U_half(p):
    """U^{(1/2)}(p_hat): 2x2 unitary from pyHELAS helicity basis at p_hat
    to spin-z basis along +z.  Helicity index order: (+1, -1)."""
    theta, phi = _angles_from_p(p)
    c = np.cos(theta / 2.0)
    s = np.sin(theta / 2.0)
    ep = np.exp(1j * phi)
    em = np.exp(-1j * phi)
    return np.array([
        [c,        -em * s],
        [ep * s,    c     ],
    ], dtype=np.complex128)


def U_one(p):
    """U^{(1)}(p_hat): 3x3 unitary from pyHELAS helicity basis at p_hat
    to spin-z basis along +z (boost factor on the longitudinal mode divided
    out to make U unitary).  Helicity index order: (+1, 0, -1)."""
    theta, phi = _angles_from_p(p)
    cth = np.cos(theta)
    sth = np.sin(theta)
    c2  = (1.0 + cth) / 2.0          # cos^2(theta/2)
    s2  = (1.0 - cth) / 2.0          # sin^2(theta/2)
    inv = 1.0 / sqrt(2.0)
    ep  = np.exp( 1j * phi)
    em  = np.exp(-1j * phi)
    return np.array([
        [c2 * ep,   -sth * inv * ep,   s2 * ep ],
        [sth * inv,  cth,             -sth * inv],
        [s2 * em,    sth * inv * em,   c2 * em ],
    ], dtype=np.complex128)


def U_total(pT, pTB, pZ):
    """Tensor product U_t (x) U_tbar (x) U_Z, a 12x12 unitary acting on the
    helicity-basis amplitude vector to give the beam-basis one."""
    return np.kron(np.kron(U_half(pT), U_half(pTB)), U_one(pZ))


######################################################
# Setup
######################################################

rs     = 1000.0
mttmin = 2*mt
mttmax = rs - mZ

# Outer scan over (m12, cos theta_Z).
nx, ny = 60, 60

# Inner-integral quadrature parameters.
ncth1 = 20        # Gauss-Legendre nodes in cth1 in [-1, 1]
nph   = 20        # Trapezoid nodes in ph    in [0, 2*pi)

# Beam four-momenta at fixed sqrt(s).
pp = {'E':  np.array([rs/2.0, 0.0, 0.0,  rs/2.0]),
      'EB': np.array([rs/2.0, 0.0, 0.0, -rs/2.0])}

# Quadrature nodes and weights, flattened into a 1-D index over (cth1, ph).
cth1_nodes, cth1_w = leggauss(ncth1)
ph_nodes           = np.linspace(0.0, 2*pi, nph, endpoint=False)
ph_w               = np.full(nph, 2*pi / nph)
quad_w             = np.outer(cth1_w, ph_w).ravel()
cth1_grid          = np.repeat(cth1_nodes, nph)
ph_grid            = np.tile(ph_nodes, ncth1)

# Helicity bookkeeping.
HEL_FINAL = [(lT, lTB, lZ)
             for lT  in (1, -1)
             for lTB in (1, -1)
             for lZ  in (1, 0, -1)]
INI_KEEP  = [(+1, -1), (-1, +1)]

# cth1 and ph have been integrated out; only the OUTER scan inputs are saved.
input_keys = ['rs', 'm12', 'th3']


######################################################
# Scan
######################################################

for mode in [0]:

    print(f"BEAM-BASIS mode = {mode}", flush=True)

    records = {k: [] for k in input_keys}
    Rpol1   = []
    Rpol2   = []

    if mode == 0:
        xar = np.linspace(mttmax, mttmin, nx)
        yar = np.linspace(-1, 1, ny)

    n_xdm    = len(xar)
    t_total0 = time.perf_counter()

    for ixdm, xdm in enumerate(xar):

        t_step0 = time.perf_counter()

        for ydm in yar:

            if mode == 0:
                m12  = xdm
                cth3 = ydm

            # Accumulate the integrated R (in beam basis) for each (lE, lEB).
            R_int = {ini: np.zeros((12, 12), dtype=np.complex128)
                     for ini in INI_KEEP}

            for k in range(quad_w.size):
                cth1 = cth1_grid[k]
                ph   = ph_grid[k]
                w    = quad_w[k]

                p1, p2, p3 = get_momenta(rs, m12, mt, mZ, cth3, cth1, ph)
                pp['T']  = np.array([p1.E, p1.px, p1.py, p1.pz])
                pp['TB'] = np.array([p2.E, p2.px, p2.py, p2.pz])
                pp['Z']  = np.array([p3.E, p3.px, p3.py, p3.pz])

                # Per-event helicity -> beam-basis rotation (12x12 unitary).
                Ut = U_total(pp['T'], pp['TB'], pp['Z'])

                for (lE, lEB) in INI_KEEP:
                    amp_hel = np.array([
                        get_amphel(pp,
                                   {'E': lE, 'EB': lEB,
                                    'T':  lT, 'TB': lTB, 'Z': lZ}).sum()
                        for (lT, lTB, lZ) in HEL_FINAL
                    ], dtype=np.complex128)
                    amp_beam = Ut @ amp_hel
                    R_int[(lE, lEB)] += w * np.outer(amp_beam,
                                                      amp_beam.conj())

            if mode == 0:
                records['rs'].append(rs)
                records['m12'].append(m12)
                records['th3'].append(acos(cth3))
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
    out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            'data', f'psinteg_beam_res_{mode}.npz')
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    np.savez_compressed(
        out_path,
        R1=np.array(Rpol1, dtype=np.complex128),
        R2=np.array(Rpol2, dtype=np.complex128),
        **{k: np.array(v) for k, v in records.items()},
    )
    print(f"Saved {len(Rpol1)} (m12, cth3) points to {out_path}", flush=True)
    print(f"  Quadrature: {ncth1} GL (cth1) x {nph} trap (ph) "
          f"= {ncth1*nph} nodes per outer point", flush=True)
