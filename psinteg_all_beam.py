"""
Beam-basis variant of psinteg_all.py.

Same 4-D phase-space quadrature as psinteg_all.py over (m_tt^2, cos theta_Z,
cos theta_t, phi), keeping the two pure-helicity beam polarisations (e_R e_L
and e_L e_R) separate, and an outer scan over sqrt(s).

Difference from psinteg_all.py
------------------------------
At each quadrature node, after pyHELAS produces the 12 helicity-basis
amplitudes M_hel(lT, lTB, lZ), we apply a per-event spin rotation

      M_beam = (U_t  x  U_tbar  x  U_Z) @ M_hel

where U_i is the unitary that takes pyHELAS's helicity eigenstate at the
particle's direction p_hat_i to the spin eigenstate along the lab z-axis
(the beam direction).  The rotated R-matrix

      R_beam(rs) = sum_(nodes) w_node * |M_beam><M_beam|

is therefore the spin density matrix in the BEAM BASIS (each particle's spin
axis = lab +z) instead of the helicity basis.

In the beam basis the integrated R is expected to be CP-symmetric under the
simple cp_map of cp_check.py (swap t<->tbar plus helicity-label flips), so
cp_check.py on the output should PASS, and the t/tbar QI asymmetries seen in
the helicity-basis plots are expected to vanish in this basis.

Output: data/psinteg_all_beam_res.npz   (same schema as psinteg_all_res.npz).

Usage:
    python3 psinteg_all_beam.py
"""

import os
import sys
import time
sys.path.append('/Users/kazuki/Projects/pyHELAS')
import numpy as np
from math import sqrt, pi
from numpy.polynomial.legendre import leggauss

from ee_ttz      import mt, mZ, get_amphel
from ee_ttz_func import get_momenta


# ----------------------------------------------------------------------
# Helpers: kinematics of the two-body splittings and the two unitary
# spin-rotations U^{(1/2)}(p) and U^{(1)}(p) from pyHELAS conventions.
# ----------------------------------------------------------------------

def get_beta(M, m1, m2):
    """beta = sqrt(lambda(M^2, m1^2, m2^2)) / M^2 (clamped to >= 0)."""
    beta_sq = (1.0
               - 2.0*(m1**2 + m2**2)/M**2
               +       (m1**2 - m2**2)**2/M**4)
    return sqrt(max(0.0, beta_sq))


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
    """U^{(1/2)}(p_hat): 2x2 unitary that maps pyHELAS's helicity eigenstates
    at direction p_hat to spin-z eigenstates along +z.

    Columns are pyHELAS's chi+(p_hat) and chi-(p_hat) expressed in the
    (m_z=+1, m_z=-1) basis.  Helicity ordering matches get_amphel:
    index 0 -> hel=+1, index 1 -> hel=-1.
    """
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
    """U^{(1)}(p_hat): 3x3 unitary that maps pyHELAS's helicity eigenstates
    of a spin-1 boson at direction p_hat to spin-z eigenstates along +z.

    Derived directly from pyHELAS's polarisation-vector convention
        eps^{m_z=+1, z_hat} = (-1, +i, 0)/sqrt(2),
        eps^{m_z=0,  z_hat} = (0, 0, 1),
        eps^{m_z=-1, z_hat} = (1, +i, 0)/sqrt(2),
    by decomposing the boosted lab-frame eps^{h, p_hat} (spatial part for
    transverse modes, spatial direction p_hat for the longitudinal mode) on
    that basis.  The (E/m) boost factor on the longitudinal mode has been
    divided out so that U is unitary (i.e. the rotation is the pure
    spin-space part).

    Helicity ordering matches get_amphel: index 0 -> hel=+1, index 1 -> hel=0,
    index 2 -> hel=-1.
    """
    theta, phi = _angles_from_p(p)
    cth = np.cos(theta)
    sth = np.sin(theta)
    c2  = (1.0 + cth) / 2.0          # cos^2(theta/2)
    s2  = (1.0 - cth) / 2.0          # sin^2(theta/2)
    inv = 1.0 / sqrt(2.0)
    ep  = np.exp( 1j * phi)
    em  = np.exp(-1j * phi)
    return np.array([
        # row m_z = +1
        [c2 * ep,   -sth * inv * ep,   s2 * ep ],
        # row m_z = 0
        [sth * inv,  cth,             -sth * inv],
        # row m_z = -1
        [s2 * em,    sth * inv * em,   c2 * em ],
    ], dtype=np.complex128)


def U_total(pT, pTB, pZ):
    """Tensor product U_t (x) U_tbar (x) U_Z, a 12x12 unitary acting on
    the helicity-basis amplitude vector to give the beam-basis one."""
    return np.kron(np.kron(U_half(pT), U_half(pTB)), U_one(pZ))


# ----------------------------------------------------------------------
# Tunable parameters (mirror psinteg_all.py)
# ----------------------------------------------------------------------

rsmin, rsmax = 450.0, 5000.0
nrs          = 30

nm12  = 12      # Gauss-Legendre on m_tt^2
ncth3 = 10      # Gauss-Legendre on cos theta_Z
ncth1 = 10      # Gauss-Legendre on cos theta_t
nph   = 12      # trapezoid on phi


# ----------------------------------------------------------------------
# Quadrature setup (sqrt(s)-independent pieces)
# ----------------------------------------------------------------------

cth3_nodes, cth3_w = leggauss(ncth3)
cth1_nodes, cth1_w = leggauss(ncth1)
ph_nodes           = np.linspace(0.0, 2*pi, nph, endpoint=False)
ph_w               = np.full(nph, 2*pi / nph)
m12sq_unit_nodes, m12sq_unit_w = leggauss(nm12)

HEL_FINAL = [(lT, lTB, lZ)
             for lT  in (1, -1)
             for lTB in (1, -1)
             for lZ  in (1, 0, -1)]
INI_KEEP  = [(+1, -1), (-1, +1)]


# ----------------------------------------------------------------------
# Scan
# ----------------------------------------------------------------------

input_keys   = ['rs']
records      = {k: [] for k in input_keys}
Rpol1, Rpol2 = [], []

rs_array = np.logspace(np.log10(rsmin), np.log10(rsmax), nrs)
n_nodes  = nm12 * ncth3 * ncth1 * nph
print(f"BEAM-BASIS sqrt(s) scan: {nrs} points in [{rsmin:.0f}, {rsmax:.0f}] GeV",
      flush=True)
print(f"phase-space quadrature: {nm12} GL (m_tt^2) x {ncth3} GL (cth_Z) "
      f"x {ncth1} GL (cth_t) x {nph} trap (phi) = {n_nodes} nodes / sqrt(s)",
      flush=True)

t_total0 = time.perf_counter()

for irs, rs in enumerate(rs_array):

    t_step0 = time.perf_counter()

    pp = {'E':  np.array([rs/2.0, 0.0, 0.0,  rs/2.0]),
          'EB': np.array([rs/2.0, 0.0, 0.0, -rs/2.0])}

    m12sq_min = (2.0 * mt)**2
    m12sq_max = (rs - mZ)**2
    half = 0.5 * (m12sq_max - m12sq_min)
    mid  = 0.5 * (m12sq_max + m12sq_min)
    m12sq_nodes = half * m12sq_unit_nodes + mid
    m12sq_w_arr = half * m12sq_unit_w

    R_int = {ini: np.zeros((12, 12), dtype=np.complex128) for ini in INI_KEEP}

    for i_m, m12sq in enumerate(m12sq_nodes):
        m12   = sqrt(m12sq)
        beta1 = get_beta(rs,  m12, mZ)
        beta2 = get_beta(m12, mt,  mt)
        w_m   = m12sq_w_arr[i_m] * beta1 * beta2

        for i_3, cth3 in enumerate(cth3_nodes):
            w_m3 = w_m * cth3_w[i_3]

            for i_1, cth1 in enumerate(cth1_nodes):
                w_m31 = w_m3 * cth1_w[i_1]

                for i_p, ph in enumerate(ph_nodes):
                    w = w_m31 * ph_w[i_p]

                    p1, p2, p3 = get_momenta(rs, m12, mt, mZ, cth3, cth1, ph)
                    pp['T']  = np.array([p1.E, p1.px, p1.py, p1.pz])
                    pp['TB'] = np.array([p2.E, p2.px, p2.py, p2.pz])
                    pp['Z']  = np.array([p3.E, p3.px, p3.py, p3.pz])

                    # Per-event helicity -> beam-basis rotation.
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

    records['rs'].append(rs)
    Rpol1.append(R_int[(+1, -1)].copy())
    Rpol2.append(R_int[(-1, +1)].copy())

    t_step  = time.perf_counter() - t_step0
    t_total = time.perf_counter() - t_total0
    eta     = t_total / (irs + 1) * (nrs - irs - 1)
    print(f"  IRS={irs+1:>2d}/{nrs}   RS={rs:8.2f} GeV   "
          f"step = {t_step:8.2f} s   total = {t_total:8.2f} s   "
          f"ETA = {eta:8.2f} s",
          flush=True)


# ----------------------------------------------------------------------
# Save
# ----------------------------------------------------------------------

out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        'data', 'psinteg_all_beam_res.npz')
os.makedirs(os.path.dirname(out_path), exist_ok=True)
np.savez_compressed(
    out_path,
    R1=np.array(Rpol1, dtype=np.complex128),
    R2=np.array(Rpol2, dtype=np.complex128),
    **{k: np.array(v) for k, v in records.items()},
)
print(f"Saved {len(Rpol1)} sqrt(s) points to {out_path}", flush=True)
