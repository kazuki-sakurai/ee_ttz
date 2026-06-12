"""
Phase-space integration of the e+ e- -> t tbar Z spin density matrix over
the entire 4-D internal phase space (m_tt, cos theta_Z, cos theta_t, phi),
keeping the two pure-helicity beam polarisations (e_R e_L and e_L e_R)
separate.  The result is a 12x12 integrated R-matrix per polarisation, per
sqrt(s).  An outer scan over sqrt(s) is performed.

Phase-space measure
-------------------
The Lorentz-invariant 3-body phase space, after dropping the trivial overall
lab-frame azimuth (factor 2 pi) and after introducing the t tbar invariant
mass m_tt, factorises as

    dPhi_3 ~ beta1(m_tt^2) * beta2(m_tt^2)
             * d(m_tt^2) * d(cos theta_Z) * d(cos theta_t) * d(phi)

where
    beta1 = sqrt(lambda(s, m_tt^2, m_Z^2)) / s    (sqrt(s) -> ttbar + Z)
    beta2 = sqrt(lambda(m_tt^2, m_t^2, m_t^2)) / m_tt^2
                                                  (ttbar -> t + tbar)
and lambda is the Kallen function.  The Lorentz-invariant prefactors
((32 pi^2)^-2, 1/(2 pi), 1/(2 s)) are dropped throughout because they cancel
in the trace normalisation rho = R / Tr R applied downstream.

Integration: tensor product of
    Gauss-Legendre on m_tt^2  in [(2 m_t)^2, (sqrt(s) - m_Z)^2]
    Gauss-Legendre on cth_Z   in [-1, 1]
    Gauss-Legendre on cth_t   in [-1, 1]
    composite trapezoid on phi in [0, 2 pi)        (periodic -> spectral)

Tunable parameters at the top of this file:
    rsmin, rsmax, nrs    : sqrt(s) scan endpoints and number of samples
    nm12, ncth3, ncth1, nph : node counts for the four phase-space directions

Output: data/psinteg_all_res.npz
    R1             : (Nrs, 12, 12) complex   for (lE, lEB) = (+1, -1)  [e_R e_L]
    R2             : (Nrs, 12, 12) complex   for (lE, lEB) = (-1, +1)  [e_L e_R]
    rs             : (Nrs,)        float
    sigma_eR_pb    : (Nrs,)        float     polarised sigma^{+-}        [pb]
    sigma_eL_pb    : (Nrs,)        float     polarised sigma^{-+}        [pb]
    sigma_unpol_pb : (Nrs,)        float     unpolarised total sigma     [pb]

Cross sections
--------------
The integrated R-matrices are constructed without the constant prefactor
1/(1024 pi^4) of the phase-space measure (eq 2.13 of the draft note); that
prefactor already absorbs the trivial int d phi_Z = 2 pi.  The R-matrices
also do NOT include the t-tbar colour multiplicity: pyHELAS returns the
colour-stripped amplitude M_0 (the colour structure is delta^{ij}, with
M^{ij} = delta^{ij} M_0), and the spin density matrix rho = R/Tr R is
colour-independent.  Summing over the t-tbar colours then contributes an
overall N_c = 3 factor to the cross section.  For massless initial beams
the per-polarisation total cross section is therefore

    sigma^{lE lEB} = N_c * Tr(R_int^{lE lEB}) / (2048 pi^4 s)    [GeV^-2],

and the unpolarised cross section averages over the four (only two non-zero)
initial-helicity combinations:

    sigma_unpol = ( sigma^{+-} + sigma^{-+} ) / 4 .

The output arrays are converted to pb with (hbar c)^2 = 3.8937937e8 pb GeV^2.

Usage:
    python3 psinteg_all.py
"""

import os
import sys
import time
sys.path.append('/Users/kazuki/Projects/pyHELAS')
import numpy as np
from math import sqrt, pi
from numpy.polynomial.legendre import leggauss

from ee_ttz import *           # mt, mZ, ...
from ee_ttz_func import *      # get_momenta, get_amphel


def get_beta(M, m1, m2):
    """Two-body phase-space factor: sqrt(lambda(M^2, m1^2, m2^2)) / M^2.

    Returns 0 (rather than NaN) for kinematics infinitesimally below
    threshold so that boundary nodes don't contaminate the quadrature.
    """
    beta_sq = (1.0
               - 2.0 * (m1**2 + m2**2) / M**2
               +       (m1**2 - m2**2)**2 / M**4)
    return sqrt(max(0.0, beta_sq))


######################################################
# Tunable parameters
######################################################

# sqrt(s) scan
rsmin, rsmax = 450.0, 5000.0
nrs          = 30

# Quadrature node counts.  These defaults give ~5-digit precision on the
# smooth (m_tt^2, cth_Z, cth_t, phi) integrand.  Bump them for more
# precision; lower for faster but less-accurate runs.
nm12  = 12     # Gauss-Legendre on m_tt^2
ncth3 = 10     # Gauss-Legendre on cos theta_Z
ncth1 = 10     # Gauss-Legendre on cos theta_t
nph   = 12     # trapezoid on phi

######################################################
# Quadrature setup (sqrt(s)-independent pieces)
######################################################

# Angular integrals: built once and reused for every sqrt(s).
cth3_nodes, cth3_w = leggauss(ncth3)                      # on [-1, 1]
cth1_nodes, cth1_w = leggauss(ncth1)                      # on [-1, 1]
ph_nodes           = np.linspace(0.0, 2*pi, nph, endpoint=False)
ph_w               = np.full(nph, 2*pi / nph)

# m_tt^2 quadrature: Gauss-Legendre on the unit interval [-1, 1] that we
# affine-map onto [m12sq_min, m12sq_max] inside the sqrt(s) loop.
m12sq_unit_nodes, m12sq_unit_w = leggauss(nm12)

# Helicity bookkeeping.
HEL_FINAL = [(lT, lTB, lZ)
             for lT  in (1, -1)
             for lTB in (1, -1)
             for lZ  in (1, 0, -1)]                       # 12 combinations
INI_KEEP  = [(+1, -1), (-1, +1)]                          # DY-allowed pair


######################################################
# Scan
######################################################

input_keys   = ['rs']
records      = {k: [] for k in input_keys}
Rpol1, Rpol2 = [], []

rs_array = np.logspace(np.log10(rsmin), np.log10(rsmax), nrs)
n_nodes  = nm12 * ncth3 * ncth1 * nph
print(f"sqrt(s) scan: {nrs} points in [{rsmin:.0f}, {rsmax:.0f}] GeV",
      flush=True)
print(f"phase-space quadrature: {nm12} GL (m_tt^2) x {ncth3} GL (cth_Z) "
      f"x {ncth1} GL (cth_t) x {nph} trap (phi) = {n_nodes} nodes / sqrt(s)",
      flush=True)

t_total0 = time.perf_counter()

for irs, rs in enumerate(rs_array):

    t_step0 = time.perf_counter()

    # Beam four-momenta at this sqrt(s).
    pp = {'E':  np.array([rs/2.0, 0.0, 0.0,  rs/2.0]),
          'EB': np.array([rs/2.0, 0.0, 0.0, -rs/2.0])}

    # Affine-map the unit Gauss-Legendre rule onto [m12sq_min, m12sq_max].
    m12sq_min = (2.0 * mt)**2
    m12sq_max = (rs - mZ)**2
    half = 0.5 * (m12sq_max - m12sq_min)
    mid  = 0.5 * (m12sq_max + m12sq_min)
    m12sq_nodes = half * m12sq_unit_nodes + mid
    m12sq_w_arr = half * m12sq_unit_w

    R_int = {ini: np.zeros((12, 12), dtype=np.complex128) for ini in INI_KEEP}

    for i_m, m12sq in enumerate(m12sq_nodes):
        m12   = sqrt(m12sq)
        beta1 = get_beta(rs,  m12, mZ)        # sqrt(s) -> ttbar + Z
        beta2 = get_beta(m12, mt,  mt)        # ttbar   -> t + tbar
        # Bring the m_tt^2 weight and beta1 * beta2 inside the outermost loop
        # so the deeper loops only multiply by the angular weights.
        w_m = m12sq_w_arr[i_m] * beta1 * beta2

        for i_3, cth3 in enumerate(cth3_nodes):
            w_m3 = w_m * cth3_w[i_3]

            for i_1, cth1 in enumerate(cth1_nodes):
                w_m31 = w_m3 * cth1_w[i_1]

                for i_p, ph in enumerate(ph_nodes):
                    w = w_m31 * ph_w[i_p]

                    # Kinematics depend ONLY on (m12, cth3, cth1, ph),
                    # not on the initial polarisation -- build once, use for
                    # both kept (lE, lEB) values.
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


######################################################
# Total cross section (helicities summed)
######################################################
#
# See the module docstring for the derivation:
#   sigma^{lE lEB} = N_c * Tr(R_int^{lE lEB}) / (2048 pi^4 s)   [GeV^-2]
#   sigma_unpol    = ( sigma^{+-} + sigma^{-+} ) / 4
# The N_c = 3 factor is the t-tbar colour multiplicity: pyHELAS returns
# the colour-stripped amplitude M_0 (color structure delta^{ij}), so
# sum_{ij} |M^{ij}|^2 = N_c * |M_0|^2.  rho = R / Tr R is colour-independent,
# but the cross section is not.
# Conversion factor:  1 GeV^-2 = 3.8937937e8 pb  (PDG (hbar c)^2).

NC             = 3                  # number of QCD colours (t-tbar pair)
GEV_INV2_TO_PB = 3.8937937e8

R1_arr   = np.array(Rpol1, dtype=np.complex128)            # (Nrs, 12, 12)
R2_arr   = np.array(Rpol2, dtype=np.complex128)
rs_arr   = np.array(records['rs'])                         # (Nrs,)

trace_R1 = np.einsum('iaa->i', R1_arr).real                # (Nrs,)
trace_R2 = np.einsum('iaa->i', R2_arr).real

inv_factor      = NC / (2048.0 * np.pi**4 * rs_arr**2)     # N_c / (2 s * 1024 pi^4)
sigma_eR_GeV2   = inv_factor * trace_R1                    # (+1, -1) = e_R e_L
sigma_eL_GeV2   = inv_factor * trace_R2                    # (-1, +1) = e_L e_R
sigma_unpol_GV2 = 0.25 * (sigma_eR_GeV2 + sigma_eL_GeV2)

sigma_eR_pb    = sigma_eR_GeV2   * GEV_INV2_TO_PB
sigma_eL_pb    = sigma_eL_GeV2   * GEV_INV2_TO_PB
sigma_unpol_pb = sigma_unpol_GV2 * GEV_INV2_TO_PB

print()
print("Total cross sections (helicities summed)")
print(f"  {'sqrt(s) [GeV]':>13s}  {'sig(eR eL) [pb]':>17s}  "
      f"{'sig(eL eR) [pb]':>17s}  {'sig(unpol)  [pb]':>17s}")
for i in range(len(rs_arr)):
    print(f"  {rs_arr[i]:>13.2f}  {sigma_eR_pb[i]:>17.6e}  "
          f"{sigma_eL_pb[i]:>17.6e}  {sigma_unpol_pb[i]:>17.6e}")


######################################################
# Save
######################################################

out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        'data', 'psinteg_all_res.npz')
os.makedirs(os.path.dirname(out_path), exist_ok=True)
np.savez_compressed(
    out_path,
    R1=R1_arr,
    R2=R2_arr,
    sigma_eR_pb=sigma_eR_pb,
    sigma_eL_pb=sigma_eL_pb,
    sigma_unpol_pb=sigma_unpol_pb,
    **{k: np.array(v) for k, v in records.items()},
)
print(f"\nSaved {len(Rpol1)} sqrt(s) points to {out_path}")
