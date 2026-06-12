"""
Fiducial cross section + cross-section-weighted averaged density matrix
for e+ e- -> t tbar Z, with kinematic cuts on the (m_{t tbar}, cos theta_Z)
plane and three beam-polarisation settings.

What this script computes, for each sqrt(s) label:
    sigma_fid(s, setting)   [fb]      (1 pb = 1000 fb)
        = int int d^2 sigma^{setting} / (d m_tt d cos theta_Z) dm_tt dcos theta_Z
          over the cut region |cos theta_Z| < c_cut, m_tt < mtt_cut.
          The integrand dsigma in the npz is in pb/GeV; we multiply by 1000
          on the way out so all stored / printed sigma_fid values are fb.

    rho_avg(s, setting)     [12 x 12 complex, normalised]
        = ( int int dsigma * rho(m_tt, cos theta_Z) dm dcos ) / sigma_fid

Three beam settings (matching qi_from_psinteg_realpol.py conventions):
    unpol   (P_e-, P_e+) = ( 0.0,  0.0)
    setI    (P_e-, P_e+) = (+0.8, -0.3)
    setII   (P_e-, P_e+) = (-0.8, +0.3)

How
---
1. Load data/psinteg_res_<rs>.npz which contains, for an outer 60x60 scan
   in (m_tt, cos theta_Z),  R1, R2  (the two pure-helicity colour-stripped
   12x12 spin density matrices, integrated over the t-decay angles), and
   the per-helicity differential cross sections dsig_eR_pbperGeV,
   dsig_eL_pbperGeV in pb/GeV.
2. Pivot those flat (Npts,) arrays into a regular 2-D grid (Nm, Nc) by
   identifying the unique m_tt and cos(theta_Z) axis values.
3. For each beam setting build
        R_setting(m,c)  = w_+- * R1 + w_-+ * R2          (mix R's)
        dsig_setting(m,c) = w_+- * dsig_eR + w_-+ * dsig_eL
   with weights from eq (2.9) of the note (and (1/4, 1/4) for unpol).
4. Build bilinear RegularGridInterpolator's for the scalar dsig_setting
   and the (12, 12) matrix dsig_setting * rho_setting, evaluate them on
   a fine sub-grid that EXACTLY fills the cut box, then integrate via
   trapz over both axes.  Refinement is what gives sub-percent boundary
   accuracy without having to align the cut with the coarse grid.
5. Compute the full QI suite (purities, concurrences, log-negativities,
   GMN_HMG) on each rho_avg, matching the analysis.py /
   qi_from_psinteg_realpol.py definitions exactly.
6. Save fiducial_<rs>.npz and print a console summary.

Outputs (next to this script):
    fiducial_<rs>.npz
        Schema:
            rs                  : float  sqrt(s) [GeV]
            cuts                : dict   {'c_cut': ..., 'mtt_cut': ...}
            settings            : array  ['unpol', 'setI', 'setII']  (S=3)
            Pe_minus / Pe_plus  : (S,)   the beam polarisations
            sigma_fid           : (S,)   fiducial cross section [fb]
            rho_avg             : (S, 12, 12) complex   averaged density matrix
            <QI key>            : (S,)   purities, EN_*, GMN_HMG, ...

Usage
-----
    python3 fiducial_from_psinteg.py 0.6 1          # explicit labels
    python3 fiducial_from_psinteg.py                # auto-discover all
    python3 fiducial_from_psinteg.py 0.6 --no-qi    # skip the QI suite
                                                    # (saves only sigma_fid + rho_avg)
    python3 fiducial_from_psinteg.py 0.6 --no-save  # print only
"""

import glob
import os
import re
import sys
import time
import numpy as np
from math import sqrt

sys.path.append('/Users/kazuki/Projects/pyHELAS')
from QI_functions import (normalise, purity, concurrence,
                          log_neg_bip, log_negativity)


# ----------------------------------------------------------------------
# Per-rs cuts.  Keys are the rs LABELS used by psinteg.py's filenames
# (so '0.6' means sqrt(s) = 600 GeV, '1' means 1 TeV).  Edit freely.
# Add new labels here and the script will pick them up automatically.
# ----------------------------------------------------------------------
CUTS_BY_RS = {
    '1':   dict(c_cut=0.5, mtt_cut=420.0),
}


# ----------------------------------------------------------------------
# Polygon-shaped fiducial regions, as a complement to the rectangular
# (c_cut, mtt_cut) cuts above.  Each entry is one named region with:
#
#   'vertices'  : list of (m_tt [GeV], cos theta_Z) tuples that define
#                 the polygon, in any consistent (CCW or CW) order.  The
#                 last segment is implicit (vertex[-1] -> vertex[0]).
#   'rs_labels' : (optional) iterable of rs labels at which to evaluate
#                 the polygon.  Omit / set to None to apply at every rs.
#                 Useful when the polygon extends past the kinematic
#                 range of the smaller sqrt(s) values.
#
# Output per region: fiducial_<rs>_<region_name>.npz, with the same
# schema as fiducial_<rs>.npz (the rectangular case) so downstream
# consumers can use either uniformly.
#
# The N1-fiducial region below has its right-hand vertex at m_tt = 700
# GeV, which fits inside the rs=1 (sqrt(s)=1 TeV) data but lies past the
# rs=0.6 kinematic edge (~509 GeV); rs_labels restricts it to rs=1.
# ----------------------------------------------------------------------
POLYGON_REGIONS = {
    'N3-fiducial': dict(
        vertices=[
            (346.0,  0.7),
            (550.0,  0.7),
            (550.0, -0.7),
            (346.0, -0.7),
        ],
        rs_labels=['1'],
    ),
}


# ----------------------------------------------------------------------
# Beam-polarisation settings.  Order here = order in the output arrays.
# ----------------------------------------------------------------------
POL_SETTINGS = [
    dict(name='unpol', label='unpolarised',
         Pe_minus= 0.0, Pe_plus= 0.0),
    dict(name='setI',  label='(P_e-, P_e+) = (+0.8, -0.3)',
         Pe_minus=+0.8, Pe_plus=-0.3),
    dict(name='setII', label='(P_e-, P_e+) = (-0.8, +0.3)',
         Pe_minus=-0.8, Pe_plus=+0.3),
]


# ----------------------------------------------------------------------
# Refinement of the coarse 60x60 grid for the cut integration.  600x600
# gives ~1/10 of the coarse spacing in each direction, which is enough
# for sub-percent boundary error on the rectangular cut.  Bump for more
# accuracy, lower for speed; cost is O(N_fine_m * N_fine_c * 144) per
# setting.
# ----------------------------------------------------------------------
N_FINE_M = 600
N_FINE_C = 600


# ----------------------------------------------------------------------
# QI quantities to report on each rho_avg.  Same names as in
# qi_from_psinteg_realpol.py / analysis.py.  EN_x are the LOG-negativities
# computed directly via QI_functions.log_negativity / log_neg_bip; the
# matching N_x linear negativities are derived afterwards via
#     N = (2^{E^N} - 1) / 2          (note eq 4.9)
# in _derived_negativities() so that everything stored matches what
# plot_qi._add_derived() builds for the heatmaps.
#
# This matters for cross-checking with the plots: the heatmaps show N
# (linear), while EN can be as much as 2x larger numerically for a given
# state; comparing N_avg vs the heatmap is the apples-to-apples check.
# ----------------------------------------------------------------------
QI_ENTRIES = [
    'pure', 'pure1', 'pure2', 'pure3',
    'c12', 'EN12', 'EN13', 'EN23',
    'c1', 'c2', 'c3', 'EN1', 'EN2', 'EN3',
    'GMN_HMG',
]

# Derived linear-negativity columns, built from QI_ENTRIES.  Same naming
# convention as plot_qi._add_derived().
DERIVED_NEG_KEYS = ['N12', 'N13', 'N23', 'N1', 'N2', 'N3']
_EN_TO_N = {'EN12': 'N12', 'EN13': 'N13', 'EN23': 'N23',
            'EN1':  'N1',  'EN2':  'N2',  'EN3':  'N3'}


def _fmt_dt(seconds):
    seconds = max(0.0, float(seconds))
    h, rem = divmod(int(seconds), 3600)
    m, s   = divmod(rem, 60)
    return f"{h:d}:{m:02d}:{s:02d}" if h else f"{m:02d}:{s:02d}"


def _polarised_weights(name, Pe_minus, Pe_plus):
    """Joint initial-helicity probabilities.

    Pure helicities (massless electron, SM):
        P_{+-} = (1/4)(1 + P_e-)(1 - P_e+)        e_R e_L  (R1 channel)
        P_{-+} = (1/4)(1 - P_e-)(1 + P_e+)        e_L e_R  (R2 channel)
    These are the SAME formulas as qi_from_psinteg_realpol.py.

    Unpolarised case (P_e- = P_e+ = 0) reduces to (1/4, 1/4), which
    matches psinteg.py's dsig_unpol = (dsig_eR + dsig_eL) / 4.
    """
    P_pm = 0.25 * (1.0 + Pe_minus) * (1.0 - Pe_plus)
    P_mp = 0.25 * (1.0 - Pe_minus) * (1.0 + Pe_plus)
    return P_pm, P_mp


def _load_grid(rs_label, here):
    """Return (m_axis, c_axis, R1_2d, R2_2d, dsig_eR_2d, dsig_eL_2d) for
    `data/psinteg_res_<rs>.npz`, pivoted into a regular 2-D (Nm, Nc) grid.

    psinteg.py stores the outer scan as flat 1-D arrays of length
    Npts = Nm * Nc (with m_tt the outer loop and cos theta_Z the inner).
    We rebuild the 2-D layout by finding the unique m_tt and cos(theta_Z)
    values and reverse-mapping each row, which is robust regardless of
    the storage order (descending m, etc.) chosen by psinteg.py.
    """
    src = os.path.join(here, 'data', f'psinteg_res_{rs_label}.npz')
    if not os.path.isfile(src):
        raise SystemExit(
            f"missing {src}; run `python3 psinteg.py` for "
            f"sqrt(s) = {float(rs_label)*1000:g} GeV first."
        )
    d = np.load(src)
    R1   = d['R1']            # (Npts, 12, 12) complex
    R2   = d['R2']            # (Npts, 12, 12) complex
    m12  = d['m12'].astype(float)
    th3  = d['th3'].astype(float)        # radians
    if not ('dsig_eR_pbperGeV' in d.files and 'dsig_eL_pbperGeV' in d.files):
        raise SystemExit(
            f"{src} is missing dsig_eR_pbperGeV / dsig_eL_pbperGeV.  "
            f"Regenerate it with the up-to-date psinteg.py."
        )
    dsig_eR = d['dsig_eR_pbperGeV'].astype(float)
    dsig_eL = d['dsig_eL_pbperGeV'].astype(float)

    cth = np.cos(th3)

    # Unique axes, ascending.  Round to avoid float-equality misses.
    m_axis = np.unique(np.round(m12, 6))
    c_axis = np.unique(np.round(cth, 6))
    Nm, Nc = len(m_axis), len(c_axis)
    if Nm * Nc != len(m12):
        raise SystemExit(
            f"grid pivot failed for {src}: Nm={Nm}, Nc={Nc}, "
            f"Npts={len(m12)} (Nm*Nc != Npts).  Is this a 2-D outer scan?"
        )

    m_idx = np.searchsorted(m_axis, np.round(m12, 6))
    c_idx = np.searchsorted(c_axis, np.round(cth, 6))

    R1_2d      = np.empty((Nm, Nc, 12, 12), dtype=np.complex128)
    R2_2d      = np.empty((Nm, Nc, 12, 12), dtype=np.complex128)
    dsig_eR_2d = np.empty((Nm, Nc), dtype=float)
    dsig_eL_2d = np.empty((Nm, Nc), dtype=float)
    for k in range(len(m12)):
        i, j = m_idx[k], c_idx[k]
        R1_2d[i, j]      = R1[k]
        R2_2d[i, j]      = R2[k]
        dsig_eR_2d[i, j] = dsig_eR[k]
        dsig_eL_2d[i, j] = dsig_eL[k]
    return m_axis, c_axis, R1_2d, R2_2d, dsig_eR_2d, dsig_eL_2d


def _setting_fields(R1_2d, R2_2d, dsig_eR_2d, dsig_eL_2d, P_pm, P_mp):
    """Per-grid-point (Nm, Nc) and (Nm, Nc, 12, 12) integrands for ONE
    beam setting.

    Returns:
        dsig_2d        : (Nm, Nc)         dsigma_setting [pb/GeV]
        weighted_R_2d  : (Nm, Nc, 12, 12) dsigma_setting * rho_setting
                                          (integrand for the rho numerator)
    """
    R_set    = P_pm * R1_2d + P_mp * R2_2d                  # (Nm, Nc, 12, 12)
    dsig_set = P_pm * dsig_eR_2d + P_mp * dsig_eL_2d        # (Nm, Nc)

    # tr(R_set) over the last two axes -> (Nm, Nc), real to first order.
    tr_R = np.einsum('ijkk->ij', R_set).real

    # Guard against divide-by-zero at any sub-threshold corner cell where
    # both helicities give ~0.  Where tr_R = 0 the rho integrand is set
    # to 0 (so it doesn't contribute to the integral, and the matching
    # dsig is also 0 so the denominator is unaffected).
    safe_tr = np.where(np.abs(tr_R) > 0.0, tr_R, 1.0)
    rho_set = R_set / safe_tr[..., None, None]
    rho_set = np.where(np.abs(tr_R)[..., None, None] > 0.0, rho_set, 0.0)

    weighted_R_2d = dsig_set[..., None, None] * rho_set
    return dsig_set, weighted_R_2d


def _bilinear_grid(values_2d, x_axis, y_axis, x_fine, y_fine):
    """Bilinear interpolation of values_2d (shape (Nx, Ny, *trailing))
    onto the OUTER product of x_fine and y_fine.  Returns an array of
    shape (n_fine_x, n_fine_y, *trailing).

    Hand-rolled (vectorised) so the script has no scipy dependency.
    Both x_axis and y_axis must be ascending and strictly monotone.
    Query points outside the bracket are clipped (no extrapolation).
    """
    # Index of the left bracket on each axis, clipped to [0, N-2] so
    # i+1 is always valid.
    i = np.clip(np.searchsorted(x_axis, x_fine, side='right') - 1,
                0, len(x_axis) - 2)
    j = np.clip(np.searchsorted(y_axis, y_fine, side='right') - 1,
                0, len(y_axis) - 2)

    # Per-axis fractional distance into the cell, clipped to [0,1].
    tx = np.clip((x_fine - x_axis[i]) / (x_axis[i + 1] - x_axis[i]),
                 0.0, 1.0)               # (n_fine_x,)
    ty = np.clip((y_fine - y_axis[j]) / (y_axis[j + 1] - y_axis[j]),
                 0.0, 1.0)               # (n_fine_y,)

    # Gather the four corners using outer-product fancy indexing.
    I  = i[:, None]
    Ip = (i + 1)[:, None]
    J  = j[None, :]
    Jp = (j + 1)[None, :]

    v00 = values_2d[I,  J ]   # (n_fine_x, n_fine_y, *trailing)
    v10 = values_2d[Ip, J ]
    v01 = values_2d[I,  Jp]
    v11 = values_2d[Ip, Jp]

    # Broadcast tx, ty against the (n_fine_x, n_fine_y, *trailing) shape.
    tx_b = tx[:, None]
    ty_b = ty[None, :]
    extra = (None,) * (values_2d.ndim - 2)         # for trailing dims
    tx_b = tx_b[(slice(None), slice(None)) + extra]
    ty_b = ty_b[(slice(None), slice(None)) + extra]

    return ((1 - tx_b) * (1 - ty_b) * v00
            + tx_b     * (1 - ty_b) * v10
            + (1 - tx_b) * ty_b     * v01
            + tx_b     * ty_b       * v11)


def _integrate_cut(m_axis, c_axis, dsig_2d, weighted_R_2d,
                   c_cut, mtt_cut, n_fine_m, n_fine_c):
    """Bilinear-refined trapezoidal integration over the rectangular cut.

    Returns:
        sigma_fid   : float    pb
        rho_num     : (12,12)  complex  (= int int dsig * rho dm dc)
    """
    # Cut box, clipped to the actual data extent so we never extrapolate.
    m_lo = max(m_axis[0],  2.0 * 173.0)        # m_tt threshold ~ 2 m_t
    m_hi = min(m_axis[-1], mtt_cut)
    c_lo = max(c_axis[0],  -c_cut)
    c_hi = min(c_axis[-1], +c_cut)
    if m_hi <= m_lo or c_hi <= c_lo:
        raise SystemExit(
            f"empty cut region after clipping to data: "
            f"m_tt in [{m_lo:g}, {m_hi:g}], "
            f"cos theta_Z in [{c_lo:g}, {c_hi:g}].  "
            f"Tighten the cuts or regenerate the source npz with a "
            f"wider scan."
        )

    # Fine sub-grid that exactly fills the cut box.
    m_fine = np.linspace(m_lo, m_hi, n_fine_m)
    c_fine = np.linspace(c_lo, c_hi, n_fine_c)

    dsig_fine = _bilinear_grid(dsig_2d,        m_axis, c_axis,
                               m_fine, c_fine)              # (Nm', Nc')
    wR_fine_r = _bilinear_grid(weighted_R_2d.real, m_axis, c_axis,
                               m_fine, c_fine)              # (Nm', Nc', 12, 12)
    wR_fine_i = _bilinear_grid(weighted_R_2d.imag, m_axis, c_axis,
                               m_fine, c_fine)
    wR_fine   = wR_fine_r + 1j * wR_fine_i

    # 2-D trapezoid: integrate over cos theta_Z first, then m_tt.
    # (np.trapz is deprecated in NumPy 2.x in favour of np.trapezoid.)
    trap = getattr(np, 'trapezoid', np.trapz)
    sigma_fid = trap(trap(dsig_fine, c_fine, axis=1), m_fine, axis=0)
    rho_num   = trap(trap(wR_fine,   c_fine, axis=1), m_fine, axis=0)
    return float(sigma_fid), rho_num


def _polygon_mask(m_fine, c_fine, vertices):
    """Bool mask of shape (n_fine_m, n_fine_c) flagging fine-grid points
    that fall INSIDE the polygon defined by `vertices`, a list of
    (m_tt [GeV], cos theta_Z) tuples.

    Uses matplotlib.path.Path.contains_points (ray-casting with proper
    handling of horizontal edges); the polygon can be either CW or CCW
    oriented, and it can be concave.
    """
    from matplotlib.path import Path
    verts = np.asarray(vertices, dtype=float)
    Mf, Cf = np.meshgrid(m_fine, c_fine, indexing='ij')
    pts = np.column_stack([Mf.ravel(), Cf.ravel()])
    inside = Path(verts).contains_points(pts).reshape(Mf.shape)
    return inside


def _integrate_polygon(m_axis, c_axis, dsig_2d, weighted_R_2d,
                       vertices, n_fine_m, n_fine_c, name=''):
    """Bilinear-refined trapezoidal integration over a POLYGON region.

    Algorithm:
      1. Compute the axis-aligned bounding box of `vertices`, clipped to
         the actual data extent ([m_axis[0], m_axis[-1]] x
         [c_axis[0], c_axis[-1]]).
      2. Sample the integrand (scalar and matrix) onto a fine sub-grid
         that fills this box, using `_bilinear_grid`.
      3. Mask out the fine-grid points OUTSIDE the polygon (via
         matplotlib.path.Path.contains_points).  The bilinear refinement
         keeps the polygon edge-error to roughly one fine-cell width,
         which is sub-percent at 600x600.
      4. 2-D trapezoid integrate the masked integrand.

    If any polygon vertex lies outside the data bounding box, a warning
    is printed (the integral then covers polygon INTERSECT data box,
    NOT the literal polygon).

    Returns:
        sigma_fid : float    pb
        rho_num   : (12,12)  complex  (int int dsig * rho dm dc)
    """
    verts = np.asarray(vertices, dtype=float)
    m_v_lo, m_v_hi = float(verts[:, 0].min()), float(verts[:, 0].max())
    c_v_lo, c_v_hi = float(verts[:, 1].min()), float(verts[:, 1].max())

    # Polygon vertices vs data extent.  Warn if the polygon spills out.
    m_data_lo, m_data_hi = float(m_axis[0]), float(m_axis[-1])
    c_data_lo, c_data_hi = float(c_axis[0]), float(c_axis[-1])
    spills = (m_v_lo < m_data_lo or m_v_hi > m_data_hi
              or c_v_lo < c_data_lo or c_v_hi > c_data_hi)
    if spills:
        print(f"    ! polygon '{name}' extends beyond the data box: "
              f"polygon m_tt in [{m_v_lo:g}, {m_v_hi:g}] vs data in "
              f"[{m_data_lo:g}, {m_data_hi:g}] GeV, "
              f"polygon cos_theta_Z in [{c_v_lo:g}, {c_v_hi:g}] vs data "
              f"in [{c_data_lo:g}, {c_data_hi:g}].  Integral covers "
              f"polygon INTERSECT data box only.")

    m_lo = max(m_data_lo, m_v_lo)
    m_hi = min(m_data_hi, m_v_hi)
    c_lo = max(c_data_lo, c_v_lo)
    c_hi = min(c_data_hi, c_v_hi)
    if m_hi <= m_lo or c_hi <= c_lo:
        raise SystemExit(
            f"empty polygon region '{name}' after clipping to data: "
            f"m_tt in [{m_lo:g}, {m_hi:g}], "
            f"cos theta_Z in [{c_lo:g}, {c_hi:g}]")

    m_fine = np.linspace(m_lo, m_hi, n_fine_m)
    c_fine = np.linspace(c_lo, c_hi, n_fine_c)

    dsig_fine = _bilinear_grid(dsig_2d,        m_axis, c_axis,
                               m_fine, c_fine)              # (Nm', Nc')
    wR_fine_r = _bilinear_grid(weighted_R_2d.real, m_axis, c_axis,
                               m_fine, c_fine)              # (Nm', Nc', 12, 12)
    wR_fine_i = _bilinear_grid(weighted_R_2d.imag, m_axis, c_axis,
                               m_fine, c_fine)
    wR_fine   = wR_fine_r + 1j * wR_fine_i

    # Apply the polygon mask: zero the contribution outside the polygon.
    mask = _polygon_mask(m_fine, c_fine, vertices)
    dsig_fine = np.where(mask, dsig_fine, 0.0)
    wR_fine   = np.where(mask[..., None, None], wR_fine, 0+0j)

    trap = getattr(np, 'trapezoid', np.trapz)
    sigma_fid = trap(trap(dsig_fine, c_fine, axis=1), m_fine, axis=0)
    rho_num   = trap(trap(wR_fine,   c_fine, axis=1), m_fine, axis=0)
    return float(sigma_fid), rho_num


def _derived_negativities(q):
    """Add linear-negativity keys (N12, N13, N23, N1, N2, N3) to a QI dict
    `q` that already contains the corresponding EN_x log-negativities, via
    N = (2^{E^N} - 1) / 2  (note eq 4.9).

    Mirrors plot_qi._add_derived() so the values reported here are
    numerically identical to those drawn on the per-point heatmaps.
    """
    for en_key, n_key in _EN_TO_N.items():
        if en_key in q:
            q[n_key] = (2.0**q[en_key] - 1.0) / 2.0
    return q


def _rho_health_report(name, rho, tol_herm=1e-10, tol_psd=1e-9):
    """Print a one-line sanity check on rho_avg: Hermiticity, trace, and
    smallest eigenvalue.  Catches regressions in the cut integration or
    bilinear refinement that would silently corrupt rho_avg.
    """
    herm = float(np.max(np.abs(rho - rho.conj().T)))
    tr   = complex(np.trace(rho))
    rho_h = 0.5 * (rho + rho.conj().T)
    eigs  = np.linalg.eigvalsh(rho_h)
    flags = []
    if herm > tol_herm:        flags.append(f"non-Hermitian (|delta|={herm:.2e})")
    if abs(tr.real - 1) > 1e-9 or abs(tr.imag) > 1e-9:
        flags.append(f"Tr != 1 (Tr={tr.real:.6f}+{tr.imag:.1e}j)")
    if eigs.min() < -tol_psd:  flags.append(f"not PSD (min eig={eigs.min():.2e})")
    status = ', '.join(flags) if flags else 'OK'
    print(f"      rho_avg health [{name}]: "
          f"|herm|={herm:.1e}  Tr={tr.real:.6f}  "
          f"eig in [{eigs.min():.2e}, {eigs.max():.4f}]  -> {status}")


def _compute_qi(rho):
    """Same QI suite as qi_from_psinteg_realpol.compute_qi(), with the
    GMN_HMG / Julia import done lazily so users who pass --no-qi (or
    don't need GMN) don't pay the 30 s Julia warm-up.
    """
    from ppt_julia import gmn_hmg          # lazy import (Julia warm-up)
    q = {}
    q['pure'] = purity(rho)

    rt   = rho.reshape(2, 2, 3, 2, 2, 3)
    rho1 = np.einsum('xijxab->ijab', rt).reshape(6, 6)
    rho2 = np.einsum('ixjaxb->ijab', rt).reshape(6, 6)
    rho3 = np.einsum('ijxabx->ijab', rt).reshape(4, 4)

    q['pure1'] = purity(rho1)
    q['pure2'] = purity(rho2)
    q['pure3'] = purity(rho3)
    q['c1']    = sqrt(max(0.0, 2.0 * (1.0 - q['pure1'])))
    q['c2']    = sqrt(max(0.0, 2.0 * (1.0 - q['pure2'])))
    q['c3']    = sqrt(max(0.0, 2.0 * (1.0 - q['pure3'])))

    q['c12']  = concurrence(rho3)
    q['EN12'] = log_neg_bip(rho3, [2, 2])
    q['EN13'] = log_neg_bip(rho2, [2, 3])
    q['EN23'] = log_neg_bip(rho1, [2, 3])
    q['EN1']  = log_negativity(rho, 'A')
    q['EN2']  = log_negativity(rho, 'B')
    q['EN3']  = log_negativity(rho, 'C')

    q['GMN_HMG'] = gmn_hmg(rho)
    return q


def _settings_loop(region_spec, rs_GeV,
                   m_axis, c_axis, R1_2d, R2_2d,
                   dsig_eR_2d, dsig_eL_2d, do_qi):
    """Per-setting loop shared by the rectangular and polygon paths.

    `region_spec` is a dict with:
        type='rect'    -> uses c_cut, mtt_cut
        type='polygon' -> uses vertices (and an optional 'name' for warns)

    Returns the `results` dict; the caller decides where to save it.
    """
    results = dict(
        rs        = float(rs_GeV),
        settings  = np.array([s['name']     for s in POL_SETTINGS]),
        labels    = np.array([s['label']    for s in POL_SETTINGS]),
        Pe_minus  = np.array([s['Pe_minus'] for s in POL_SETTINGS]),
        Pe_plus   = np.array([s['Pe_plus']  for s in POL_SETTINGS]),
        sigma_fid = np.zeros(len(POL_SETTINGS), dtype=float),
        rho_avg   = np.zeros((len(POL_SETTINGS), 12, 12),
                             dtype=np.complex128),
        region_type = region_spec['type'],
        region_name = region_spec.get('name', ''),
    )
    if region_spec['type'] == 'rect':
        results['cuts'] = dict(c_cut=region_spec['c_cut'],
                               mtt_cut=region_spec['mtt_cut'])
    else:
        # Store vertices as a (N, 2) float array; numpy.savez can persist
        # this as a plain ndarray instead of a Python object.
        results['polygon_vertices'] = np.asarray(
            region_spec['vertices'], dtype=float)
    if do_qi:
        for k in QI_ENTRIES:
            results[k] = np.zeros(len(POL_SETTINGS), dtype=float)
        for k in DERIVED_NEG_KEYS:
            results[k] = np.zeros(len(POL_SETTINGS), dtype=float)

    for is_, setting in enumerate(POL_SETTINGS):
        P_pm, P_mp = _polarised_weights(
            setting['name'], setting['Pe_minus'], setting['Pe_plus'])
        dsig_2d, wR_2d = _setting_fields(
            R1_2d, R2_2d, dsig_eR_2d, dsig_eL_2d, P_pm, P_mp)

        if region_spec['type'] == 'rect':
            sigma_fid, rho_num = _integrate_cut(
                m_axis, c_axis, dsig_2d, wR_2d,
                region_spec['c_cut'], region_spec['mtt_cut'],
                N_FINE_M, N_FINE_C)
        else:
            sigma_fid, rho_num = _integrate_polygon(
                m_axis, c_axis, dsig_2d, wR_2d,
                region_spec['vertices'],
                N_FINE_M, N_FINE_C,
                name=region_spec.get('name', ''))

        # rho_avg = rho_num / Tr(rho_num).  In exact arithmetic Tr(rho_num)
        # equals sigma_fid, but the trace of the integrated matrix is more
        # numerically stable so we use it.
        trn = np.trace(rho_num).real
        if abs(trn) > 0.0:
            rho_avg = rho_num / trn
        else:
            rho_avg = np.zeros_like(rho_num)
        # Convert sigma_fid from pb (native unit from dsig_*_pbperGeV * dm
        # * dcos) to fb (1 pb = 1000 fb) so every consumer sees the same
        # unit.  rho_avg is dimensionless, no conversion needed.
        sigma_fid_fb = 1000.0 * sigma_fid
        results['sigma_fid'][is_] = sigma_fid_fb
        results['rho_avg'][is_]   = rho_avg

        print(f"    {setting['name']:<6s}  {setting['label']:<32s}  "
              f"weights (P_+-, P_-+) = ({P_pm:6.4f}, {P_mp:6.4f})  "
              f"sigma_fid = {sigma_fid_fb:.5g} fb")
        _rho_health_report(setting['name'], rho_avg)

        if do_qi:
            q = _compute_qi(rho_avg)
            q = _derived_negativities(q)         # add N1..N3, N12..N23
            for k in QI_ENTRIES:
                results[k][is_] = q[k]
            for k in DERIVED_NEG_KEYS:
                results[k][is_] = q[k]
    return results


def _save_results(results, out_path):
    """np.savez_compressed wrapper that handles the 'cuts' dict (which
    numpy can't store directly as a key) by serialising it to an object
    array.  All other keys are plain ndarrays or scalars."""
    payload = {}
    for k, v in results.items():
        if isinstance(v, dict):
            payload[k] = np.array(v, dtype=object)
        else:
            payload[k] = v
    np.savez_compressed(out_path, **payload)


def process_rs(rs_label, here, do_qi=True, do_save=True):
    """Full pipeline for one rs label, covering ALL configured regions:
        - the rectangular CUTS_BY_RS region (saved as fiducial_<rs>.npz)
        - every polygon in POLYGON_REGIONS whose rs_labels include
          rs_label   (saved as fiducial_<rs>_<region_name>.npz)

    Returns dict {region_name: results} where the rectangular key is
    the literal string 'rect' (for compatibility with the previous
    single-result return).
    """
    if rs_label not in CUTS_BY_RS:
        raise SystemExit(
            f"no cuts defined for rs label '{rs_label}'.  Edit "
            f"CUTS_BY_RS in fiducial_from_psinteg.py to add it.")
    cuts = CUTS_BY_RS[rs_label]
    c_cut, mtt_cut = cuts['c_cut'], cuts['mtt_cut']

    rs_GeV = float(rs_label) * 1000.0
    print(f"\n==> rs label '{rs_label}'  (sqrt(s) = {rs_GeV:g} GeV)")

    t0 = time.perf_counter()
    m_axis, c_axis, R1_2d, R2_2d, dsig_eR_2d, dsig_eL_2d = _load_grid(
        rs_label, here)
    print(f"    grid: Nm = {len(m_axis)}  (m_tt in "
          f"[{m_axis[0]:g}, {m_axis[-1]:g}] GeV)  "
          f"x  Nc = {len(c_axis)}  (cos theta_Z in "
          f"[{c_axis[0]:g}, {c_axis[-1]:g}])")
    print(f"    refining to {N_FINE_M} x {N_FINE_C} for the region integrals")

    # Build the list of regions to process for this rs.  Rectangular cut
    # always runs; polygon regions run only when rs_label is in their
    # rs_labels filter (or rs_labels is None / missing).
    region_jobs = []
    region_jobs.append((
        'rect',
        os.path.join(here, f'fiducial_{rs_label}.npz'),
        dict(type='rect', name='rect',
             c_cut=c_cut, mtt_cut=mtt_cut),
        f"|cos theta_Z| < {c_cut:g},  m_tt < {mtt_cut:g} GeV  (rectangular)",
    ))
    for name, region in POLYGON_REGIONS.items():
        rs_filter = region.get('rs_labels')
        if rs_filter is not None and rs_label not in rs_filter:
            continue
        # Filesystem-safe filename suffix: replace runs of non-alnum
        # characters with '_'.  Keeps 'N1-fiducial' intact except for
        # the hyphen, giving 'fiducial_<rs>_N1_fiducial.npz'.
        tag = re.sub(r'[^0-9A-Za-z]+', '_', name).strip('_')
        out = os.path.join(here, f'fiducial_{rs_label}_{tag}.npz')
        verts = region['vertices']
        v_summary = ', '.join(f"({m:g},{c:g})" for m, c in verts)
        region_jobs.append((
            name, out,
            dict(type='polygon', name=name, vertices=verts),
            f"polygon vertices = [{v_summary}]",
        ))

    all_results = {}
    for region_name, out_path, region_spec, region_desc in region_jobs:
        print()
        print(f"    region '{region_name}': {region_desc}")
        results = _settings_loop(
            region_spec, rs_GeV,
            m_axis, c_axis, R1_2d, R2_2d, dsig_eR_2d, dsig_eL_2d,
            do_qi=do_qi)
        all_results[region_name] = results
        if do_save:
            _save_results(results, out_path)
            print(f"    saved {out_path}")

    print(f"    elapsed: {_fmt_dt(time.perf_counter() - t0)}")
    return all_results


def _print_summary(rs_label, results, do_qi):
    """Compact console summary suitable for copy-paste into a slide.

    Works for both rectangular and polygon regions: the cuts / vertices
    line is built from `results['region_type']` and friends, written by
    _settings_loop().

    The QI columns use LINEAR negativity N = (2^{E^N} - 1) / 2 to match
    the per-point heatmaps produced by plot_qi.py / plot_psinteg_realpol.py
    (which plot N, not E^N).  Convexity of N guarantees
        N(rho_avg) <= max_i N(rho_i)
    inside the region, so a fiducial N strictly larger than the maximum
    per-point N on the plot would indicate a bug.
    """
    region_name = str(results.get('region_name', '')) or 'rect'
    print()
    print(f"========== summary  rs label '{rs_label}'  "
          f"sqrt(s) = {results['rs']:g} GeV  "
          f"region '{region_name}' ==========")
    rtype = str(results.get('region_type', 'rect'))
    if rtype == 'rect':
        cuts = results['cuts']
        # `cuts` may have been round-tripped through an object array.
        if hasattr(cuts, 'item') and getattr(cuts, 'shape', None) == ():
            cuts = cuts.item()
        print(f"cuts: |cos theta_Z| < {cuts['c_cut']:g},  "
              f"m_tt < {cuts['mtt_cut']:g} GeV  (rectangular)")
    else:
        verts = np.asarray(results['polygon_vertices'])
        v_summary = ', '.join(f"({m:g},{c:g})" for m, c in verts)
        print(f"polygon vertices: [{v_summary}]")
    print()
    print("Linear negativities N -- SAME UNITS as the per-point heatmaps "
          "(plot_qi shows N, not E^N).")
    hdr = f"{'setting':<7s} {'(P_e-, P_e+)':>14s} {'sigma_fid [fb]':>18s}"
    if do_qi:
        hdr += (f"  {'N1':>7s} {'N2':>7s} {'N3':>7s}"
                f" {'N12':>7s} {'N13':>7s} {'N23':>7s}"
                f" {'GMN_HMG':>9s}")
    print(hdr)
    print("-" * len(hdr))
    for is_, name in enumerate(results['settings']):
        Pe_m = results['Pe_minus'][is_]
        Pe_p = results['Pe_plus'][is_]
        row  = (f"{str(name):<7s} ({Pe_m:+4.1f}, {Pe_p:+4.1f}) "
                f"{results['sigma_fid'][is_]:18.6g}")
        if do_qi:
            row += (f"  {results['N1'][is_]:7.4f}"
                    f" {results['N2'][is_]:7.4f}"
                    f" {results['N3'][is_]:7.4f}"
                    f" {results['N12'][is_]:7.4f}"
                    f" {results['N13'][is_]:7.4f}"
                    f" {results['N23'][is_]:7.4f}"
                    f" {results['GMN_HMG'][is_]:9.4f}")
        print(row)
    print()
    if do_qi:
        print("(EN_x log-negativities are also saved in the npz, related "
              "by N = (2^{E^N} - 1) / 2; reproduce them with the "
              "EN1..EN3, EN12..EN23 columns of fiducial_<rs>.npz.)")
        print()


def _discover_rs_labels(here):
    pattern = os.path.join(here, 'data', 'psinteg_res_*.npz')
    rx = re.compile(r'psinteg_res_(?P<rs>.+)\.npz$')
    labels = []
    for p in glob.glob(pattern):
        m = rx.match(os.path.basename(p))
        if m:
            labels.append(m.group('rs'))

    def _key(s):
        try:
            return float(s)
        except ValueError:
            return float('inf')
    return sorted(set(labels), key=_key)


def main(argv):
    args    = [a for a in argv[1:] if not a.startswith('-')]
    flags   = [a for a in argv[1:] if a.startswith('-')]
    do_qi   = '--no-qi' not in flags
    do_save = '--no-save' not in flags

    here = os.path.dirname(os.path.abspath(__file__))

    if args:
        rs_labels = args
    else:
        rs_labels = _discover_rs_labels(here)
        if not rs_labels:
            raise SystemExit(
                f"no data/psinteg_res_*.npz files next to "
                f"{os.path.basename(__file__)}; run psinteg.py first.")
        # Keep only labels that have a cut entry, so we don't try to
        # process random scratch files like psinteg_res_TEST.npz.
        rs_labels = [r for r in rs_labels if r in CUTS_BY_RS]
        if not rs_labels:
            raise SystemExit(
                f"none of the discovered rs labels has an entry in "
                f"CUTS_BY_RS; add one (e.g. for '0.6' or '1') or pass "
                f"the rs label(s) on the command line.")
        print(f"Auto-discovered {len(rs_labels)} sqrt(s) "
              f"label(s) with cuts defined: {rs_labels}")

    all_results = {}
    for rs in rs_labels:
        res = process_rs(rs, here, do_qi=do_qi, do_save=do_save)
        all_results[rs] = res         # dict {region_name: results}

    # Summary tables at the end, all together (so they're easy to copy).
    # One table per (rs, region) pair, in the order they were processed.
    for rs in rs_labels:
        for region_name, results in all_results[rs].items():
            _print_summary(rs, results, do_qi)


if __name__ == '__main__':
    main(sys.argv)
