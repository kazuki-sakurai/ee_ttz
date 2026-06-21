"""
Differential cross section and HMG genuine multipartite negativity (GMN)
heatmaps on the (m_tt, cos theta_Z) plane, for the TWO realistic
ILC-like beam polarisation settings,

    setting i  :  (P_e-, P_e+) = (+0.8, -0.3)   ->  pol = 1
    setting ii :  (P_e-, P_e+) = (-0.8, +0.3)   ->  pol = 2

Run on one or more sqrt(s) labels (matching psinteg.py's file naming, e.g.
'0.5', '0.6', '1').  The unpolarised baseline is intentionally NOT
included here -- it already ships with qi_from_psinteg.py as pol = 3.

Why we read data/psinteg_res_<rs>.npz, not scan_results_psinteg_<rs>.npz
-----------------------------------------------------------------------
GMN is a non-linear functional of rho, so it cannot be obtained from the
GMN values that scan_results_psinteg_*.npz stores for pol = 1, 2, 3
(those are GMN of three specific density matrices, not of the realistic
mixtures we need).  We instead load the per-point pure-helicity
R-matrices R1, R2 (and the matching dsig_eR_pbperGeV, dsig_eL_pbperGeV
columns) and build the polarised density matrix point-by-point:

    R_pol = P_+- * R1  +  P_-+ * R2
    rho   = R_pol / tr(R_pol)
    GMN   = gmn_hmg(rho)                   # Julia/Mosek SDP via ppt_julia
    dsig  = P_+- * dsig_eR + P_-+ * dsig_eL

with the helicity weights from eq (2.9) of the draft note,

    P_{+,-} = (1/4)(1 + P_e-)(1 - P_e+)    # e_R e_L
    P_{-,+} = (1/4)(1 - P_e-)(1 + P_e+)    # e_L e_R

The other QI quantities (purities, concurrences, EN_*) are computed for
free off the same rho, so the resulting scan_results_psinteg_realpol_*.npz
is fully plot_qi.py-compatible.

Output
------
    scan_results_psinteg_realpol_<rs>.npz
        Same schema as scan_results_psinteg_<rs>.npz from qi_from_psinteg.py
        (rho + INPUT_KEYS + ENTRIES + dsigma_pbperGeV), but with pol
        restricted to {1, 2} and re-defined to mean the realistic
        settings above.

    qi_psinteg_realpol_<rs>_dsigma_fbperGeV.pdf
        Two-panel (one per setting) log-scale heatmap in fb/GeV, via
        plot_qi.py (which derives the fb column from dsigma_pbperGeV in
        _add_derived).

    qi_psinteg_realpol_<rs>_GMN_HMG.pdf
        Two-panel heatmap of the HMG GMN N_G.

Both PDFs are written in-process by importing plot_qi and overriding its
COL_TITLES so the panel headers read the realistic settings rather than
the default pure-helicity labels.

Usage
-----
    python3 qi_from_psinteg_realpol.py 0.6 1          # explicit labels
    python3 qi_from_psinteg_realpol.py                # auto-discover all
    python3 qi_from_psinteg_realpol.py 0.6 --no-plot  # only build the QI npz
    python3 qi_from_psinteg_realpol.py 0.6 --force    # regenerate the QI npz
                                                      # even if it already exists
"""

import glob
import os
import re
import sys
import time
import numpy as np
from math import sqrt

# pyHELAS / project-local helpers, same imports as analysis.py and
# qi_from_psinteg.py so the QI definitions stay in sync.
sys.path.append('/Users/kazuki/Projects/pyHELAS')
from QI_functions import (normalise, purity, concurrence,
                          log_neg_bip, log_negativity)
#from ppt_julia import gmn_hmg
from ppt_cvxpy import get_GMN as _get_GMN
def gmn_hmg(rho):
    return _get_GMN(rho, dims=[2, 2, 3])

# We drive plot_qi.py in-process (not via subprocess) so we can patch its
# COL_TITLES dict before each plot.  Importing has the side-effect of
# applying plot_qi's matplotlib rcParams; that's intentional.
import plot_qi


# ----------------------------------------------------------------------
# Beam polarisation settings.  These are the same two as in
# qi_from_psinteg_all.py (the 1-D vs sqrt(s) pipeline), so the panel
# labels stay consistent across both pipelines.
# ----------------------------------------------------------------------
POL_SETTINGS = {
    1: dict(label=r'$(\mathcal{P}_{e^-},\,\mathcal{P}_{e^+}) = (+0.8,\,-0.3)$',
            Pe_minus=+0.8, Pe_plus=-0.3),
    2: dict(label=r'$(\mathcal{P}_{e^-},\,\mathcal{P}_{e^+}) = (-0.8,\,+0.3)$',
            Pe_minus=-0.8, Pe_plus=+0.3),
}

# Same QI list as analysis.py / qi_from_psinteg.py, in the same order, so
# plot_qi.py treats the resulting file identically.
ENTRIES = [
    'pure', 'pure1', 'pure2', 'pure3',
    'c12', 'EN12', 'EN13', 'EN23',
    'c1', 'c2', 'c3', 'EN1', 'EN2', 'EN3',
    'GMN_HMG',
]
INPUT_KEYS = ['rs', 'm12', 'th3', 'cth1', 'ph', 'pol']

# Quantities we actually want plotted at the end.  Everything else in
# scan_results_psinteg_realpol_*.npz is computed for free (so it stays
# plot_qi-compatible) but is not drawn here.
PLOT_QUANTITIES = ('dsigma_fbperGeV', 'GMN_HMG')


def _fmt_dt(seconds):
    """Format an elapsed/remaining time in compact mm:ss or hh:mm:ss form.
    Used by the per-point progress lines below."""
    seconds = max(0.0, float(seconds))
    h, rem = divmod(int(seconds), 3600)
    m, s   = divmod(rem, 60)
    return f"{h:d}:{m:02d}:{s:02d}" if h else f"{m:02d}:{s:02d}"


def _polarised_weights(Pe_minus, Pe_plus):
    """Joint initial-helicity probabilities from eq (2.9) of the draft
    note, for the two SM-surviving combinations (the (+,+) and (-,-)
    terms vanish in the massless-electron SM and are not returned).
    """
    P_pm = 0.25 * (1.0 + Pe_minus) * (1.0 - Pe_plus)   # e_R e_L weight
    P_mp = 0.25 * (1.0 - Pe_minus) * (1.0 + Pe_plus)   # e_L e_R weight
    return P_pm, P_mp


def compute_qi(rho):
    """All QI quantities for one 12x12 rho on subsystems (t, tbar, Z)
    with Hilbert-space dimensions (2, 2, 3).  Identical to the block in
    qi_from_psinteg_all.py / qi_from_psinteg.py.
    """
    q = {}
    q['pure'] = purity(rho)

    rt   = rho.reshape(2, 2, 3, 2, 2, 3)
    rho1 = np.einsum('xijxab->ijab', rt).reshape(6, 6)   # rho_{tbar Z}
    rho2 = np.einsum('ixjaxb->ijab', rt).reshape(6, 6)   # rho_{t Z}
    rho3 = np.einsum('ijxabx->ijab', rt).reshape(4, 4)   # rho_{t tbar}

    q['pure1'] = purity(rho1)
    q['pure2'] = purity(rho2)
    q['pure3'] = purity(rho3)

    # Clamp to >= 0 against floating-point noise (purity ~ 1 + epsilon for
    # nearly pure global states would otherwise crash math.sqrt).
    q['c1'] = sqrt(max(0.0, 2.0 * (1.0 - q['pure1'])))
    q['c2'] = sqrt(max(0.0, 2.0 * (1.0 - q['pure2'])))
    q['c3'] = sqrt(max(0.0, 2.0 * (1.0 - q['pure3'])))

    q['c12']  = concurrence(rho3)
    q['EN12'] = log_neg_bip(rho3, [2, 2])
    q['EN13'] = log_neg_bip(rho2, [2, 3])
    q['EN23'] = log_neg_bip(rho1, [2, 3])

    q['EN1'] = log_negativity(rho, 'A')
    q['EN2'] = log_negativity(rho, 'B')
    q['EN3'] = log_negativity(rho, 'C')

    q['GMN_HMG'] = gmn_hmg(rho)
    return q


def build_qi_file(rs_label, here):
    """Build scan_results_psinteg_realpol_<rs>.npz from
    data/psinteg_res_<rs>.npz, doing the realistic-polarisation mixing
    and the full QI suite per (point, setting).  Returns the output path.
    """
    src = os.path.join(here, 'data', f'psinteg_res_{rs_label}.npz')
    if not os.path.isfile(src):
        raise SystemExit(
            f"missing {src}; run `python3 psinteg.py` with sqrt(s) = "
            f"{float(rs_label)*1000:g} GeV first."
        )
    d = np.load(src)

    R1   = d['R1']                          # (Npts, 12, 12) complex
    R2   = d['R2']                          # (Npts, 12, 12) complex
    Npts = R1.shape[0]

    # Outer-scan inputs.  psinteg.py records m12 (GeV) and th3 (angle in
    # radians; plot_qi takes the cosine via COL_INFO).  cth1 / ph are
    # integrated out so they're stored as NaN (plot_qi skips NaN
    # constants in its title strings).
    m12_arr = d['m12']
    th3_arr = d['th3']

    have_xs = ('dsig_eR_pbperGeV' in d.files
               and 'dsig_eL_pbperGeV' in d.files)
    if have_xs:
        dsig_eR = d['dsig_eR_pbperGeV']
        dsig_eL = d['dsig_eL_pbperGeV']
    else:
        dsig_eR = np.full(Npts, np.nan)
        dsig_eL = np.full(Npts, np.nan)
        print("  ! pure-pol differential cross sections NOT in source npz; "
              "dsigma_pbperGeV will be NaN.")
        print("    Regenerate data/psinteg_res_*.npz with the up-to-date "
              "psinteg.py to populate them.")

    rs_GeV = float(rs_label) * 1000.0       # '0.5' -> 500, '0.6' -> 600, '1' -> 1000

    print(f"Loaded {src}: {Npts} (m_tt, cos theta_Z) points, "
          f"sqrt(s) = {rs_GeV:g} GeV")

    # Pre-compute per-setting helicity weights once.
    weights = {}
    for pol_code, setting in POL_SETTINGS.items():
        Pe_m, Pe_p = setting['Pe_minus'], setting['Pe_plus']
        P_pm, P_mp = _polarised_weights(Pe_m, Pe_p)
        weights[pol_code] = (P_pm, P_mp)
        print(f"  pol {pol_code}: {setting['label']}  "
              f"-> weights (P_+-, P_-+) = ({P_pm:.4f}, {P_mp:.4f})")

    records = {k: [] for k in INPUT_KEYS + ENTRIES + ['dsigma_pbperGeV']}
    rho_all = []

    # ------------------------------------------------------------------
    # Progress reporting.  The HMG GMN SDP is the slow step (Julia/Mosek,
    # ~1-2 s per call) so we print a status line every PROGRESS_STEP
    # points along with running elapsed time and an ETA.  flush=True so
    # the output streams to disk/terminal immediately, which matters when
    # this script is piped through `tee` from the run_*.sh wrapper.
    #
    # Defaults to ~20 lines per polarisation setting; cap at one line per
    # SDP call for tiny grids and floor at 1 so % math doesn't blow up.
    # ------------------------------------------------------------------
    Nset          = len(POL_SETTINGS)
    Ntot          = Nset * Npts
    progress_step = max(1, Npts // 20)
    t_start       = time.perf_counter()
    done_total    = 0

    print(f"  progress: every {progress_step} point(s) per setting "
          f"(~{Ntot // progress_step // Nset + 1} updates per setting, "
          f"{Ntot} SDP calls in total)", flush=True)

    # Outer loop: polarisation setting.  Inner loop: scan points.  This
    # ordering matches qi_from_psinteg.py so the resulting npz row order
    # is identical, and load_scan.select(pol=...) keeps working.
    for pol_code in POL_SETTINGS:
        P_pm, P_mp = weights[pol_code]
        t_pol      = time.perf_counter()
        print(f"  pol = {pol_code} ({POL_SETTINGS[pol_code]['label']}) "
              f"-- starting {Npts} points ...", flush=True)
        for i in range(Npts):
            R_pol = P_pm * R1[i] + P_mp * R2[i]
            rho   = normalise(R_pol)
            q     = compute_qi(rho)

            records['rs'  ].append(rs_GeV)
            records['m12' ].append(float(m12_arr[i]))
            records['th3' ].append(float(th3_arr[i]))
            records['cth1'].append(np.nan)
            records['ph'  ].append(np.nan)
            records['pol' ].append(pol_code)
            records['dsigma_pbperGeV'].append(
                float(P_pm * dsig_eR[i] + P_mp * dsig_eL[i])
            )
            for e in ENTRIES:
                records[e].append(q[e])
            rho_all.append(rho)

            done_total += 1
            if ((i + 1) % progress_step == 0) or ((i + 1) == Npts):
                elapsed = time.perf_counter() - t_start
                rate    = done_total / max(elapsed, 1e-9)        # SDPs/sec
                eta     = (Ntot - done_total) / max(rate, 1e-9)  # seconds
                pct_pol = 100.0 * (i + 1) / Npts
                pct_tot = 100.0 * done_total / Ntot
                print(f"    pol={pol_code} {i+1:4d}/{Npts} "
                      f"({pct_pol:5.1f}% setting, "
                      f"{pct_tot:5.1f}% total)   "
                      f"elapsed {_fmt_dt(elapsed)}   "
                      f"ETA {_fmt_dt(eta)}",
                      flush=True)

        t_pol_dt = time.perf_counter() - t_pol
        print(f"  pol = {pol_code} done in {_fmt_dt(t_pol_dt)} "
              f"({Npts} points, {Npts/max(t_pol_dt,1e-9):.2f} pt/s)",
              flush=True)

    out_npz = os.path.join(
        here, f'scan_results_psinteg_realpol_{rs_label}.npz')
    np.savez_compressed(
        out_npz,
        rho=np.array(rho_all, dtype=np.complex128),
        **{k: np.array(v) for k, v in records.items()},
    )
    print(f"Saved {out_npz}  ({len(rho_all)} rows = "
          f"{Npts} points x {len(POL_SETTINGS)} settings)")
    return out_npz


def _override_panel_titles():
    """Re-label plot_qi.COL_TITLES so the two panels read the realistic
    beam settings instead of the pure-helicity defaults.  Idempotent.
    """
    for k, v in POL_SETTINGS.items():
        plot_qi.COL_TITLES[k] = v['label']


def run_plots(out_npz, rs_label, here):
    """Drive plot_qi.process_file in-process for each PLOT_QUANTITIES key.
    Both PDFs land next to this script (same convention as qi_from_psinteg.py).
    """
    _override_panel_titles()
    for q in PLOT_QUANTITIES:
        out_pdf = os.path.join(
            here, f'qi_psinteg_realpol_{rs_label}_{q}.pdf')
        plot_qi.process_file(out_npz, out_pdf, only_quantity=q)


def _discover_rs_labels(here):
    """Return every rs label for which data/psinteg_res_<rs>.npz exists,
    sorted numerically when possible (so '0.5' comes before '1')."""
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
    do_plot = '--no-plot' not in flags
    force   = ('--force' in flags) or ('-f' in flags)

    here = os.path.dirname(os.path.abspath(__file__))

    if args:
        rs_labels = args
    else:
        rs_labels = _discover_rs_labels(here)
        if not rs_labels:
            raise SystemExit(
                f"no data/psinteg_res_*.npz files next to "
                f"{os.path.basename(__file__)}; run psinteg.py first."
            )
        print(f"Auto-discovered {len(rs_labels)} sqrt(s) "
              f"label(s): {rs_labels}")

    for rs_label in rs_labels:
        print()
        print(f"==> processing rs label '{rs_label}'")
        out_npz = os.path.join(
            here, f'scan_results_psinteg_realpol_{rs_label}.npz')

        # Skip the (Julia/Mosek-heavy) QI build step when the npz is already
        # on disk; pass --force / -f to override and recompute from scratch.
        if os.path.exists(out_npz) and not force:
            print(f"Using existing {out_npz}  (pass --force to regenerate)")
        else:
            out_npz = build_qi_file(rs_label, here)

        if do_plot:
            run_plots(out_npz, rs_label, here)


if __name__ == '__main__':
    main(sys.argv)
