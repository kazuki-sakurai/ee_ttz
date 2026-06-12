"""
Compute every QI quantity from the (cos theta_t, phi)-integrated R-matrices
written by psinteg.py, and produce the PDF plots via plot_qi.py.

File naming
-----------
psinteg.py labels its outputs by the sqrt(s) value it ran at (e.g.
data/psinteg_res_1.npz for sqrt(s) = 1 TeV, data/psinteg_res_0.5.npz for
sqrt(s) = 500 GeV).  This script propagates that label through to every
downstream output: scan_results_psinteg_<rs>.npz and qi_from_psinteg_<rs>.pdf.

Pipeline (one full pass per sqrt(s) label):
    1. Load data/psinteg_res_<rs>.npz    (R1, R2, the outer-scan inputs,
       and -- if present -- dsig_eR_pbperGeV, dsig_eL_pbperGeV,
       dsig_unpol_pbperGeV written by the up-to-date psinteg.py).
    2. For every (outer point) x (polarisation) build the 12x12 spin density
       matrix rho = R / Tr(R), where
            pol = 1  ->  R1                (e_R e_L)
            pol = 2  ->  R2                (e_L e_R)
            pol = 3  ->  R1 + R2           (unpolarised)
    3. Compute every QI quantity ('pure', 'pureN', 'cN', 'c12', 'EN..',
       'EN1/2/3', 'GMN_HMG'), exactly the same formulas as analysis.py.
    4. Save the per-point results to scan_results_psinteg_<rs>.npz, in the
       same schema that load_scan.py / plot_qi.py already understand.  The
       per-row differential cross section d^2 sigma / (d m_tt d cos theta_Z)
       in pb/GeV is carried along as the `dsigma_pbperGeV` column so plot_qi.py
       renders it alongside the QI heatmaps.
    5. Subprocess plot_qi.py on that file to produce qi_from_psinteg_<rs>.pdf
       in the current directory (multi-page parent only -- plot_qi.py's
       --split flag is NOT passed, so no per-quantity standalones are emitted
       here).  Pipelines that want the individual plots (e.g. build_plot_all.sh)
       should bypass step 5 with --no-plot and drive plot_qi.py themselves
       with the desired flags.

Note on the kinematic axes: cth1 and ph have been integrated out, so they
are stored as NaN and won't appear in plot_qi.py's title strings.  The
two free outer variables (m12, cos theta_Z) become the plot axes.

Usage:
    python3 qi_from_psinteg.py                 # auto-discover: process every
                                               # data/psinteg_res_*.npz on disk
    python3 qi_from_psinteg.py 0.5             # explicit rs label -> processes
                                               # data/psinteg_res_0.5.npz
    python3 qi_from_psinteg.py 1               # processes data/psinteg_res_1.npz
    python3 qi_from_psinteg.py 0.5 --no-plot   # only save the QI npz, skip the PDF
    python3 qi_from_psinteg.py 0.5 --force     # regenerate the QI npz even if
                                               # scan_results_psinteg_0.5.npz
                                               # already exists (default: reuse)
"""

import glob
import os
import re
import sys
import subprocess
import numpy as np
from math import sqrt, acos

sys.path.append('/Users/kazuki/Projects/pyHELAS')
from QI_functions import *           # normalise, purity, concurrence,
                                     # log_neg_bip, log_negativity
from ee_ttz import *                 # mt, mZ, ...
from ee_ttz_func import *
from ppt_julia import gmn_hmg


# Same QI list as analysis.py, in the same order, so plot_qi.py treats the
# resulting file identically.
ENTRIES = [
    'pure', 'pure1', 'pure2', 'pure3',
    'c12', 'EN12', 'EN13', 'EN23',
    'c1', 'c2', 'c3', 'EN1', 'EN2', 'EN3',
    'GMN_HMG',
]
INPUT_KEYS = ['rs', 'm12', 'th3', 'cth1', 'ph', 'pol']


def compute_qi(rho):
    """All QI quantities for one 12x12 rho on subsystems (t, tbar, Z) with
    Hilbert-space dimensions (2, 2, 3).  Mirrors the analysis.py block."""
    q = {}
    q['pure'] = purity(rho)

    rho_t = rho.reshape(2, 2, 3, 2, 2, 3)
    rho1  = np.einsum('xijxab->ijab', rho_t).reshape(6, 6)   # rho_{tbar Z}
    rho2  = np.einsum('ixjaxb->ijab', rho_t).reshape(6, 6)   # rho_{t Z}
    rho3  = np.einsum('ijxabx->ijab', rho_t).reshape(4, 4)   # rho_{t tbar}

    q['pure1'] = purity(rho1)
    q['pure2'] = purity(rho2)
    q['pure3'] = purity(rho3)

    # Clamp to >= 0 against numerical noise (purity ~ 1 + epsilon for nearly
    # pure global states would otherwise crash math.sqrt).
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


def build_qi_file(rs_label):
    """Build the QI npz for one sqrt(s) label.

    `rs_label` is the string that appears in data/psinteg_res_<rs_label>.npz
    (e.g. '1' for sqrt(s) = 1 TeV, '0.5' for sqrt(s) = 500 GeV) -- the same
    string is used to name the output file scan_results_psinteg_<rs_label>.npz.
    """
    here   = os.path.dirname(os.path.abspath(__file__))
    src    = os.path.join(here, 'data', f'psinteg_res_{rs_label}.npz')
    d      = np.load(src)
    R1     = d['R1']
    R2     = d['R2']
    rs     = d['rs']
    m12    = d['m12']
    th3    = d['th3']
    n_pts  = R1.shape[0]
    print(f"Loaded {src}: {n_pts} (m12, cth3) points, R shape = {R1.shape}")

    # Polarisation -> per-point R-matrix stack.
    R_by_pol = {1: R1, 2: R2, 3: R1 + R2}

    # Per-polarisation differential cross sections (added by the up-to-date
    # psinteg.py).  Fall back to NaN-padded arrays when the source npz pre-
    # dates that addition, so the rest of this script still runs; downstream
    # plot_qi.py renders missing cells as blank.
    def _dsig(key):
        return d[key] if key in d.files else np.full(n_pts, np.nan)
    dsig_by_pol = {
        1: _dsig('dsig_eR_pbperGeV'),       # e_R e_L
        2: _dsig('dsig_eL_pbperGeV'),       # e_L e_R
        3: _dsig('dsig_unpol_pbperGeV'),    # unpolarised average
    }
    if 'dsig_unpol_pbperGeV' in d.files:
        finite = np.isfinite(dsig_by_pol[3])
        if np.any(finite):
            print(f"  differential cross sections present: "
                  f"dsig_unpol range = [{np.nanmin(dsig_by_pol[3][finite]):.3e}, "
                  f"{np.nanmax(dsig_by_pol[3][finite]):.3e}] pb/GeV")
    else:
        print("  d sigma / (d m_tt d cos theta_Z) NOT in source npz "
              "(NaN-padded in output);")
        print("  regenerate data/psinteg_res_<mode>.npz with the up-to-date "
              "psinteg.py to populate it.")

    records = {k: [] for k in INPUT_KEYS + ENTRIES + ['dsigma_pbperGeV']}
    rho_all = []

    n_skipped = 0
    for pol in (1, 2, 3):
        Rstack  = R_by_pol[pol]
        dsigmas = dsig_by_pol[pol]
        for i in range(n_pts):
            R_i  = Rstack[i]
            trR  = np.trace(R_i)

            # Skip degenerate boundary points where pyHELAS produced NaN/inf
            # entries (e.g. m_tt = m_ttmax with the Z at rest in the lab, so
            # the Z's longitudinal polarisation eps^0 = (|p|/m, (E/m) p_hat)
            # blows up).  Fill QI columns with NaN; plot_qi.py renders these
            # cells as blank.
            bad = (not np.all(np.isfinite(R_i))) \
                  or (not np.isfinite(trR)) \
                  or (abs(trR) < 1e-30)

            records['rs'].append(float(rs[i]))
            records['m12'].append(float(m12[i]))
            records['th3'].append(float(th3[i]))
            records['cth1'].append(np.nan)
            records['ph'].append(np.nan)
            records['pol'].append(pol)
            records['dsigma_pbperGeV'].append(float(dsigmas[i]))

            if bad:
                n_skipped += 1
                for e in ENTRIES:
                    records[e].append(np.nan)
                rho_all.append(
                    np.full((12, 12), np.nan, dtype=np.complex128))
                continue

            rho = R_i / trR
            q   = compute_qi(rho)
            for e in ENTRIES:
                records[e].append(q[e])
            rho_all.append(rho)

        print(f"  pol = {pol}:  done ({n_pts} points)")

    if n_skipped > 0:
        print(f"  (skipped {n_skipped} degenerate point(s); QI set to NaN)")

    out_npz = os.path.join(here, f'scan_results_psinteg_{rs_label}.npz')
    np.savez_compressed(
        out_npz,
        rho=np.array(rho_all, dtype=np.complex128),
        **{k: np.array(v) for k, v in records.items()},
    )
    print(f"Wrote {out_npz} with {len(rho_all)} (point x pol) rows")
    return out_npz


def run_plot_qi(npz_path, rs_label):
    """Drive plot_qi.py on the QI npz to produce qi_from_psinteg_<rs_label>.pdf."""
    here    = os.path.dirname(os.path.abspath(__file__))
    plot_qi = os.path.join(here, 'plot_qi.py')
    out_pdf = os.path.join(here, f'qi_from_psinteg_{rs_label}.pdf')
    subprocess.run(
        [sys.executable, plot_qi, npz_path, out_pdf],
        check=True,
    )


def _discover_rs_labels(here):
    """Return the list of rs labels for which data/psinteg_res_<rs>.npz exists,
    sorted by ascending numeric value of the label when parseable.
    """
    pattern = os.path.join(here, 'data', 'psinteg_res_*.npz')
    labels  = []
    for path in glob.glob(pattern):
        m = re.match(r'psinteg_res_(.+)\.npz$', os.path.basename(path))
        if m:
            labels.append(m.group(1))

    def _key(s):
        try:
            return (0, float(s))
        except ValueError:
            return (1, s)
    return sorted(set(labels), key=_key)


def main(argv):
    args  = [a for a in argv[1:] if not a.startswith('-')]
    flags = [a for a in argv[1:] if a.startswith('-')]
    do_plot = '--no-plot' not in flags
    force   = ('--force' in flags) or ('-f' in flags)

    here = os.path.dirname(os.path.abspath(__file__))

    if args:
        # Explicit sqrt(s) label, e.g. "0.5" or "1".  Kept as a string so
        # filenames round-trip exactly with psinteg.py's naming.
        rs_labels = [args[0]]
    else:
        rs_labels = _discover_rs_labels(here)
        if not rs_labels:
            raise SystemExit(
                f"no data/psinteg_res_*.npz files found next to "
                f"{os.path.basename(__file__)}; run psinteg.py first."
            )
        print(f"Auto-discovered {len(rs_labels)} sqrt(s) "
              f"label(s): {rs_labels}")

    for rs_label in rs_labels:
        print()
        print(f"==> processing rs label '{rs_label}'")
        out_npz = os.path.join(here, f'scan_results_psinteg_{rs_label}.npz')

        # Skip the (Julia/Mosek-heavy) QI build step when the npz is already
        # on disk; pass --force / -f to override and recompute from scratch.
        if os.path.exists(out_npz) and not force:
            print(f"Using existing {out_npz}  (pass --force to regenerate)")
        else:
            out_npz = build_qi_file(rs_label)

        if do_plot:
            run_plot_qi(out_npz, rs_label)


if __name__ == '__main__':
    main(sys.argv)
