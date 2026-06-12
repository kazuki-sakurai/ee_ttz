"""
Compute every QI quantity from the (cos theta_Z, phi)-integrated R-matrices
written by `psinteg_cth1.py`, and produce the PDF plots via plot_qi.py.

This is the (m_tt, cos theta_t)-plane companion of `qi_from_psinteg.py`.

File naming
-----------
`psinteg_cth1.py` writes its outputs to:
    data/psinteg_cth1_res_<rs>.npz                  # cth3_cut = 1 (full)
    data/psinteg_cth1_res_cut<cth3_cut>_<rs>.npz    # restricted variant

This script propagates the entire suffix (`<rs>` or `cut<cth3_cut>_<rs>`)
into the downstream output filenames so the two variants don't collide:
    scan_results_psinteg_cth1_<rs>.npz
    scan_results_psinteg_cth1_cut<cth3_cut>_<rs>.npz
    qi_from_psinteg_cth1_<rs>.pdf
    qi_from_psinteg_cth1_cut<cth3_cut>_<rs>.pdf

Pipeline (one full pass per source npz):
    1. Load `data/psinteg_cth1_res_<tag>.npz`  (R1, R2, the outer-scan
       inputs m12 and cth1, the scalar cth3_cut, and -- when present --
       dsig_eR_pbperGeV, dsig_eL_pbperGeV, dsig_unpol_pbperGeV).
    2. For every (outer point) x (polarisation) build the 12x12 spin
       density matrix rho = R / Tr(R), where
            pol = 1  ->  R1                (e_R e_L)
            pol = 2  ->  R2                (e_L e_R)
            pol = 3  ->  R1 + R2           (unpolarised)
    3. Compute the QI suite ('pure', 'pureN', 'cN', 'c12', 'ENxx',
       'EN1/2/3', 'GMN_HMG') -- same formulas as `qi_from_psinteg.py`
       and `analysis.py`.
    4. Save per-point results to `scan_results_psinteg_cth1_<tag>.npz`,
       in the same schema plot_qi.py / load_scan.py already understand.
       The differential cross section (now d^2 sigma / (d m_tt d cos
       theta_t), in pb/GeV) rides along as the `dsigma_pbperGeV` column.
    5. Subprocess plot_qi.py to produce
       `qi_from_psinteg_cth1_<tag>.pdf` in the current directory.

Note on kinematic-axis bookkeeping: cth3 and ph were integrated out, so
th3 and ph are stored as NaN.  The two free outer variables (m12, cos
theta_t) become the plot axes; plot_qi.detect_axes() picks them up by
"which two columns vary" and ignores the NaN ones.

Usage:
    python3 qi_from_psinteg_cth1.py                    # auto-discover all
    python3 qi_from_psinteg_cth1.py 1                  # explicit tag
    python3 qi_from_psinteg_cth1.py cut0.75_1          # restricted variant
    python3 qi_from_psinteg_cth1.py 1 --no-plot        # only the QI npz
    python3 qi_from_psinteg_cth1.py 1 --force          # regenerate npz even
                                                       # if it already exists
"""

import glob
import os
import re
import sys
import subprocess
import numpy as np
from math import sqrt

sys.path.append('/Users/kazuki/Projects/pyHELAS')
from QI_functions import *           # normalise, purity, concurrence,
                                     # log_neg_bip, log_negativity
from ee_ttz import *                 # mt, mZ, ...
from ee_ttz_func import *
from ppt_julia import gmn_hmg


# Same QI list as analysis.py / qi_from_psinteg.py, in the same order so
# plot_qi.py treats the resulting file identically.
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

    # Clamp to >= 0 against numerical noise (purity ~ 1 + epsilon for
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


def build_qi_file(tag):
    """Build the QI npz for one psinteg_cth1 output file.

    `tag` is the full suffix in `data/psinteg_cth1_res_<tag>.npz`:
        tag = '1'              -> data/psinteg_cth1_res_1.npz
        tag = 'cut0.75_1'      -> data/psinteg_cth1_res_cut0.75_1.npz
    The same tag is used to name `scan_results_psinteg_cth1_<tag>.npz`.
    """
    here   = os.path.dirname(os.path.abspath(__file__))
    src    = os.path.join(here, 'data', f'psinteg_cth1_res_{tag}.npz')
    if not os.path.isfile(src):
        raise SystemExit(
            f"missing {src}; run psinteg_cth1.py first "
            f"(or check the tag '{tag}' you passed).")
    d      = np.load(src)
    R1     = d['R1']
    R2     = d['R2']
    rs     = d['rs']
    m12    = d['m12']
    cth1   = d['cth1']
    cth3_cut = float(d['cth3_cut']) if 'cth3_cut' in d.files else 1.0
    n_pts  = R1.shape[0]
    print(f"Loaded {src}: {n_pts} (m12, cth1) points  "
          f"(cth3 integration: |cos theta_Z| < {cth3_cut:g})")

    # Polarisation -> per-point R-matrix stack.
    R_by_pol = {1: R1, 2: R2, 3: R1 + R2}

    # Per-polarisation differential cross sections.  These are the
    # (cos theta_t, phi)-plane d^2 sigma / (d m_tt d cos theta_t) integrated
    # over cos theta_Z in [-cth3_cut, +cth3_cut], in pb/GeV.
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
                  f"dsig_unpol range = "
                  f"[{np.nanmin(dsig_by_pol[3][finite]):.3e}, "
                  f"{np.nanmax(dsig_by_pol[3][finite]):.3e}] pb/GeV")
    else:
        print("  d sigma / (d m_tt d cos theta_t) NOT in source npz "
              "(NaN-padded in output); regenerate with psinteg_cth1.py.")

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
            # entries.  Fill QI columns with NaN so plot_qi.py renders them
            # as blank cells.
            bad = (not np.all(np.isfinite(R_i))) \
                  or (not np.isfinite(trR)) \
                  or (abs(trR) < 1e-30)

            records['rs'  ].append(float(rs[i]))
            records['m12' ].append(float(m12[i]))
            # cth1 IS the second free variable here; th3 and ph are
            # integrated out and stored as NaN so plot_qi.detect_axes
            # picks (m12, cth1) automatically.
            records['th3' ].append(np.nan)
            records['cth1'].append(float(cth1[i]))
            records['ph'  ].append(np.nan)
            records['pol' ].append(pol)
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

    out_npz = os.path.join(here, f'scan_results_psinteg_cth1_{tag}.npz')
    # cth3_cut is a per-run constant, but `load_scan.select(data, pol=p)`
    # blindly applies the row mask to every array in the dict -- a
    # 0-dimensional scalar would crash with "too many indices for array".
    # Broadcast it to length-N so it indexes cleanly.
    n_rows = len(rho_all)
    np.savez_compressed(
        out_npz,
        rho=np.array(rho_all, dtype=np.complex128),
        cth3_cut=np.full(n_rows, cth3_cut, dtype=np.float64),
        **{k: np.array(v) for k, v in records.items()},
    )
    print(f"Wrote {out_npz} with {len(rho_all)} (point x pol) rows")
    return out_npz


def run_plot_qi(npz_path, tag):
    """Drive plot_qi.py on the QI npz to produce qi_from_psinteg_cth1_<tag>.pdf."""
    here    = os.path.dirname(os.path.abspath(__file__))
    plot_qi = os.path.join(here, 'plot_qi.py')
    out_pdf = os.path.join(here, f'qi_from_psinteg_cth1_{tag}.pdf')
    subprocess.run(
        [sys.executable, plot_qi, npz_path, out_pdf],
        check=True,
    )


def _discover_tags(here):
    """Return the list of tags for which data/psinteg_cth1_res_<tag>.npz
    exists, sorted with the full-range variants first (numeric ascending),
    then the restricted variants.  Tags look like '1', '0.5',
    'cut0.75_1', etc.
    """
    pattern = os.path.join(here, 'data', 'psinteg_cth1_res_*.npz')
    tags    = []
    rx      = re.compile(r'psinteg_cth1_res_(.+)\.npz$')
    for path in glob.glob(pattern):
        m = rx.match(os.path.basename(path))
        if m:
            tags.append(m.group(1))

    def _key(s):
        # Full-range labels (pure floats) sort numerically; restricted
        # labels (prefixed 'cut') come after them, then alphabetically.
        try:
            return (0, float(s))
        except ValueError:
            return (1, s)
    return sorted(set(tags), key=_key)


def main(argv):
    args  = [a for a in argv[1:] if not a.startswith('-')]
    flags = [a for a in argv[1:] if a.startswith('-')]
    do_plot = '--no-plot' not in flags
    force   = ('--force' in flags) or ('-f' in flags)

    here = os.path.dirname(os.path.abspath(__file__))

    if args:
        # Explicit tag(s).  Pass the full suffix exactly as it appears in
        # `data/psinteg_cth1_res_<tag>.npz` -- e.g. "1", "0.5", or
        # "cut0.75_1".  Multiple tags are processed in sequence.
        tags = list(args)
    else:
        tags = _discover_tags(here)
        if not tags:
            raise SystemExit(
                f"no data/psinteg_cth1_res_*.npz files found next to "
                f"{os.path.basename(__file__)}; run psinteg_cth1.py first.")
        print(f"Auto-discovered {len(tags)} tag(s): {tags}")

    for tag in tags:
        print()
        print(f"==> processing tag '{tag}'")
        out_npz = os.path.join(here,
                               f'scan_results_psinteg_cth1_{tag}.npz')

        # Skip the (Julia/Mosek-heavy) QI build step when the npz is already
        # on disk; pass --force / -f to override and recompute from scratch.
        if os.path.exists(out_npz) and not force:
            print(f"Using existing {out_npz}  (pass --force to regenerate)")
        else:
            out_npz = build_qi_file(tag)

        if do_plot:
            run_plot_qi(out_npz, tag)


if __name__ == '__main__':
    main(sys.argv)
