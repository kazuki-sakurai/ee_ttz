"""
Compute every QI quantity from the R-matrix dumps written by analysis.py.

analysis.py (refactored) saves only the colour-stripped 12x12 R-matrices
per kinematic point, for the two SM-surviving initial helicities:
    scan_R_<mode>_<rs_str>.npz
        R1, R2 : (Npts, 12, 12) complex
        rs, m12, th3, cth1, ph, mode_idx : (Npts,)

This script consumes one or more of those files, builds the density
matrices for the three pol settings,
    pol = 1 : rho = normalise(R1)              -- e_R e_L pure helicity
    pol = 2 : rho = normalise(R2)              -- e_L e_R pure helicity
    pol = 3 : rho = normalise(R1 + R2)         -- unpolarised mixture
and computes the full QI suite per (point, pol):
    purity   gamma (full state + 3 one-subsystem reductions)
    c_ABC    sqrt(2 * (1 - gamma_X)), the linear-entropy concurrence
    c12      Wootters concurrence on rho_{t tbar}
    EN_x     log-negativities (one-to-other and one-to-one cuts; raw
             output of QI_functions.log_negativity / log_neg_bip)
    GMN_HMG  HMG genuine multipartite negativity via Julia/Mosek SDP

The output schema matches the previous monolithic analysis.py, so the
downstream pipeline (plot_qi.py, plot_psinteg_realpol.py, fiducial_*,
luminosity_*) continues to work unmodified:
    scan_results_<mode>_<rs_str>.npz
        rho      : (Npts*3, 12, 12) complex   one per (point, pol)
        rs, m12, th3, cth1, ph, pol            : (Npts*3,)
        pure, pure1, pure2, pure3
        c1, c2, c3, c12
        EN12, EN13, EN23, EN1, EN2, EN3
        GMN_HMG
            -- all (Npts*3,) float

Usage
-----
    python3 qi_from_R.py                    # process every scan_R_*.npz
                                              next to this script
    python3 qi_from_R.py 0_1                # one file (scan_R_0_1.npz)
    python3 qi_from_R.py 0_1 2_1            # several files
    python3 qi_from_R.py scan_R_0_1.npz     # full filename also works
    python3 qi_from_R.py 0_1 --no-sdp       # skip GMN_HMG (no Julia)
    python3 qi_from_R.py 0_1 --force        # regenerate the QI npz even
                                              # if it already exists
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


# Output ENTRY columns, in the order the previous analysis.py wrote them.
ENTRIES = ['pure', 'pure1', 'pure2', 'pure3',
           'c12', 'EN12', 'EN13', 'EN23',
           'c1', 'c2', 'c3',
           'EN1', 'EN2', 'EN3',
           'GMN_HMG']
INPUT_KEYS = ['rs', 'm12', 'th3', 'cth1', 'ph', 'pol']


def _fmt_dt(seconds):
    seconds = max(0.0, float(seconds))
    h, rem = divmod(int(seconds), 3600)
    m, s   = divmod(rem, 60)
    return f"{h:d}:{m:02d}:{s:02d}" if h else f"{m:02d}:{s:02d}"


def _qi_suite(rho, gmn_hmg=None):
    """Full QI suite for one 12x12 rho on subsystems (t, tbar, Z) with
    Hilbert-space dimensions (2, 2, 3).  Returns a dict keyed by ENTRIES.

    `gmn_hmg`, if not None, is a callable taking rho and returning the
    HMG GMN; when None, GMN_HMG is set to NaN (used by --no-sdp).
    """
    q = {}
    q['pure'] = purity(rho)

    rt   = rho.reshape(2, 2, 3, 2, 2, 3)
    rho1 = np.einsum('xijxab->ijab', rt).reshape(6, 6)    # rho_{tbar Z}
    rho2 = np.einsum('ixjaxb->ijab', rt).reshape(6, 6)    # rho_{t Z}
    rho3 = np.einsum('ijxabx->ijab', rt).reshape(4, 4)    # rho_{t tbar}

    q['pure1'] = purity(rho1)
    q['pure2'] = purity(rho2)
    q['pure3'] = purity(rho3)

    # Clamp to >= 0 against floating-point noise (purity ~ 1 + epsilon
    # for nearly pure global states would otherwise crash math.sqrt).
    q['c1'] = sqrt(max(0.0, 2.0 * (1.0 - q['pure1'])))
    q['c2'] = sqrt(max(0.0, 2.0 * (1.0 - q['pure2'])))
    q['c3'] = sqrt(max(0.0, 2.0 * (1.0 - q['pure3'])))

    q['c12']  = concurrence(rho3)
    q['EN12'] = log_neg_bip(rho3, [2, 2])
    q['EN13'] = log_neg_bip(rho2, [2, 3])
    q['EN23'] = log_neg_bip(rho1, [2, 3])
    q['EN1']  = log_negativity(rho, 'A')
    q['EN2']  = log_negativity(rho, 'B')
    q['EN3']  = log_negativity(rho, 'C')

    q['GMN_HMG'] = float('nan') if gmn_hmg is None else gmn_hmg(rho)
    return q


def _load_R(path):
    """Read a scan_R_<mode>_<rs>.npz produced by analysis.py and return
    the column dict.  Validates that the expected keys are present."""
    d = np.load(path)
    required = ('R1', 'R2', 'rs', 'm12', 'th3', 'cth1', 'ph')
    missing = [k for k in required if k not in d.files]
    if missing:
        raise SystemExit(
            f"{path} is missing required columns: {missing}.  "
            f"Was it produced by the refactored analysis.py?")
    Npts = d['R1'].shape[0]
    if (d['R1'].shape != (Npts, 12, 12)
            or d['R2'].shape != (Npts, 12, 12)):
        raise SystemExit(
            f"{path}: R1/R2 must have shape (Npts, 12, 12); got "
            f"{d['R1'].shape}, {d['R2'].shape}")
    return d


def process_one(in_path, out_path, do_sdp=True, force=False):
    """Convert a single scan_R_*.npz into the matching scan_results_*.npz.

    Returns the output path on success; raises SystemExit on any
    pre-condition failure.
    """
    if (not force) and os.path.exists(out_path):
        print(f"Using existing {out_path}  (pass --force to regenerate)")
        return out_path

    d = _load_R(in_path)
    R1   = d['R1']
    R2   = d['R2']
    Npts = R1.shape[0]
    rs_  = d['rs']; m12 = d['m12']; th3 = d['th3']
    cth1 = d['cth1']; ph = d['ph']

    # Lazy import of the Julia SDP so --no-sdp avoids the ~30 s warm-up.
    gmn_callable = None
    if do_sdp:

        from ppt_cvxpy import get_GMN as _get_GMN
        def gmn_hmg(rho):
            return _get_GMN(rho, dims=[2, 2, 3])        
        #from ppt_julia import gmn_hmg
        gmn_callable = gmn_hmg

    print(f"  loaded {Npts} kinematic points from {os.path.basename(in_path)}")
    print(f"  computing QI suite for 3 pols x {Npts} points = "
          f"{3*Npts} rows  (GMN SDP {'ON' if do_sdp else 'OFF'})",
          flush=True)

    records = {k: [] for k in INPUT_KEYS + ENTRIES}
    rho_all = []

    # Progress reporting.  ~20 lines total per pol per file.
    n_total       = 3 * Npts
    progress_step = max(1, Npts // 20)
    t_start       = time.perf_counter()
    n_done        = 0

    for pol_code, R_fn in (
            (1, lambda i: R1[i]),
            (2, lambda i: R2[i]),
            (3, lambda i: R1[i] + R2[i]),
    ):
        t_pol = time.perf_counter()
        print(f"  pol = {pol_code}: starting {Npts} points ...", flush=True)
        for i in range(Npts):
            rho = normalise(R_fn(i))
            q   = _qi_suite(rho, gmn_hmg=gmn_callable)

            records['rs'  ].append(float(rs_[i]))
            records['m12' ].append(float(m12[i]))
            records['th3' ].append(float(th3[i]))
            records['cth1'].append(float(cth1[i]))
            records['ph'  ].append(float(ph[i]))
            records['pol' ].append(pol_code)
            for ent in ENTRIES:
                records[ent].append(q[ent])
            rho_all.append(rho)

            n_done += 1
            if ((i + 1) % progress_step == 0) or ((i + 1) == Npts):
                elapsed = time.perf_counter() - t_start
                rate    = n_done / max(elapsed, 1e-9)
                eta     = (n_total - n_done) / max(rate, 1e-9)
                print(f"    pol={pol_code}  {i+1:4d}/{Npts}  "
                      f"(total {100*n_done/n_total:5.1f}%)   "
                      f"elapsed {_fmt_dt(elapsed)}   "
                      f"ETA {_fmt_dt(eta)}",
                      flush=True)
        t_pol_dt = time.perf_counter() - t_pol
        print(f"  pol = {pol_code} done in {_fmt_dt(t_pol_dt)} "
              f"({Npts/max(t_pol_dt,1e-9):.2f} pt/s)", flush=True)

    np.savez_compressed(
        out_path,
        rho=np.array(rho_all, dtype=np.complex128),
        **{k: np.array(v) for k, v in records.items()},
    )
    print(f"Saved {len(rho_all)} rows to {out_path}", flush=True)
    return out_path


# ----------------------------------------------------------------------
# CLI: pick which scan_R_<...>.npz file(s) to process.
# ----------------------------------------------------------------------

def _discover_R_files(here):
    """Every scan_R_<...>.npz next to this script, sorted by mode then rs."""
    pattern = os.path.join(here, 'scan_R_*.npz')
    rx = re.compile(r'^scan_R_(?P<tag>.+)\.npz$')

    entries = []
    for p in sorted(glob.glob(pattern)):
        m = rx.match(os.path.basename(p))
        if m:
            entries.append((m.group('tag'), p))
    return entries


def _resolve_one_arg(here, arg):
    """Accept three CLI forms:
        '0_1'                 -> scan_R_0_1.npz (next to this script)
        'scan_R_0_1.npz'      -> same
        '/abs/path/to/file.npz' -> taken verbatim
    Returns (tag, abs_path).
    """
    if os.path.isabs(arg) and os.path.exists(arg):
        base = os.path.basename(arg)
        m = re.match(r'^scan_R_(?P<tag>.+)\.npz$', base)
        if m:
            return m.group('tag'), arg
        raise SystemExit(
            f"{arg} doesn't match the scan_R_<tag>.npz pattern.")

    if arg.endswith('.npz'):
        tag = re.match(r'^scan_R_(?P<tag>.+)\.npz$',
                       os.path.basename(arg))
        if not tag:
            raise SystemExit(
                f"{arg} doesn't match the scan_R_<tag>.npz pattern.")
        return tag.group('tag'), os.path.join(here, arg)

    # Plain tag (e.g. '0_1', '2_1', '0').
    tag = arg
    return tag, os.path.join(here, f'scan_R_{tag}.npz')


def main(argv):
    args  = [a for a in argv[1:] if not a.startswith('-')]
    flags = [a for a in argv[1:] if a.startswith('-')]
    do_sdp = '--no-sdp' not in flags
    force  = ('--force' in flags) or ('-f' in flags)
    here   = os.path.dirname(os.path.abspath(__file__))

    if args:
        jobs = [_resolve_one_arg(here, a) for a in args]
    else:
        jobs = _discover_R_files(here)
        if not jobs:
            raise SystemExit(
                f"no scan_R_*.npz files next to "
                f"{os.path.basename(__file__)}; run analysis.py first.")
        print(f"Auto-discovered {len(jobs)} scan_R_*.npz file(s): "
              f"{[t for t,_ in jobs]}")

    for tag, in_path in jobs:
        out_path = os.path.join(here, f'scan_results_{tag}.npz')
        print()
        print(f"==> processing {tag}: {in_path}")
        if not os.path.isfile(in_path):
            print(f"  ! missing {in_path}; skipping", file=sys.stderr)
            continue
        process_one(in_path, out_path, do_sdp=do_sdp, force=force)


if __name__ == '__main__':
    main(sys.argv)
