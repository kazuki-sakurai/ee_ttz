"""
Compute integrated-luminosity targets L(s) for the N1, N2, N3 linear
negativities of a saved fiducial region (e.g. fiducial_1_N3_fiducial.npz
produced by fiducial_from_psinteg.py), at significance levels s = 1, 2, 5.

The formulas (note section 6, eqs 6.3-6.4):
    L(v, s)   = s^2 * vmax^2 / (v^2 * sigma_fid * BR)            # L target
    n_Sigma   = sigma_fid * BR * L                               # fiducial events
    delta_v   = c_sys * vmax / sqrt(n_Sigma)                     # heuristic err
    sigma_v   = v / delta_v = v * sqrt(n_Sigma) / (c_sys * vmax) # naive sig.
where
    v        = N1, N2 or N3 linear negativity from the npz
    vmax     = upper bound on v: 1 for N3, 0.5 for N1 and N2
               (note eq 4.9: N <= (d_min - 1) / 2; d_min = 3 for N3,
                d_min = 2 for the qubit cuts N1, N2)
    sigma_fid= fiducial cross section [fb] from the npz
    BR       = user-supplied branching ratio (dimensionless)
    L        = integrated luminosity [1/fb]; default 8000 (user-overridable)
    c_sys    = systematic factor c_sys >= 1; the "most optimistic" value
               c_sys = 1 (matching note section 6) is the default
gives L(v, s) in inverse fb (1/fb), n_Sigma as a count of events, and
sigma_v as a (dimensionless) significance in standard deviations.

Output
------
One table per beam setting (unpol, setI, setII) showing
    sigma_fid          [fb]
    sigma_fid * BR     [fb]
    n_Sigma            (= sigma_fid * BR * L) events
and a row per (N3, N1, N2) with the negativity value, L(1), L(2), L(5),
and the naive significance sigma_v at the configured L (default 8000/fb).

Usage
-----
    python3 luminosity_from_fiducial.py
    python3 luminosity_from_fiducial.py fiducial_1_N3_fiducial.npz
    python3 luminosity_from_fiducial.py --br 0.123
    python3 luminosity_from_fiducial.py path/to/some_other_fiducial.npz --br 0.05
"""

import argparse
import os
import sys
import numpy as np


# Default npz to load if the user doesn't pass a positional path.
DEFAULT_INPUT = 'fiducial_1_N3_fiducial.npz'

# Significance levels at which to report L.
SIGMA_LEVELS = (1, 2, 5)

# Default integrated luminosity for the naive-significance column,
# in 1/fb.  Override on the command line with --L <value>.
DEFAULT_L_FB_INV = 8000.0

# Default systematic factor c_sys (note section 6).  c_sys = 1 is the
# "most optimistic" value used in the draft.  Override with --csys.
DEFAULT_C_SYS = 1.0

# Upper bound on each negativity (note eq 4.9).  Qubit cuts (N1, N2)
# have d_min = 2 -> vmax = 0.5; the (qubit, qubit)-vs-qutrit cut N3 has
# d_min = 3 -> vmax = 1.
VMAX = {'N1': 0.5, 'N2': 0.5, 'N3': 1.0}

# Order in which the N rows appear in the printed table (matches the
# user's table example: N3 first, then N1, then N2).
N_ORDER = ('N3', 'N1', 'N2')


def L_value(s, v, vmax, sigma_fid, BR):
    """L(s, v) = s^2 * vmax^2 / (v^2 * sigma_fid * BR).  Returns +inf
    when the denominator vanishes (v = 0, BR = 0, sigma_fid = 0) so the
    caller can print a sentinel rather than crash on a NaN/Inf."""
    v_f, sig_f, br_f = float(v), float(sigma_fid), float(BR)
    denom = (v_f * v_f) * sig_f * br_f
    if denom <= 0.0 or not np.isfinite(denom):
        return float('inf')
    return (s * s) * (vmax * vmax) / denom


def sigma_v_value(v, vmax, sigma_fid, BR, L_fb_inv, c_sys=DEFAULT_C_SYS):
    """Naive significance at a fixed integrated luminosity L (note eq 6.4):
        sigma_v = v / delta_v = v * sqrt(n_Sigma) / (c_sys * vmax)
    with n_Sigma = sigma_fid * BR * L.  Returns 0 when n_Sigma <= 0
    (no events, no significance)."""
    n_sigma = float(sigma_fid) * float(BR) * float(L_fb_inv)
    if n_sigma <= 0.0 or not np.isfinite(n_sigma) or vmax <= 0.0:
        return 0.0
    return float(v) * np.sqrt(n_sigma) / (float(c_sys) * float(vmax))


def parse_args(argv):
    p = argparse.ArgumentParser(
        prog='luminosity_from_fiducial.py',
        description='Compute integrated-luminosity targets L(s) for N1, '
                    'N2 and N3 from a saved fiducial-region npz.',
    )
    p.add_argument(
        'input', nargs='?', default=DEFAULT_INPUT,
        help=f'fiducial-region .npz to read.  '
             f'Default: {DEFAULT_INPUT} (in current directory).')
    p.add_argument(
        '--br', type=float, default=None,
        help='Branching ratio (skips the interactive prompt).  Must be '
             'in the open interval (0, 1].')
    p.add_argument(
        '--no-prompt', action='store_true',
        help='Fail instead of prompting when --br is not supplied.  '
             'Useful in non-interactive (CI/cron) runs.')
    p.add_argument(
        '--L', dest='L_fb_inv', type=float, default=DEFAULT_L_FB_INV,
        metavar='L',
        help=f'Integrated luminosity [1/fb] for the naive-significance '
             f'sigma_v column.  Default: {DEFAULT_L_FB_INV:g}.')
    p.add_argument(
        '--csys', dest='c_sys', type=float, default=DEFAULT_C_SYS,
        metavar='C',
        help=f'Systematic factor c_sys (>= 1) used in delta_v = c_sys * '
             f'vmax / sqrt(n_Sigma).  Default: {DEFAULT_C_SYS:g} '
             f'(the "most optimistic" value in note section 6).')
    return p.parse_args(argv[1:])


def prompt_BR():
    """Interactive BR prompt with light validation.  Re-prompts on bad
    input; raises SystemExit on EOF (non-interactive terminal)."""
    while True:
        try:
            raw = input('Enter BR (branching ratio, 0 < BR <= 1, '
                        'e.g. 0.123): ').strip()
        except EOFError:
            print()
            sys.exit("BR input required.  Pass --br <value> for a "
                     "non-interactive run.")
        try:
            br = float(raw)
        except ValueError:
            print(f"  '{raw}' is not a number; try again.")
            continue
        if br <= 0.0 or br > 1.0:
            print(f"  BR must satisfy 0 < BR <= 1; got {br}.  Try again.")
            continue
        return br


def _load_fiducial(path):
    """Open and minimally validate a fiducial npz produced by
    fiducial_from_psinteg.py.  Returns a dict of the columns this script
    actually consumes, raising SystemExit with a useful message if
    something is missing.
    """
    if not os.path.isfile(path):
        sys.exit(f"missing input file: {path}")
    d = np.load(path, allow_pickle=True)
    required = ('settings', 'sigma_fid', 'N1', 'N2', 'N3', 'rs')
    missing = [k for k in required if k not in d.files]
    if missing:
        sys.exit(
            f"{path} is missing required columns: {missing}.  "
            f"Did you run fiducial_from_psinteg.py with --no-qi?  "
            f"N1..N3 are derived from EN1..EN3 inside that script, so "
            f"they will be present whenever QI was computed.")
    return dict(
        settings  = [str(s) for s in d['settings']],
        sigma_fid = np.asarray(d['sigma_fid'], dtype=float),
        N1        = np.asarray(d['N1'],  dtype=float),
        N2        = np.asarray(d['N2'],  dtype=float),
        N3        = np.asarray(d['N3'],  dtype=float),
        rs_GeV    = float(d['rs']),
        region    = (str(d['region_name']) if 'region_name' in d.files
                     else 'rect'),
        cuts      = (d['cuts'].item() if 'cuts' in d.files
                     and hasattr(d['cuts'], 'item') else None),
    )


def _fmt_L(L):
    """Compact, slide-ready luminosity formatter."""
    if not np.isfinite(L):
        return '   inf  '
    return f"{L:11.5g}"


def print_tables(loaded, BR, L_fb_inv=DEFAULT_L_FB_INV, c_sys=DEFAULT_C_SYS):
    print()
    cuts = loaded.get('cuts')
    rs_GeV = loaded['rs_GeV']
    region = loaded['region']
    print(f"sqrt(s) = {rs_GeV:g} GeV       region: {region}")
    if cuts:
        print(f"  cuts (rectangular): {cuts}")
    print(f"BR     = {BR:g}")
    print(f"L      = {L_fb_inv:g}  1/fb   (used for the sigma_v column)")
    print(f"c_sys  = {c_sys:g}")
    print()

    settings  = loaded['settings']
    sigma_fid = loaded['sigma_fid']         # fb
    N = {k: loaded[k] for k in ('N1', 'N2', 'N3')}

    sv_col_label = f'sigma_v(L={L_fb_inv:g}/fb)'
    for is_, name in enumerate(settings):
        sigma    = float(sigma_fid[is_])
        sigma_BR = sigma * BR
        n_Sigma  = sigma_BR * L_fb_inv
        print(f"=== beam setting: {name} ===")
        print(f"  sigma_fid             = {sigma:14.6g}  fb")
        print(f"  sigma_fid * BR        = {sigma_BR:14.6g}  fb")
        print(f"  n_Sigma (= * L)       = {n_Sigma:14.6g}  events  "
              f"(at L = {L_fb_inv:g}/fb)")
        print()
        # Column layout: N | value | L(1) | L(2) | L(5)  | sigma_v(L)
        hdr = (f"  {'N':<3s} {'value':>10s}"
               + ''.join(f" {f'L({s}) [1/fb]':>14s}" for s in SIGMA_LEVELS)
               + f" {sv_col_label:>18s}")
        print(hdr)
        print('  ' + '-' * (len(hdr) - 2))
        for n_key in N_ORDER:
            v = float(N[n_key][is_])
            vmax = VMAX[n_key]
            row = f"  {n_key:<3s} {v:10.4f}"
            for s in SIGMA_LEVELS:
                row += f" {_fmt_L(L_value(s, v, vmax, sigma, BR)):>14s}"
            sv = sigma_v_value(v, vmax, sigma, BR, L_fb_inv, c_sys=c_sys)
            row += f" {sv:18.5g}"
            print(row)
        print()


def main(argv):
    args = parse_args(argv)
    loaded = _load_fiducial(args.input)
    print(f"Input file: {args.input}")

    if args.br is not None:
        if args.br <= 0.0 or args.br > 1.0:
            sys.exit(f"--br must satisfy 0 < BR <= 1; got {args.br}")
        BR = args.br
    elif args.no_prompt:
        sys.exit("--br is required when --no-prompt is set.")
    else:
        BR = prompt_BR()

    if args.L_fb_inv <= 0.0:
        sys.exit(f"--L must be positive; got {args.L_fb_inv}")
    if args.c_sys <= 0.0:
        sys.exit(f"--csys must be positive (>= 1 in practice); "
                 f"got {args.c_sys}")

    print_tables(loaded, BR, L_fb_inv=args.L_fb_inv, c_sys=args.c_sys)


if __name__ == '__main__':
    main(sys.argv)
