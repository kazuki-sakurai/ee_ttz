"""
1-D line plots of QI quantities vs sqrt(s), produced from the integrated
R-matrices written by qi_from_psinteg_all.py.  One quantity per page; three
curves per panel (e_R e_L solid red, e_L e_R solid blue, unpolarised dashed
black) by default.

The same LABELS / RANGES / COL_TITLES / POL_FILE_LABELS dictionaries as
plot_qi.py are reused so axis labels, legend tags, and filename suffixes
stay consistent across the 2-D and 1-D pipelines.

Usage:
    python3 plot_qi_vs_rs.py
        Read qi_from_psinteg_all_res.npz next to this file, write
        qi_vs_rs.pdf with all 15 quantities.

    python3 plot_qi_vs_rs.py qi_from_psinteg_all_res.npz qi_vs_rs.pdf
        Explicit input and output paths.

    python3 plot_qi_vs_rs.py -q GMN_HMG
        Single quantity -> qi_vs_rs_GMN_HMG.pdf

    python3 plot_qi_vs_rs.py -p 1
        Single polarisation curve, no legend -> qi_vs_rs_eR.pdf

    python3 plot_qi_vs_rs.py --list
        Print available quantity names and exit.
"""

import argparse
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.ticker import NullLocator

# Reuse plot_qi.py's dictionaries + matplotlib font setup.  Importing has
# the (deliberate) side-effect of setting the same rcParams.
from plot_qi import (
    LABELS, RANGES, COL_TITLES, POL_FILE_LABELS,
    ENTANGLEMENT, ENT_LEVEL,
    _add_derived,
)

# ----------------------------------------------------------------------
# Polarisation legend labels for the qi_from_psinteg_all pipeline.
#
# Override plot_qi.py's COL_TITLES (which describe pure-helicity beams)
# with labels that match the realistic beam settings configured in
# qi_from_psinteg_all.py.  Keep the same `pol` integer codes so the
# downstream npz schema stays unchanged.
#
# If you change the (P_e-, P_e+) values in qi_from_psinteg_all.POL_SETTINGS,
# update these labels in lock-step so the plot legends stay accurate.
# ----------------------------------------------------------------------
COL_TITLES = {
    1: r'$(\mathcal{P}_{e^-},\,\mathcal{P}_{e^+}) = (+0.8,\,-0.3)$',
    2: r'$(\mathcal{P}_{e^-},\,\mathcal{P}_{e^+}) = (-0.8,\,+0.3)$',
    3: r'Unpolarised',
}


# ----------------------------------------------------------------------
# Font sizes.  Edit these to retune every plot in this script in one
# place.  They are applied as matplotlib rcParams below, so they take
# effect for every figure built afterwards without per-call overrides.
#
# AXIS_LABEL_FONTSIZE : "x [TeV]" / "$N, N_G$" / colour-bar labels.
# TICK_LABEL_FONTSIZE : numbers on x and y axes (and on log colour bars).
# LEGEND_FONTSIZE     : entries inside ax.legend(); also set explicitly on
#                       each ax.legend() call below as a belt-and-braces
#                       guard against future rcParam regressions.
# TITLE_FONTSIZE      : subplot title (set via plt.rcParams['axes.titlesize']
#                       — note this is currently NOT used because we
#                       comment out the per-page title; left here for
#                       symmetry).
# ----------------------------------------------------------------------
AXIS_LABEL_FONTSIZE = 20.0
TICK_LABEL_FONTSIZE = 15.0
LEGEND_FONTSIZE     = 17.0
TITLE_FONTSIZE      = 14.0

plt.rcParams['axes.labelsize']  = AXIS_LABEL_FONTSIZE
plt.rcParams['xtick.labelsize'] = TICK_LABEL_FONTSIZE
plt.rcParams['ytick.labelsize'] = TICK_LABEL_FONTSIZE
plt.rcParams['legend.fontsize'] = LEGEND_FONTSIZE
plt.rcParams['axes.titlesize']  = TITLE_FONTSIZE


# Curve styling per polarisation.  Keys are the integer pol values stored
# in the data file.
POL_STYLE = {
    1: dict(color='C3', linestyle='-',  linewidth=2.0),   # red, solid
    2: dict(color='C0', linestyle='-',  linewidth=2.0),   # blue, solid
    3: dict(color='k',  linestyle='--', linewidth=2.0),   # black, dashed
}


# ----------------------------------------------------------------------
# Quantity groups used by the unpolarised "superimposed" extra pages
# (see process_file -- one extra page per group, only for pol = 3).
#
# To customise the appearance of any curve, edit its entry in `styles`
# below.  Each style dict is forwarded directly to ax.plot(**style), so
# any matplotlib Line2D kwarg is fair game: color, linestyle, linewidth,
# marker, markersize, alpha, dashes, zorder, ...
#
# The `keys` list also fixes the legend ORDER.  Re-order keys to re-order
# the legend.  A key whose style dict is missing simply falls back to
# DEFAULT_STYLE (and gets the next colour from matplotlib's cycle if no
# colour is set).
#
# `standalone_pdf`: if set to a filename (str), this group's page is
# written as its OWN single-page PDF next to the main output PDF instead
# of being appended to it.  Set to None (or omit) to keep it inside the
# main multi-page qi_vs_rs.pdf.  Relative paths are resolved against the
# main output PDF's directory; absolute paths are used as-is.
# ----------------------------------------------------------------------

DEFAULT_STYLE = dict(linewidth=2.0)

# `ylim`: optional (ymin, ymax) tuple to FIX the y-axis range of the group's
# plot.  Set it to override the auto-derived envelope of per-quantity
# RANGES.  Leave it as None (or omit the key) to keep the auto-derived
# bounds (envelope of RANGES[k] for k in keys, padded by 5% on each side).
SUPERIMPOSED_GROUPS = [
    # ---- (a) four purities: full state + the three bipartite reductions
    dict(
        keys=['pure', 'pure1', 'pure2', 'pure3'],
        ylabel=r'$\gamma$',
        ylim=None,           # e.g. (0.0, 1.0) to fix the y-range manually
        styles={
            'pure':  dict(color='r',  linestyle='-',  linewidth=2.2),
            'pure1': dict(color='orange', linestyle='-',  linewidth=2.0),
            'pure2': dict(color='g', linestyle='--',  linewidth=2.0),
            'pure3': dict(color='b', linestyle='-',  linewidth=2.0),
        },
        standalone_pdf='qi_vs_rs_gammas.pdf',
    ),
    # ---- (b) three one-to-other (linear) negativities + the HMG GMN N_G
    dict(
        keys=['N1', 'N2', 'N3', 'GMN_HMG'],
        ylabel=r'$N,\;\mathcal{N}_G$',
        ylim=[-0.005, 0.1],           # e.g. (0.0, 0.3) to fix the y-range manually
        styles={
            'N1':      dict(color='orange', linestyle='-',  linewidth=2.0),
            'N2':      dict(color='g', linestyle='--',  linewidth=2.0),
            'N3':      dict(color='b', linestyle='-',  linewidth=2.0),
            'GMN_HMG': dict(color='r',  linestyle='-', linewidth=2.2),
        },
        standalone_pdf='qi_vs_rs_N_NG.pdf',
    ),
]


def _select(data, **filters):
    """Lightweight load_scan.select() reimplementation (so this script
    doesn't depend on load_scan.py being on the path)."""
    n = len(data['pol'])
    mask = np.ones(n, dtype=bool)
    for k, v in filters.items():
        col = data[k]
        if callable(v):
            mask &= v(col)
        else:
            mask &= (col == v)
    return {k: vv[mask] for k, vv in data.items()}


def plot_quantity(data, key, pdf, only_pol=None):
    """Render one page: <quantity> vs sqrt(s), with one curve per kept pol."""
    if key not in data:
        return

    pols = sorted(set(int(p) for p in data['pol']))
    if only_pol is not None:
        if only_pol not in pols:
            raise SystemExit(
                f"polarisation {only_pol} not in data (available: {pols})"
            )
        pols = [only_pol]

    fig, ax = plt.subplots(figsize=(6.5, 4.6), constrained_layout=True)

    for p in pols:
        sub   = _select(data, pol=p)
        order = np.argsort(sub['rs'])
        ax.plot(
            sub['rs'][order] / 1000.0,        # plot in TeV
            sub[key][order],
            label=COL_TITLES.get(p, f'pol {p}'),
            **POL_STYLE.get(p, {}),
        )

    ax.set_xlabel(r'$\sqrt{s}\;[\mathrm{TeV}]$')
    ax.set_xscale('log')
    ax.set_xlim([0.45, 5])    
    # Explicit major x-ticks; suppress the default log minor ticks so the
    # axis shows only the labelled values.
    ax.set_xticks([0.45, 0.6, 0.8, 1.0, 1.5, 2.0, 3.0, 5.0])
    ax.set_xticklabels(['0.45', '0.6', '0.8', '1', '1.5', '2', '3', '5'])
    ax.xaxis.set_minor_locator(NullLocator())
    ax.set_ylabel(LABELS.get(key, key))
    #ax.set_title(LABELS.get(key, key))

    # Use the dimension-aware physical bounds as the y-limits, with a small
    # margin, so the natural ceiling of each measure is visible.  Skip when
    # the data range alone is more informative (purities of the integrated
    # state are usually well below 1 across the sqrt(s) scan).
    if key in RANGES:
        vmin, vmax = RANGES[key]
        margin = 0.05 * (vmax - vmin)
        ax.set_ylim(vmin - margin, vmax + margin)

    # For entanglement measures, hint at the zero / non-zero boundary used in
    # the 2-D plots' contour line.
    if key in ENTANGLEMENT:
        ax.axhline(0.0, color='gray', linestyle=':', linewidth=0.8, alpha=0.7)

    ax.grid(True, linestyle=':', linewidth=0.5, alpha=0.5)

    if len(pols) > 1:
        ax.legend(loc='best', frameon=False, fontsize=LEGEND_FONTSIZE)

    pdf.savefig(fig, dpi=300)
    plt.close(fig)


def plot_superimposed(data, keys, pdf, pol=3, ylabel=None, styles=None,
                      ylim=None):
    """Render one page: several QI quantities vs sqrt(s) on the same axes,
    for a SINGLE polarisation (default pol = 3, unpolarised).

    Curve appearance is controlled by `styles`, a dict mapping each key in
    `keys` to a dict of matplotlib Line2D kwargs (color, linestyle,
    linewidth, marker, ...).  Per-key entries override DEFAULT_STYLE.  Keys
    with no entry fall back to DEFAULT_STYLE and matplotlib's auto colour
    cycle.  Legend labels come from LABELS (paper notation).  Keys absent
    from `data` are silently skipped.  X-axis styling matches
    plot_quantity().

    `ylim`, if given as a (ymin, ymax) tuple, fixes the y-axis range.  When
    `ylim` is None the y-bounds are auto-derived from the envelope of the
    per-quantity RANGES of the included keys (padded by 5% on each side).
    """
    present = [k for k in keys if k in data]
    if not present:
        return False

    sub = _select(data, pol=pol)
    if len(sub.get('rs', [])) == 0:
        return False
    order = np.argsort(sub['rs'])

    fig, ax = plt.subplots(figsize=(6.5, 4.6), constrained_layout=True)

    styles = styles or {}
    for k in present:
        style = {**DEFAULT_STYLE, **(styles.get(k) or {})}
        ax.plot(
            sub['rs'][order] / 1000.0,        # plot in TeV
            sub[k][order],
            label=LABELS.get(k, k),
            **style,
        )

    ax.set_xlabel(r'$\sqrt{s}\;[\mathrm{TeV}]$')
    ax.set_xscale('log')
    ax.set_xlim([0.45, 5])
    ax.set_xticks([0.45, 0.6, 0.8, 1.0, 1.5, 2.0, 3.0, 5.0])
    ax.set_xticklabels(['0.45', '0.6', '0.8', '1', '1.5', '2', '3', '5'])
    ax.xaxis.set_minor_locator(NullLocator())
    if ylabel is not None:
        ax.set_ylabel(ylabel)

    # Y-limits.  Manual override wins; otherwise envelope the per-quantity
    # RANGES of the included keys (with a 5% margin on each side).
    if ylim is not None:
        ax.set_ylim(*ylim)
    else:
        bounds = [RANGES[k] for k in present if k in RANGES]
        if bounds:
            vmin = min(r[0] for r in bounds)
            vmax = max(r[1] for r in bounds)
            margin = 0.05 * (vmax - vmin)
            ax.set_ylim(vmin - margin, vmax + margin)

    # When every curve is an entanglement measure, hint at the zero / non-zero
    # boundary used in the 2-D plots' contour line.
    if all(k in ENTANGLEMENT for k in present):
        ax.axhline(0.0, color='gray', linestyle=':', linewidth=0.8, alpha=0.7)

    ax.grid(True, linestyle=':', linewidth=0.5, alpha=0.5)
    ax.legend(loc='best', frameon=False, fontsize=LEGEND_FONTSIZE)

    pdf.savefig(fig, dpi=300)
    plt.close(fig)
    return True


# ----------------------------------------------------------------------
# Cross-section plot (sigma vs sqrt(s)).  Renders one curve per pol from
# data['sigma_pb'] if that column is present.  Always written as a
# standalone PDF next to the main multi-page output; the filename below
# can be edited freely.  Set SIGMA_STANDALONE_PDF = None to disable the
# standalone file and only append the page to the main PDF instead.
# ----------------------------------------------------------------------

SIGMA_LABEL          = r'$\sigma(e^+e^- \to t\bar t Z)\;[\mathrm{fb}]$'
SIGMA_STANDALONE_PDF = 'qi_vs_rs_xsec.pdf'

# Optional manual y-limits for the cross-section plot.  Set to a (ymin, ymax)
# tuple in fb (e.g. SIGMA_YLIM = (1e-1, 1e3)) to fix the range; leave None for
# matplotlib's auto-scaling.
SIGMA_YLIM = None


def plot_cross_section(data, pdf, only_pol=None):
    """Render one page: total cross section sigma(e+ e- -> t tbar Z) vs
    sqrt(s), with one curve per kept polarisation (log y-axis).

    Reads data['sigma_fb'] (a column of length Nrs * Npol in the same row
    order as the QI quantities; built by qi_from_psinteg_all.py from the
    sigma_*_pb arrays in data/psinteg_all_res.npz, converted to fb).
    Returns False (and writes nothing) when that column is absent or
    entirely NaN.
    """
    if 'sigma_fb' not in data:
        return False
    sig_all = data['sigma_fb']
    if not np.any(np.isfinite(sig_all)):
        return False

    pols = sorted(set(int(p) for p in data['pol']))
    if only_pol is not None:
        if only_pol not in pols:
            return False
        pols = [only_pol]

    fig, ax = plt.subplots(figsize=(6.5, 4.6), constrained_layout=True)

    plotted_any = False
    for p in pols:
        sub   = _select(data, pol=p)
        order = np.argsort(sub['rs'])
        rs_TeV = sub['rs'][order] / 1000.0
        sig    = sub['sigma_fb'][order]
        mask   = np.isfinite(sig) & (sig > 0.0)
        if not np.any(mask):
            continue
        ax.plot(rs_TeV[mask], sig[mask],
                label=COL_TITLES.get(p, f'pol {p}'),
                **POL_STYLE.get(p, {}))
        plotted_any = True

    if not plotted_any:
        plt.close(fig)
        return False

    ax.set_xlabel(r'$\sqrt{s}\;[\mathrm{TeV}]$')
    ax.set_ylabel(SIGMA_LABEL)
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlim([0.45, 5])
    ax.set_xticks([0.45, 0.6, 0.8, 1.0, 1.5, 2.0, 3.0, 5.0])
    ax.set_xticklabels(['0.45', '0.6', '0.8', '1', '1.5', '2', '3', '5'])
    ax.xaxis.set_minor_locator(NullLocator())
    if SIGMA_YLIM is not None:
        ax.set_ylim(*SIGMA_YLIM)
    ax.grid(True, which='both', linestyle=':', linewidth=0.5, alpha=0.5)
    if len(pols) > 1:
        ax.legend(loc='best', frameon=False, fontsize=LEGEND_FONTSIZE)

    pdf.savefig(fig, dpi=300)
    plt.close(fig)
    return True


def _default_outpath(only_quantity=None, only_pol=None):
    parts = ['qi_vs_rs']
    if only_quantity is not None:
        parts.append(only_quantity)
    if only_pol is not None:
        parts.append(POL_FILE_LABELS[only_pol])
    return '_'.join(parts) + '.pdf'


def parse_args(argv):
    parser = argparse.ArgumentParser(
        prog='plot_qi_vs_rs.py',
        description=(
            'Line plots of QI quantities vs sqrt(s) from the integrated '
            'R-matrices written by qi_from_psinteg_all.py.'
        ),
    )
    parser.add_argument(
        'input', nargs='?', default=None,
        help='input npz (default: qi_from_psinteg_all_res.npz next to '
             'this script).',
    )
    parser.add_argument(
        'output', nargs='?', default=None,
        help='output PDF path. Default is derived from --quantity/--pol.',
    )
    parser.add_argument(
        '-q', '--quantity', dest='quantity', default=None,
        choices=list(LABELS.keys()), metavar='NAME',
        help='Plot only this single QI quantity.',
    )
    parser.add_argument(
        '-p', '--pol', dest='pol', default=None, type=int, choices=[1, 2, 3],
        help='Plot only one polarisation curve: 1 = e_R e_L, 2 = e_L e_R, '
             '3 = unpolarised.',
    )
    parser.add_argument(
        '-l', '--list', action='store_true',
        help='Print supported quantity names and exit.',
    )
    return parser.parse_args(argv[1:])


def process_file(in_path, out_path, only_quantity=None, only_pol=None):
    npz  = np.load(in_path)
    data = {k: npz[k] for k in npz.files}
    _add_derived(data)   # adds DGMN = min(EN1, EN2, EN3) - GMN_HMG (numpy-only)

    if only_quantity is not None:
        if only_quantity not in LABELS:
            raise SystemExit(f"unknown quantity {only_quantity!r}")
        if only_quantity not in data:
            raise SystemExit(
                f"quantity {only_quantity!r} not present in {in_path}"
            )
        keys_to_plot = [only_quantity]
    else:
        keys_to_plot = [k for k in LABELS if k in data]

    written = 0
    with PdfPages(out_path) as pdf:
        for k in keys_to_plot:
            plot_quantity(data, k, pdf, only_pol=only_pol)
            written += 1

        # Cross-section page (sigma vs sqrt(s)).  Emitted whenever the QI
        # npz carries a `sigma_pb` column (qi_from_psinteg_all.py copies it
        # from data/psinteg_all_res.npz when present).  Skipped if the user
        # restricted to a single quantity.
        if only_quantity is None:
            xs_out_dir = os.path.dirname(os.path.abspath(out_path))
            if SIGMA_STANDALONE_PDF:
                sa_path = (SIGMA_STANDALONE_PDF
                           if os.path.isabs(SIGMA_STANDALONE_PDF)
                           else os.path.join(xs_out_dir,
                                             SIGMA_STANDALONE_PDF))
                with PdfPages(sa_path) as sa_pdf:
                    ok_xs = plot_cross_section(data, sa_pdf,
                                               only_pol=only_pol)
                if ok_xs:
                    print(f"  wrote standalone {sa_path}")
                else:
                    # Remove the (empty) PDF that PdfPages opened so we don't
                    # leave a 0-page file lying around when sigma_pb is absent.
                    try:
                        os.remove(sa_path)
                    except OSError:
                        pass
            else:
                if plot_cross_section(data, pdf, only_pol=only_pol):
                    written += 1

        # Extra unpolarised-only pages: one per SUPERIMPOSED_GROUPS entry.
        # Skipped when the user has restricted to a single quantity, or to a
        # non-unpolarised polarisation (in which case the unpolarised curves
        # would be confusing).
        if only_quantity is None and (only_pol is None or only_pol == 3):
            out_dir = os.path.dirname(os.path.abspath(out_path))
            for group in SUPERIMPOSED_GROUPS:
                sa = group.get('standalone_pdf')
                if sa:
                    # One-page standalone PDF, written next to the main
                    # multi-page PDF (or at an absolute path if given).
                    sa_path = (sa if os.path.isabs(sa)
                               else os.path.join(out_dir, sa))
                    with PdfPages(sa_path) as sa_pdf:
                        ok = plot_superimposed(
                            data, group['keys'], sa_pdf, pol=3,
                            ylabel=group.get('ylabel'),
                            styles=group.get('styles'),
                            ylim=group.get('ylim'),
                        )
                    if ok:
                        print(f"  wrote standalone {sa_path}")
                else:
                    # Append to the main multi-page PDF (legacy behaviour).
                    if plot_superimposed(
                            data, group['keys'], pdf, pol=3,
                            ylabel=group.get('ylabel'),
                            styles=group.get('styles'),
                            ylim=group.get('ylim')):
                        written += 1

        info = pdf.infodict()
        info['Title']   = (
            f'QI vs sqrt(s)'
            + (f' [{only_quantity}]' if only_quantity else '')
            + (f' [{POL_FILE_LABELS[only_pol]}]' if only_pol else '')
        )
        info['Subject'] = 'ttZ density-matrix QI quantities, full PS integral'

    print(f"Wrote {out_path} with {written} page{'s' if written != 1 else ''}"
          + (f", quantity {only_quantity}" if only_quantity else "")
          + (f", pol {only_pol} ({POL_FILE_LABELS[only_pol]})"
             if only_pol is not None else ""))


def main(argv):
    args = parse_args(argv)

    if args.list:
        for k, v in LABELS.items():
            print(f'  {k:<10}  {v}')
        return

    here = os.path.dirname(os.path.abspath(__file__))

    in_path  = args.input or os.path.join(here, 'qi_from_psinteg_all_res.npz')
    if not os.path.isabs(in_path):
        in_path = os.path.join(here, in_path)

    out_path = args.output or _default_outpath(args.quantity, args.pol)
    if not os.path.isabs(out_path):
        out_path = os.path.join(here, out_path)

    process_file(in_path, out_path,
                 only_quantity=args.quantity, only_pol=args.pol)


if __name__ == '__main__':
    main(sys.argv)
