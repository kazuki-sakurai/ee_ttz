"""
Plot every QI quantity stored in a scan_results file on whichever 2D parameter
plane was scanned (m_tt vs cos theta_3, cos theta_1 vs phi, or cos theta_3 vs
phi).  One quantity per page of a PDF, three panels per page (one per
polarisation).  For every entanglement measure a dashed contour line marks
the boundary between zero and non-zero entanglement.

The two varying input columns are detected automatically from the file, so
the same script handles every mode that analysis.py produces.

Usage:
    python3 plot_qi.py
        Process every scan_results_*.npz next to this file and write a
        matching qi_scan_*.pdf for each.

    python3 plot_qi.py scan_results_0.npz
        Process a single file. Output goes to qi_scan_0.pdf.

    python3 plot_qi.py scan_results_0.npz qi_scan.pdf
        Explicit output path.

    python3 plot_qi.py -q GMN_HMG
        Restrict every PDF to just one quantity (one page, three panels).
        With no input file, this still loops over every scan_results_*.npz
        and writes qi_scan_<n>_GMN_HMG.pdf for each.

    python3 plot_qi.py scan_results_0.npz -q EN3
        Single quantity, single input file -> qi_scan_0_EN3.pdf.

    python3 plot_qi.py -p 1
        Restrict every PDF to a single polarisation panel (1 = e_R, 2 = e_L,
        3 = unpolarised).  Combined with --quantity if you want a single
        page with a single panel (-> qi_scan_<n>_<quantity>_<polTag>.pdf).

    python3 plot_qi.py scan_results_3.npz -q GMN_HMG -p 2
        One quantity, one polarisation -> qi_scan_3_GMN_HMG_eL.pdf.

    python3 plot_qi.py --list
        Print the available quantity names (the keys of LABELS) and exit.

Standalone per-quantity PDFs (opt-in via --split):
    By DEFAULT, plot_qi.py writes only the multi-page parent PDF and nothing
    else, so a bare `python3 plot_qi.py` leaves no extra files in the
    current directory.

    Pass `--split` to ALSO emit per-quantity / per-polarisation standalones
    next to the multi-page output:

        <out_stem>_<key>.pdf            # 3-panel (all pols side by side)
        <out_stem>_<key>_eR.pdf         # 1 panel, e_R e_L only
        <out_stem>_<key>_eL.pdf         # 1 panel, e_L e_R only
        <out_stem>_<key>_unpol.pdf      # 1 panel, unpolarised only

    For 16 keys: 16 combined-pol + 16 * 3 per-pol = 64 standalones per
    parent.  Add `--no-multipage` together with `--split` to skip the
    multi-page parent itself and write only the standalones; the positional
    output path then serves as the standalone-stem-and-directory hint.

    `build_plot_all.sh` uses `--split --no-multipage` to drop every
    standalone straight into a plot_all/ folder; running plot_qi.py
    interactively without flags will not affect that folder.
"""

import argparse
import contextlib
import glob
import os
import re
import sys
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.colors import LogNorm
from matplotlib.backends.backend_pdf import PdfPages

from load_scan import load_scan, select


# ----------------------------------------------------------------------
# LaTeX-style fonts.  The default below uses matplotlib's Computer Modern
# mathtext font, which does NOT require a system LaTeX install.  If you do
# have a working LaTeX install and want fully LaTeX-rendered text, set
#   plt.rcParams['text.usetex'] = True
# ----------------------------------------------------------------------
plt.rcParams['mathtext.fontset'] = 'cm'
plt.rcParams['font.family']      = 'serif'
plt.rcParams['font.serif']       = ['cmr10', 'DejaVu Serif']
plt.rcParams['axes.formatter.use_mathtext'] = True
# plt.rcParams['text.usetex'] = True

# ----------------------------------------------------------------------
# Font sizes for titles, axis labels, and tick labels (covers both x/y axes
# AND the colour-bar label + tick labels).  Numbers below are matplotlib's
# defaults multiplied by FONT_SCALE.
# ----------------------------------------------------------------------
FONT_SCALE       = 1.3        # baseline scale for ticks
LABEL_EXTRA      = 1.3        # additional bump applied only to x/y/z labels
# Subplot title is now matched to the x/y label size, so the per-panel
# "<pol>: <fixed kinematics>" line reads at the same weight as the axis labels.
#TITLE_FONTSIZE   = 10.0 * FONT_SCALE * LABEL_EXTRA
TITLE_FONTSIZE   = 13
plt.rcParams['axes.titlesize']   = TITLE_FONTSIZE                    # subplot titles
plt.rcParams['axes.labelsize']   = 10.0 * FONT_SCALE * LABEL_EXTRA   # x, y, cbar labels
plt.rcParams['xtick.labelsize']  = 10.0 * FONT_SCALE                 # x tick labels
plt.rcParams['ytick.labelsize']  = 10.0 * FONT_SCALE                 # y / cbar tick labels
plt.rcParams['figure.titlesize'] = 12.0 * FONT_SCALE                 # figure suptitle


# ----------------------------------------------------------------------
# Quantities to plot and their LaTeX-style labels.
# ----------------------------------------------------------------------
# Labels follow the notation of Sec. 4 of note.pdf (Goncalves, Navarro,
# Sakurai, "Tripartite Entanglement in e+ e- -> ttbar Z").  The N_*
# columns are the linear negativities, computed on the fly from the log
# negativities EN_* stored in the npz files via
#       N(rho) = (2^{E^N(rho)} - 1) / 2    (note eq 4.9)
# in _add_derived() below; that keeps existing scan_results_*.npz files
# unchanged.
#   purities          gamma(.)             Sec. 4 (purities of full + reductions)
#   bipartite C       calligraphic C       (concurrence on rho_{t tbar})
#   one-to-other C    calligraphic C(.|.)  (only meaningful for pure parents)
#   one-to-one neg.   N_{..}               eq (4.14), bound (4.15)
#   one-to-other neg. N(.|.)               eq (4.16), bound (4.17)
#   GMN (HMG)         N_G                  eq (4.23) (already a linear
#                                                     negativity; no conv.)
#   gap                D_G                 D_G = min N(.|.) - N_G   (>= 0)
LABELS = {
    'pure':    r'$\gamma(\rho)$',
    'pure1':   r'$\gamma_{\bar t Z}$',
    'pure2':   r'$\gamma_{t Z}$',
    'pure3':   r'$\gamma_{t \bar t}$',
    'c1':      r'$\mathcal{C}(t|\bar t Z)$',
    'c2':      r'$\mathcal{C}(\bar t | t Z)$',
    'c3':      r'$\mathcal{C}(Z | t \bar t)$',
    'c12':     r'$\mathcal{C}_{t \bar t}$',
    'N12':     r'$N_{t \bar t}$',
    'N13':     r'$N_{t Z}$',
    'N23':     r'$N_{\bar t Z}$',
    'N1':      r'$N(t|\bar t Z)$',
    'N2':      r'$N(\bar t|t Z)$',
    'N3':      r'$N(Z|t \bar t)$',
    'GMN_HMG': r'${\cal N}_G$',
    # Derived: D_G = min{ N(t|tbarZ), N(tbar|tZ), N(Z|ttbar) } - N_G(rho).
    # Built on the fly from GMN_HMG and N1/N2/N3 in _add_derived(), so it
    # never re-invokes the HMG-GMN Julia/Mosek SDP.  All terms are now in
    # linear-negativity units, so the subtraction is unit-consistent.
    'DGMN':    r'$D_G$',
    # Differential cross section.  Stored in the npz as `dsigma_pbperGeV`
    # (pb/GeV) by qi_from_psinteg.py; _add_derived() multiplies by 1000 to
    # produce `dsigma_fbperGeV` (fb/GeV), which is what we actually plot
    # because fb is the natural scale for e+ e- -> t tbar Z at lepton-
    # collider energies.  Auto-scaled by default (no entry in RANGES below);
    # add an entry to RANGES (in fb/GeV) to fix the colour-bar limits.
    'dsigma_fbperGeV': r'$\frac{\partial^{2}\sigma}{ \partial m_{t\bar t}\,\partial\cos\theta_{Z} }\;[\mathrm{fb/GeV}]$',
}

# Per-quantity colour-scale bounds (vmin, vmax), set to the physical range of
# each variable for our (qubit, qubit, qutrit) ttZ system: A=t (dim 2), B=tbar
# (dim 2), C=Z (dim 3).  Used as the pcolormesh / colour-bar limits, NOT to
# clip the data.
#
#   - Tr ρ²    : in [1/d, 1] for a state on a d-dimensional Hilbert space.
#                pure  : full state, d = 2*2*3 = 12         -> [1/12, 1]
#                pure1 : ρ_BC (A traced out), d = 2*3 = 6   -> [1/6,  1]
#                pure2 : ρ_AC (B traced out), d = 2*3 = 6   -> [1/6,  1]
#                pure3 : ρ_AB (C traced out), d = 2*2 = 4   -> [1/4,  1]
#
#   - c_X = sqrt(2(1 - Tr ρ_X²)) (linear-entropy / I-concurrence) :
#                in [0, sqrt(2*(1 - 1/d_X))].
#                c1 (X=BC, d=6) -> [0, sqrt(5/3)]
#                c2 (X=AC, d=6) -> [0, sqrt(5/3)]
#                c3 (X=AB, d=4) -> [0, sqrt(3/2)]
#
#   - C(ρ_AB) (Wootters concurrence on a 2x2) : [0, 1].
#
#   - N_M (linear negativity, note eq 4.9):  bounded by (d_min - 1)/2 for
#     the smaller-dimension subsystem of the cut.
#                N12, N13, N23, N1, N2 : d_min = 2 -> [0, 1/2]
#                N3                    : d_min = 3 -> [0, 1]
#
#   - HMG GMN (N_G, note eq 4.23):  convex roof of the bipartite negativity,
#     bounded above by the smallest one-to-other bipartite negativity, hence
#     also (d_min - 1)/2 = 1/2 for the qubit cuts.
RANGES = {
    'pure':    (1.0/12.0, 1.0),
    'pure1':   (1.0/6.0,  1.0),
    'pure2':   (1.0/6.0,  1.0),
    'pure3':   (1.0/4.0,  1.0),
    'c1':      (0.0, np.sqrt(2.0 * (1.0 - 1.0/6.0))),
    'c2':      (0.0, np.sqrt(2.0 * (1.0 - 1.0/6.0))),
    'c3':      (0.0, np.sqrt(2.0 * (1.0 - 1.0/4.0))),
    'c12':     (0.0, 1.0),
    'N12':     (0.0, 0.5),
    'N13':     (0.0, 0.5),
    'N23':     (0.0, 0.5),
    'N1':      (0.0, 0.5),
    'N2':      (0.0, 0.5),
    'N3':      (0.0, 1.0),
    'GMN_HMG': (0.0, 0.5),
    # D_G = min(N1, N2, N3) - N_G.  N_G <= min N(.|.) in theory, so D_G >= 0.
    # Bounded above by the smallest one-to-other negativity (0.5 for the
    # qubit cuts).
    'DGMN':    (0.0, 0.5),
    'dsigma_fbperGeV':    (10**-4, 10**-1),
    # --- Differential cross section colour-bar limits (in fb/GeV) ---
    # Uncomment the line below (and tweak the numbers) to fix the colour-bar
    # range of the dsigma_fbperGeV log-scale plot.  Units are fb/GeV (1 pb =
    # 1000 fb), to match the colour-bar label.  Both values MUST be strictly
    # positive (LogNorm rejects <= 0); otherwise the entry is ignored and
    # the script falls back to auto-fitting from positive data.  Leave
    # commented out for auto-fitting (current default).
    # 'dsigma_fbperGeV': (1.0, 1e5),
}

# Quantities for which a zero / non-zero contour line is meaningful
# (i.e. proper entanglement measures).
ENTANGLEMENT = {
    'c12', 'N12', 'N13', 'N23', 'N1', 'N2', 'N3', 'GMN_HMG',
}

# Quantities that should be drawn with a LOGARITHMIC colour scale.
# The differential cross section spans many orders of magnitude across the
# (m_tt, cos theta_Z) plane, so a linear scale washes out everything except
# the peak.  build_quantity_figure() switches to LogNorm and masks
# non-positive / NaN cells for any key in this set.
LOG_SCALE = {
    'dsigma_fbperGeV',
}

# A small positive threshold above SDP/numerical noise; values below this are
# treated as "zero entanglement", values above as "non-zero".
ENT_LEVEL = 1.*1e-3

# Subplot column titles (no "polarisation" wording, no "(pol N)").
COL_TITLES = {
    1: r'$e^{-}_{R} e^{+}_{L}$',
    2: r'$e^{-}_{L} e^{+}_{R}$',
    3: r'Unpol',
}

# Short, filename-safe tags for each polarisation, used by --pol to build
# the default output filename (e.g. qi_scan_3_GMN_HMG_eR.pdf).
POL_FILE_LABELS = {
    1: 'eR',
    2: 'eL',
    3: 'unpol',
}


# ----------------------------------------------------------------------
# Per-input-column metadata: how to derive the plotting variable, an axis
# label, and (optionally) custom tick marks.  The transform converts the saved
# column into the value we actually plot.
# ----------------------------------------------------------------------
COL_INFO = {
    # 'rs' is always present in the saved data, but is currently never the
    # scan axis (it is omitted from AXIS_PRIORITY).  We still need an entry
    # here so _fixed_input_summary can format its constant value.
    'rs':   dict(
        label=r'$\sqrt{s}\;[\mathrm{GeV}]$',
        transform=lambda c: c,
        ticks=None, ticklabels=None,
    ),
    'm12':  dict(
        label=r'$m_{t\bar{t}}\;[\mathrm{GeV}]$',
        transform=lambda c: c,
        ticks=None, ticklabels=None,
    ),
    'th3':  dict(
        # th3 is stored as the angle; we plot cos(theta_3).
        label=r'$\cos\theta_{Z}$',
        transform=lambda c: np.cos(c),
        ticks=None, ticklabels=None,
    ),
    'cth1': dict(
        label=r'$\cos\theta_{t}$',
        transform=lambda c: c,
        ticks=None, ticklabels=None,
    ),
    'ph':   dict(
        label=r'$\phi$',
        transform=lambda c: c,
        ticks=[-np.pi, -np.pi/2, 0.0, np.pi/2, np.pi],
        ticklabels=[r'$-\pi$', r'$-\pi/2$', r'$0$', r'$\pi/2$', r'$\pi$'],
    ),
}

# When multiple input columns vary, this priority decides which one is the
# x-axis (first match in the list) and which is the y-axis (second).
AXIS_PRIORITY = ['m12', 'cth1', 'th3', 'ph']


# ----------------------------------------------------------------------
def detect_axes(data):
    """
    Find which two input columns vary in this scan and return (x_key, y_key).
    """
    varying = []
    for k in AXIS_PRIORITY:
        if k not in data:
            continue
        col = COL_INFO[k]['transform'](data[k])
        if np.unique(col).size > 1:
            varying.append(k)

    if len(varying) < 2:
        raise RuntimeError(
            f"expected at least 2 varying input columns, found {varying}"
        )
    if len(varying) > 2:
        # Keep the two highest-priority ones.
        varying = varying[:2]

    return varying[0], varying[1]


def _grid(sub, key, x_key, y_key):
    """Reshape a 1-D scan on a regular (x_key, y_key) grid into 2-D."""
    xv = COL_INFO[x_key]['transform'](sub[x_key])
    yv = COL_INFO[y_key]['transform'](sub[y_key])

    x_axis = np.unique(xv)
    y_axis = np.unique(yv)

    nx, ny = x_axis.size, y_axis.size
    if nx * ny != xv.size:
        raise RuntimeError(
            f"non-regular grid: {xv.size} points but axes have sizes "
            f"{nx} ({x_key}) x {ny} ({y_key})"
        )

    ix = np.searchsorted(x_axis, xv)
    iy = np.searchsorted(y_axis, yv)
    Z  = np.full((ny, nx), np.nan)
    Z[iy, ix] = sub[key]
    return x_axis, y_axis, Z


def _apply_ticks(ax, axis, key):
    """Apply optional custom ticks/labels for this column."""
    info = COL_INFO[key]
    ticks      = info.get('ticks')
    ticklabels = info.get('ticklabels')
    if ticks is None:
        return
    if axis == 'x':
        ax.set_xticks(ticks)
        if ticklabels is not None:
            ax.set_xticklabels(ticklabels)
    else:
        ax.set_yticks(ticks)
        if ticklabels is not None:
            ax.set_yticklabels(ticklabels)


def build_quantity_figure(data, key, x_key, y_key,
                          only_pol=None, fixed_str=''):
    """Build (but do NOT save / close) the figure for one QI quantity.

    Returns the matplotlib Figure, or None if `key` is absent from `data`.
    The caller is responsible for ``pdf.savefig(fig)`` and ``plt.close(fig)``.

    If `only_pol` is given (1, 2, or 3), only that polarisation panel is drawn
    and the figure is the corresponding fraction of the usual width.

    `fixed_str`, if non-empty, is appended to each subplot title after the
    polarisation label, e.g. "unpolarised: $\\sqrt{s}=1$ TeV, ...".
    """
    if key not in data:
        return None

    pols = sorted(set(int(p) for p in data['pol']))
    if only_pol is not None:
        if only_pol not in pols:
            raise SystemExit(
                f"polarisation {only_pol} not present in data "
                f"(available: {pols})"
            )
        pols = [only_pol]

    # Colour-scale limits.
    #
    # Resolution order, identical to the linear case:
    #     1. RANGES[key]                   -> manual override (winning hand)
    #     2. data-derived min/max          -> auto-fit (no entry in RANGES)
    #
    # For keys in LOG_SCALE the data-derived branch is additionally restricted
    # to strictly POSITIVE values (LogNorm rejects <= 0).  RANGES entries are
    # used verbatim (so the user can pick any positive (vmin, vmax) without
    # caring about the actual data), and a non-positive entry triggers an
    # immediate fallback to the auto branch.
    #
    # TO FIX THE COLOUR-BAR RANGE FOR THE DIFFERENTIAL CROSS SECTION,
    # add an entry to RANGES above, e.g.
    #     RANGES['dsigma_pbperGeV'] = (1e-3, 1e2)
    # The same trick works for any other LOG_SCALE key you add later.
    log_scale = (key in LOG_SCALE)
    if key in RANGES:
        vmin, vmax = RANGES[key]
        if log_scale and not (vmin > 0.0 and vmax > 0.0):
            # Manual override is incompatible with a log scale -- ignore it
            # and fall back to auto-fitting from strictly positive cells.
            print(f"  ! RANGES[{key!r}] = ({vmin}, {vmax}) is not strictly "
                  f"positive; ignoring and auto-fitting from data.",
                  file=sys.stderr)
            col = np.asarray(data[key], dtype=float)
            pos = col[np.isfinite(col) & (col > 0.0)]
            if pos.size == 0:
                log_scale = False
                vmin = float(np.nanmin(col))
                vmax = float(np.nanmax(col))
            else:
                vmin = float(pos.min())
                vmax = float(pos.max())
    else:
        if log_scale:
            col = np.asarray(data[key], dtype=float)
            pos = col[np.isfinite(col) & (col > 0.0)]
            if pos.size == 0:
                # No positive values to log-scale -- fall back to linear so
                # LogNorm doesn't raise.
                log_scale = False
                vmin = float(np.nanmin(col))
                vmax = float(np.nanmax(col))
            else:
                vmin = float(pos.min())
                vmax = float(pos.max())
        else:
            vmin = float(np.nanmin(data[key]))
            vmax = float(np.nanmax(data[key]))
    if vmin == vmax:                       # avoid degenerate color range
        if log_scale:
            vmax *= 1.0 + 1e-6
        else:
            vmin -= 1e-12
            vmax += 1e-12

    # `norm` and `vmin/vmax` are mutually exclusive in pcolormesh, so build
    # the colour-scale kwargs once and switch on the key.
    if log_scale:
        norm_kw = dict(norm=LogNorm(vmin=vmin, vmax=vmax))
    else:
        norm_kw = dict(vmin=vmin, vmax=vmax)

    fig, axes = plt.subplots(
        1, len(pols),
        figsize=(4.6 * len(pols), 4.2),
        sharey=True, constrained_layout=True,
    )
    if len(pols) == 1:
        axes = [axes]

    mesh = None
    for ax, p in zip(axes, pols):
        sub = select(data, pol=p)
        x_axis, y_axis, Z = _grid(sub, key, x_key, y_key)

        if log_scale:
            # Mask non-positive / NaN cells so LogNorm doesn't see them;
            # they render with the colormap's `bad` colour (transparent).
            Z = np.ma.masked_where(~(np.isfinite(Z) & (Z > 0.0)), Z)

        mesh = ax.pcolormesh(
            x_axis, y_axis, Z,
            cmap=cm.rainbow, shading='auto',
            # Suppress the thin grid lines that vector-PDF viewers draw
            # between pcolormesh cells: rasterise the mesh itself (axes,
            # labels, contour lines stay vector) and disable cell edges /
            # antialiasing.
            edgecolors='none', linewidth=0,
            antialiased=False, rasterized=True,
            **norm_kw,
        )

        # Boundary contour: only meaningful for entanglement measures and
        # only if both regions actually appear in the panel.
        if (key in ENTANGLEMENT
                and np.any(np.isfinite(Z))
                and np.nanmin(Z) <  ENT_LEVEL
                and np.nanmax(Z) >= ENT_LEVEL):
            try:
                ax.contour(
                    x_axis, y_axis, Z,
                    levels=[ENT_LEVEL],
                    colors='black', linewidths=1.4, linestyles='--',
                )
            except Exception:
                pass

        ax.set_xlabel(COL_INFO[x_key]['label'])
        pol_label = COL_TITLES.get(p, '')
        # Single-line title: "<pol>: <fixed kinematics>".  TITLE_FONTSIZE is
        # tuned so this fits in one panel of the three-panel layout.
        title = f'{pol_label}: {fixed_str}' if fixed_str else pol_label
        ax.set_title(title)
        ax.grid(False)
        _apply_ticks(ax, 'x', x_key)

    axes[0].set_ylabel(COL_INFO[y_key]['label'])
    _apply_ticks(axes[0], 'y', y_key)

    cbar = fig.colorbar(mesh, ax=axes, shrink=0.95, pad=0.02)
    cbar.set_label(LABELS.get(key, key))

    return fig


def plot_quantity(data, key, pdf, x_key, y_key, only_pol=None, fixed_str=''):
    """Render one page of the PDF (3 panels, shared color scale) for `key`.

    Backwards-compatible wrapper around build_quantity_figure(): builds the
    figure, saves it to `pdf`, closes it.
    """
    fig = build_quantity_figure(
        data, key, x_key, y_key,
        only_pol=only_pol, fixed_str=fixed_str,
    )
    if fig is None:
        return
    pdf.savefig(fig, dpi=300)
    plt.close(fig)


# PDG Z mass; used only for the symbolic detection
#   m_tt == sqrt(s) - m_Z   (i.e. the kinematic upper edge of mode 1).
# Tolerance is ~1 GeV so we don't depend on the exact mZ value used inside
# pyHELAS; any sensible value (91.18..91.2) round-trips through this check.
_M_Z_REF = 91.1876
_M_Z_TOL = 1.0


def _fixed_input_summary(data, x_key, y_key):
    """Short string describing the inputs that are held fixed in this scan.

    Used both inside subplot titles (next to the polarisation label) and in
    the PDF's Subject metadata.  sqrt(s) is rendered in TeV, m_tt in GeV,
    angles via :g formatting (so e.g. 0.00 -> 0, 0.50 -> 0.5).  Values
    within 1e-10 of zero are snapped to exactly zero so that the cos(acos(0))
    round-trip in analysis.py doesn't print 6.12e-17 in the title.

    When sqrt(s) and m_tt are both fixed and m_tt is at the kinematic upper
    edge sqrt(s) - m_Z (within 1 GeV), m_tt is rendered symbolically as
    "sqrt(s) - m_Z" instead of as its numerical value.
    """
    def _snap(v, tol=1e-10):
        return 0.0 if abs(v) < tol else v

    # Pre-pass: determine which inputs are constant and their values.
    fixed_vals = {}
    for k in ('rs', 'm12', 'th3', 'cth1', 'ph'):
        if k in (x_key, y_key) or k not in data:
            continue
        col = COL_INFO[k]['transform'](data[k])
        u = np.unique(col[np.isfinite(col)])
        if u.size != 1:
            continue
        fixed_vals[k] = _snap(float(u[0]))

    # Symbolic detection: m_tt at the kinematic upper edge.
    m12_at_edge = (
        'rs' in fixed_vals and 'm12' in fixed_vals
        and abs((fixed_vals['rs'] - fixed_vals['m12']) - _M_Z_REF) < _M_Z_TOL
    )

    fixed = []
    for k in ('rs', 'm12', 'th3', 'cth1', 'ph'):
        if k not in fixed_vals:
            continue
        val = fixed_vals[k]
        if k == 'rs':
            fixed.append(rf'$\sqrt{{s}}={val/1000.0:g}$ TeV')
        elif k == 'm12':
            if m12_at_edge:
                fixed.append(r'$m_{t\bar t}=\sqrt{s}-m_{Z}$')
            else:
                fixed.append(rf'$m_{{t\bar t}}={val:g}$ GeV')
        elif k == 'th3':
            fixed.append(rf'$\cos\theta_{{Z}}={val:g}$')
        elif k == 'cth1':
            fixed.append(rf'$\cos\theta_{{t}}={val:g}$')
        elif k == 'ph':
            fixed.append(rf'$\phi={val:g}$')
    return ', '.join(fixed)


def _add_derived(data):
    """Augment `data` in-place with quantities derived from the saved columns.

    Adds, when the source columns are present:

        N12, N13, N23, N1, N2, N3  : linear negativities, converted from the
                                     log negativities EN_x stored in the npz
                                     via N = (2^{E^N} - 1) / 2  (note eq 4.9).
                                     Source EN_x are kept unchanged so the npz
                                     schema is not disturbed.

        DGMN = min(N1, N2, N3) - GMN_HMG (D_G in note notation; gap between
                                     the smallest one-to-other negativity and
                                     the genuine multipartite negativity N_G).

    NaN propagates through np.power / np.minimum, so rows that were marked bad
    upstream stay NaN here too.  This is purely a numpy operation on the
    per-row arrays, so it does NOT re-invoke the HMG GMN Julia/Mosek SDP.
    """
    # Log-negativity (E^N) -> linear negativity (N) for every cut present.
    for k_log, k_lin in [
        ('EN12', 'N12'), ('EN13', 'N13'), ('EN23', 'N23'),
        ('EN1',  'N1'),  ('EN2',  'N2'),  ('EN3',  'N3'),
    ]:
        if k_log in data:
            data[k_lin] = (np.power(2.0, data[k_log]) - 1.0) / 2.0

    # D_G uses the LINEAR one-to-other negativities (consistent units with
    # GMN_HMG, which is already in linear-negativity units per the HMG SDP
    # in quantum-correlations/src/entanglement/RobustnessToPPTMixture.jl).
    needed = ('GMN_HMG', 'N1', 'N2', 'N3')
    if all(k in data for k in needed):
        data['DGMN'] = (
            np.minimum(np.minimum(data['N1'], data['N2']), data['N3'])
            - data['GMN_HMG']
        )

    # Differential cross section in fb/GeV.  The npz stores it in pb/GeV
    # (built by qi_from_psinteg.py from psinteg.py, where GEV_INV2_TO_PB is
    # used); we expose an fb-unit copy for plotting since fb/GeV is the
    # natural scale for e+ e- -> t tbar Z at lepton-collider energies.
    # 1 pb = 1000 fb.  The pb column is left untouched in `data` so any
    # downstream consumer keeps working.
    if 'dsigma_pbperGeV' in data:
        data['dsigma_fbperGeV'] = data['dsigma_pbperGeV'] * 1000.0
    return data


def process_file(in_path, out_path, only_quantity=None, only_pol=None,
                 split=False, write_multipage=True):
    """Build a PDF for one scan_results file.

    If `only_quantity` is given (must be a key of LABELS), the PDF contains
    just that one page; otherwise every quantity in LABELS that is present in
    `data` is written, in LABELS order.

    If `only_pol` is given (1, 2, or 3), each page contains a single panel
    for that polarisation only; otherwise the standard three-panel layout
    is used.

    If `split` is True (CLI: --split) AND no single-quantity filter was set,
    each quantity is ALSO written to its own one-page standalone PDF named
        <out_stem>_<key>.pdf
    next to `out_path`, plus one per polarisation as
        <out_stem>_<key>_<polTag>.pdf
    Default is split=False, so a bare invocation produces only the
    multi-page parent.

    If `write_multipage` is False (CLI: --no-multipage), the multi-page
    parent PDF is NOT written; only the standalones are.  `out_path` is
    then used only to derive the standalone filename stem (and the directory
    they land in).  Typically combined with `--split` to get just the
    individual plots.
    """
    data = load_scan(in_path)
    _add_derived(data)
    x_key, y_key = detect_axes(data)
    fixed_str = _fixed_input_summary(data, x_key, y_key)

    if only_quantity is not None:
        if only_quantity not in LABELS:
            raise SystemExit(
                f"unknown quantity {only_quantity!r}. "
                f"Available: {', '.join(LABELS.keys())}"
            )
        if only_quantity not in data:
            raise SystemExit(
                f"quantity {only_quantity!r} not present in {in_path}"
            )
        keys_to_plot = [only_quantity]
    else:
        keys_to_plot = [k for k in LABELS if k in data]

    # Standalone-PDF setup.  Skip when the multi-page output already covers
    # exactly one quantity (no point writing a redundant copy).
    write_standalones = split and (only_quantity is None)
    out_dir   = os.path.dirname(os.path.abspath(out_path))
    out_stem  = os.path.splitext(os.path.basename(out_path))[0]
    standalone_paths = []

    # Polarisations actually present in this data file.  We split per-pol
    # standalones across these (typically 1, 2, 3).  When the caller has
    # already filtered to a single pol via --pol, skip the per-pol split
    # (the per-quantity standalone IS the per-pol page already).
    data_pols = sorted(set(int(p) for p in data['pol']))
    do_per_pol_split = write_standalones and (only_pol is None)

    # When --no-multipage is set, replace the PdfPages context with a
    # nullcontext so the per-key loop below can still call `pdf.savefig`
    # via an `if pdf is not None` guard, without writing anything to disk.
    pdf_cm = (PdfPages(out_path) if write_multipage
              else contextlib.nullcontext(None))

    written = 0
    with pdf_cm as pdf:
        for key in keys_to_plot:
            # 1.  Combined-pol figure: 1 page in the multi-page PDF; if split
            #     is on, also written as a single-page standalone
            #     <stem>_<key>.pdf.
            fig = build_quantity_figure(
                data, key, x_key, y_key,
                only_pol=only_pol, fixed_str=fixed_str,
            )
            if fig is None:
                continue
            if pdf is not None:
                pdf.savefig(fig, dpi=300)
            if write_standalones:
                sa_path = os.path.join(out_dir, f'{out_stem}_{key}.pdf')
                with PdfPages(sa_path) as sa_pdf:
                    sa_pdf.savefig(fig, dpi=300)
                    sa_info = sa_pdf.infodict()
                    sa_info['Title']   = (
                        f'QI scan: {x_key} vs {y_key} [{key}]'
                    )
                    sa_info['Subject'] = (
                        f'ttZ density matrix QI quantity {key}, '
                        f'plane {COL_INFO[x_key]["label"]} - '
                        f'{COL_INFO[y_key]["label"]}'
                        + (f' (fixed: {fixed_str})' if fixed_str else '')
                    )
                standalone_paths.append(sa_path)
            plt.close(fig)

            # 2.  Per-polarisation single-panel standalones,
            #     <stem>_<key>_<polTag>.pdf.  Built by re-rendering the figure
            #     with only_pol=p, so the colour-scale, contour, and ticks
            #     stay identical to panel <p> of the combined figure.
            if do_per_pol_split:
                for p in data_pols:
                    fig_p = build_quantity_figure(
                        data, key, x_key, y_key,
                        only_pol=p, fixed_str=fixed_str,
                    )
                    if fig_p is None:
                        continue
                    pol_tag = POL_FILE_LABELS[p]
                    sa_p_path = os.path.join(
                        out_dir, f'{out_stem}_{key}_{pol_tag}.pdf')
                    with PdfPages(sa_p_path) as sa_p_pdf:
                        sa_p_pdf.savefig(fig_p, dpi=300)
                        sa_p_info = sa_p_pdf.infodict()
                        sa_p_info['Title']   = (
                            f'QI scan: {x_key} vs {y_key} '
                            f'[{key}, {pol_tag}]'
                        )
                        sa_p_info['Subject'] = (
                            f'ttZ density matrix QI quantity {key}, '
                            f'pol {pol_tag}, '
                            f'plane {COL_INFO[x_key]["label"]} - '
                            f'{COL_INFO[y_key]["label"]}'
                            + (f' (fixed: {fixed_str})' if fixed_str else '')
                        )
                    plt.close(fig_p)
                    standalone_paths.append(sa_p_path)

            written += 1

        if pdf is not None:
            info = pdf.infodict()
            x_lbl = COL_INFO[x_key]['label']
            y_lbl = COL_INFO[y_key]['label']
            tags = []
            if only_quantity is not None:
                tags.append(only_quantity)
            if only_pol is not None:
                tags.append(POL_FILE_LABELS[only_pol])
            suffix = f' [{", ".join(tags)}]' if tags else ''
            info['Title']   = f'QI scan: {x_key} vs {y_key}{suffix}'
            info['Subject'] = (
                f'ttZ density matrix QI quantities, plane {x_lbl} - {y_lbl}'
                + (f' (fixed: {fixed_str})' if fixed_str else '')
            )

    if write_multipage:
        print(f"Wrote {out_path} with {written} page"
              f"{'s' if written != 1 else ''} "
              f"(plane: {x_key} vs {y_key}"
              + (f", fixed {fixed_str}" if fixed_str else "")
              + (f", quantity {only_quantity}" if only_quantity else "")
              + (f", pol {only_pol} ({POL_FILE_LABELS[only_pol]})"
                 if only_pol is not None else "")
              + ")")
    else:
        print(f"No multi-page parent written (--no-multipage); "
              f"using {os.path.basename(out_path)} only as the standalone stem.")
    if standalone_paths:
        per_pol_note = (
            f", {out_stem}_<key>_<polTag>.pdf" if do_per_pol_split else ""
        )
        print(f"  + {len(standalone_paths)} standalone PDF"
              f"{'s' if len(standalone_paths) != 1 else ''}: "
              f"{out_stem}_<key>.pdf{per_pol_note}")


def _default_outpath(in_path, only_quantity=None, only_pol=None):
    """qi_scan[_<n>][_<quantity>][_<polTag>].pdf.

    The trailing scan-index (if any) on the input file is preserved, then a
    `_<quantity>` suffix is appended when only one quantity is being drawn,
    then a `_<polTag>` suffix when only one polarisation is being drawn
    (`eR`, `eL`, `unpol`).
    """
    base = os.path.basename(in_path)
    m = re.match(r'scan_results(?:_(\d+))?\.npz$', base)
    idx = m.group(1) if (m and m.group(1) is not None) else None
    parts = ['qi_scan']
    if idx is not None:
        parts.append(idx)
    if only_quantity is not None:
        parts.append(only_quantity)
    if only_pol is not None:
        parts.append(POL_FILE_LABELS[only_pol])
    return '_'.join(parts) + '.pdf'


def parse_args(argv):
    parser = argparse.ArgumentParser(
        prog='plot_qi.py',
        description=(
            'Plot QI quantities from scan_results_*.npz on the appropriate '
            '2D kinematic plane. With no flags, every quantity in LABELS is '
            'written as one page of a multi-page PDF.'
        ),
    )
    parser.add_argument(
        'input', nargs='?', default=None,
        help='scan_results_*.npz file. Omit to process every such file '
             'sitting next to this script.',
    )
    parser.add_argument(
        'output', nargs='?', default=None,
        help='Explicit output PDF path. Default is derived from the input '
             'filename (and from --quantity, if given).',
    )
    parser.add_argument(
        '-q', '--quantity', dest='quantity', default=None,
        choices=list(LABELS.keys()),
        metavar='NAME',
        help='Plot only this single QI quantity. Must be one of the keys of '
             'LABELS (see --list).',
    )
    parser.add_argument(
        '-p', '--pol', dest='pol', default=None, type=int, choices=[1, 2, 3],
        help='Plot only one polarisation: 1 = e_R e_L, 2 = e_L e_R, '
             '3 = unpolarised. Default: all three side-by-side.',
    )
    parser.add_argument(
        '-l', '--list', action='store_true',
        help='Print the supported quantity names and exit.',
    )
    parser.add_argument(
        '--split', dest='split', action='store_true', default=False,
        help='Also emit per-quantity (and per-polarisation) standalone '
             'PDFs next to the main output:  <out_stem>_<key>.pdf and '
             '<out_stem>_<key>_<polTag>.pdf.  Default: only the multi-page '
             'parent is written.',
    )
    parser.add_argument(
        '--no-multipage', dest='write_multipage', action='store_false',
        default=True,
        help='Skip writing the multi-page parent PDF. Only the per-quantity '
             '/ per-polarisation standalones are emitted. The positional '
             'output path is then used only to derive the standalone stems.',
    )
    return parser.parse_args(argv[1:])


def main(argv):
    args = parse_args(argv)

    if args.list:
        for k, v in LABELS.items():
            print(f'  {k:<10}  {v}')
        return

    here = os.path.dirname(os.path.abspath(__file__))

    if args.input is not None:
        in_path  = args.input
        out_path = (
            args.output
            or _default_outpath(in_path, args.quantity, args.pol)
        )
        if not os.path.isabs(in_path):
            in_path = os.path.join(here, in_path)
        if not os.path.isabs(out_path):
            out_path = os.path.join(here, out_path)
        process_file(in_path, out_path,
                     only_quantity=args.quantity, only_pol=args.pol,
                     split=args.split,
                     write_multipage=args.write_multipage)
        return

    # No input given: process every scan_results*.npz next to this script.
    pattern = os.path.join(here, 'scan_results*.npz')
    inputs = sorted(glob.glob(pattern))
    if not inputs:
        print(f"no scan_results*.npz files found in {here}")
        return

    if args.output is not None and len(inputs) > 1:
        raise SystemExit(
            "explicit output path is incompatible with multi-file mode "
            "(no input given). Pass an input file too."
        )

    for in_path in inputs:
        out_path = (
            args.output
            if args.output is not None
            else os.path.join(
                here, _default_outpath(in_path, args.quantity, args.pol)
            )
        )
        process_file(in_path, out_path,
                     only_quantity=args.quantity, only_pol=args.pol,
                     split=args.split,
                     write_multipage=args.write_multipage)


if __name__ == '__main__':
    main(sys.argv)
