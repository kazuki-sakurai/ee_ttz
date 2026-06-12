"""
Replot the heatmaps from scan_results_psinteg_realpol_<rs>.npz WITHOUT
re-running the SDP step.

This is the fast cosmetic-iteration sibling of qi_from_psinteg_realpol.py:
it imports plot_qi but NOT ppt_julia, so there is no ~30 s Julia/Mosek
warm-up.  Typical end-to-end runtime: 1-3 s per (rs, quantity) pair.

How to iterate
--------------
1. Tweak whatever you want in plot_qi.py -- LABELS, RANGES, LOG_SCALE,
   the colormap, font sizes, panel size, contour styling, etc.
2. Re-run this script:
        python3 plot_psinteg_realpol.py 0.6 1
3. Open the resulting PDF(s).  No Julia, no Mosek, no recompute of any
   of the QI quantities -- just a fresh render of whatever is already
   in scan_results_psinteg_realpol_*.npz.

If you change a quantity that requires the QI columns themselves to be
recomputed (e.g. you changed the gmn_hmg definition or the negativity
formula), use qi_from_psinteg_realpol.py --force to rebuild the npz
first, then come back here for cosmetic iteration.

Outputs
-------
For each rs label and each quantity in PLOT_QUANTITIES, one PDF
    qi_psinteg_realpol_<rs>_<quantity>.pdf
is written next to this script -- same name as qi_from_psinteg_realpol.py
produces, so this script is a drop-in replotter.

Usage
-----
    python3 plot_psinteg_realpol.py 0.6
    python3 plot_psinteg_realpol.py 0.6 1
    python3 plot_psinteg_realpol.py            # auto-discover every
                                               # scan_results_psinteg_realpol_*.npz
                                               # next to this script
    python3 plot_psinteg_realpol.py 0.6 -q GMN_HMG
                                               # one rs, one quantity
"""

import argparse
import glob
import os
import re
import sys

# NOTE: we import plot_qi but NOT ppt_julia / QI_functions, so this
# script has no Julia/Mosek runtime dependency.  All it does is read
# a finished npz and call plot_qi.process_file on it.
import plot_qi
import matplotlib.pyplot as plt


# ----------------------------------------------------------------------
# Font-size overrides for THIS pipeline only.
#
# `plot_qi.py` sets its own rcParams when it's imported (a global ~16.9 pt
# subplot title etc., tuned for its 3-panel layouts).  The single-panel
# realistic-pol plots produced here look better with a smaller title,
# so we override the relevant matplotlib rcParams *after* importing
# plot_qi.  None of plot_qi.py's other pipelines are affected.
#
# Edit these numbers freely.  Leave a key at `None` to keep plot_qi.py's
# original value for that field.
#
#   TITLE_FONTSIZE       : panel headline ("(P_e-, P_e+) = (+0.8, -0.3): sqrt(s)=...")
#   AXIS_LABEL_FONTSIZE  : m_tt / cos theta_Z and the colour-bar caption
#   TICK_LABEL_FONTSIZE  : tick numbers on the axes and the colour bar
#   FIGURE_TITLE_FONTSIZE: matplotlib figure suptitle (currently unused
#                          by plot_qi; here for completeness)
# ----------------------------------------------------------------------
TITLE_FONTSIZE        = 14.0     # was ~16.9 pt in plot_qi.py
AXIS_LABEL_FONTSIZE   = None
TICK_LABEL_FONTSIZE   = None
FIGURE_TITLE_FONTSIZE = None

if TITLE_FONTSIZE is not None:
    plt.rcParams['axes.titlesize']   = TITLE_FONTSIZE
if AXIS_LABEL_FONTSIZE is not None:
    plt.rcParams['axes.labelsize']   = AXIS_LABEL_FONTSIZE
if TICK_LABEL_FONTSIZE is not None:
    plt.rcParams['xtick.labelsize']  = TICK_LABEL_FONTSIZE
    plt.rcParams['ytick.labelsize']  = TICK_LABEL_FONTSIZE
if FIGURE_TITLE_FONTSIZE is not None:
    plt.rcParams['figure.titlesize'] = FIGURE_TITLE_FONTSIZE


# ----------------------------------------------------------------------
# Colour-bar (z-axis) overrides for THIS pipeline only.
#
# Both dictionaries below are applied by mutating plot_qi.RANGES and
# plot_qi.LABELS BEFORE any plot is drawn.  They override plot_qi.py's
# defaults for the keys you uncomment, and only for this script (the
# other plot_qi-driven pipelines are unaffected, because they re-import
# plot_qi in their own process).
#
# CBAR_RANGES  : (vmin, vmax) tuples for the colour scale.  For LOG_SCALE
#                quantities (dsigma_fbperGeV) BOTH values must be > 0,
#                otherwise plot_qi prints a warning and auto-fits.  Set
#                an entry to `None` to let plot_qi auto-fit from the data.
#
# CBAR_LABELS  : LaTeX string drawn next to the colour bar.  Set an entry
#                to `None` to fall through to plot_qi.LABELS[key].
#
# Uncomment / edit any entry below and re-run the script -- ~2 s per
# rebuild because no SDP is recomputed.
# ----------------------------------------------------------------------
CBAR_RANGES = {
    'dsigma_fbperGeV': (5*10**-4, 0.1),       # log scale, fb/GeV
    # 'GMN_HMG':         (0.0, 0.5),       # linear, dimensionless
}

CBAR_LABELS = {
    'dsigma_fbperGeV': r'$\frac{\partial^{2}\sigma}{ \partial m_{t\bar t}\, \partial \cos\theta_{Z}}\;[\frac{\rm fb}{\rm GeV}]$',
    # 'GMN_HMG':         r'$\mathcal{N}_G$',
}

for _k, _v in CBAR_RANGES.items():
    if _v is not None:
        plot_qi.RANGES[_k] = _v

for _k, _v in CBAR_LABELS.items():
    if _v is not None:
        plot_qi.LABELS[_k] = _v


# ----------------------------------------------------------------------
# x / y axis limits (zoom into a sub-region of the kinematic plane).
#
# Keys are the column names in plot_qi.COL_INFO; tuples are the limits
# expressed in the PLOTTED coordinate (NOT the raw npz column):
#
#   'm12'  : m_{t tbar} in GeV         (plotted as-is)
#   'th3'  : cos(theta_Z), so in [-1, 1]   (npz stores th3 itself, but
#                                       plot_qi plots cos(theta_3) via
#                                       COL_INFO['th3']['transform'])
#   'cth1' : cos(theta_t)              (plotted as-is)
#   'ph'   : phi in radians            (plotted as-is)
#
# Comment a key out or set its value to None to keep matplotlib's
# auto-fit limits.  Only the keys that are the ACTUAL plot axes for the
# given npz (the two free outer variables; here m12 and th3) take
# effect -- entries for the other keys are silently ignored.
#
# --- sqrt(s)-dependent ranges ---
# Some axes (m_tt in particular) have a kinematic upper edge that
# depends on sqrt(s):
#     m_tt_max = sqrt(s) - m_Z   (here ~509 GeV at sqrt(s)=600 GeV,
#                                 ~909 GeV at sqrt(s)=1000 GeV)
# Put rs-INDEPENDENT defaults in AXIS_LIMITS below; put rs-SPECIFIC
# overrides in AXIS_LIMITS_BY_RS.  The wrapper merges them at plot
# time, with the per-rs entries winning.  Keys of AXIS_LIMITS_BY_RS
# are the SAME string labels you pass on the command line ('0.5',
# '0.6', '1' ...), matching psinteg.py's filename convention.
# ----------------------------------------------------------------------
AXIS_LIMITS = {
    'th3':  (-1.0, 1.0),            # cos(theta_Z); same range for all rs
    # 'cth1': (-1.0, 1.0),
    # 'ph':   (-3.1416, 3.1416),
}

AXIS_LIMITS_BY_RS = {
    #'0.6': {'m12': (350.0*(1+10**-4), 510.0-(510-350)/60)},   # m_tt in GeV at sqrt(s) = 600
    #'1':   {'m12': (350.0*(1+10**-4), 910.0-(910-350)/60)},   # m_tt in GeV at sqrt(s) = 1 TeV
    '0.6': {'m12': (350.0*(1+10**-4), 510.0*(1-10**-4))},   # m_tt in GeV at sqrt(s) = 600
    '1':   {'m12': (350.0*(1+10**-4), 910.0*(1-10**-4))},   # m_tt in GeV at sqrt(s) = 1 TeV
}


def _rs_label_from_GeV(rs_GeV):
    """600.0 -> '0.6', 1000.0 -> '1', 500.0 -> '0.5'.

    Round-trips with psinteg.py's filename convention: the rs label is
    the sqrt(s) value expressed in TeV via Python's :g format, so
    integer TeV values drop the trailing '.0'.
    """
    return f'{float(rs_GeV) / 1000.0:g}'


def _effective_limits_for(rs_GeV):
    """Merge AXIS_LIMITS (defaults) with AXIS_LIMITS_BY_RS[<label>]
    (per-rs overrides), per-rs entries winning.  Returns a fresh dict;
    callers should not mutate."""
    eff = {k: v for k, v in AXIS_LIMITS.items() if v is not None}
    label = _rs_label_from_GeV(rs_GeV)
    rs_specific = AXIS_LIMITS_BY_RS.get(label) or {}
    for k, v in rs_specific.items():
        if v is not None:
            eff[k] = v
    return eff


_have_axis_overrides = (
    any(v is not None for v in AXIS_LIMITS.values())
    or any(d for d in AXIS_LIMITS_BY_RS.values())
)
if _have_axis_overrides:
    import functools as _ft

    _orig_build_quantity_figure = plot_qi.build_quantity_figure

    @_ft.wraps(_orig_build_quantity_figure)
    def _build_with_axis_limits(data, key, x_key, y_key,
                                only_pol=None, fixed_str=''):
        fig = _orig_build_quantity_figure(
            data, key, x_key, y_key,
            only_pol=only_pol, fixed_str=fixed_str,
        )
        if fig is None:
            return fig

        # Look up the current rs from the data ('rs' column is set per
        # row in build_qi_file; it is constant within one npz).  Fall
        # back to NaN -> only the rs-independent defaults apply.
        rs_arr = data.get('rs')
        rs_GeV = float(rs_arr[0]) if (rs_arr is not None
                                      and len(rs_arr) > 0) else float('nan')
        eff = _effective_limits_for(rs_GeV)

        xlim = eff.get(x_key)
        ylim = eff.get(y_key)
        if xlim is None and ylim is None:
            return fig

        # plot_qi.build_quantity_figure builds the heatmap axes first,
        # then appends ONE colour-bar axes via fig.colorbar(...).  So
        # fig.axes[:-1] are all heatmap panels.
        for ax in fig.axes[:-1]:
            if xlim is not None:
                ax.set_xlim(*xlim)
            if ylim is not None:
                ax.set_ylim(*ylim)
        return fig

    plot_qi.build_quantity_figure = _build_with_axis_limits


# ----------------------------------------------------------------------
# Reference-box overlay drawn on EVERY plot produced by this script.
#
# Three dashed lines are added on top of each heatmap to delimit a
# rectangular sub-region of interest:
#
#   top    : (xstart, yhigh) -> (xend, yhigh)
#   bottom : (xstart, ylow)  -> (xend, ylow)
#   right  : (xend,   ylow)  -> (xend, yhigh)
#
# (No left edge -- the box opens to the left.)
#
# Fields:
#   'enabled'     : master on/off switch.  Set False to disable globally.
#   'xstart'      : left edge of the box (GeV).  Default 346  (= 2 m_t).
#   'xend_by_rs'  : {<rs label>: <right edge in GeV>, ...}.  rs labels
#                   without an entry are skipped (no overlay drawn).
#   'ylow','yhigh': y extents (cos theta_Z), default (-0.5, +0.5).
#   'kwargs'      : forwarded to ax.plot; default linestyle '--', yellow.
#   'only_keys'   : if not None, restrict to these quantity keys
#                   (e.g. {'GMN_HMG'}).  Default None -> every quantity.
#   'only_pols'   : if not None, restrict to these pol codes
#                   (e.g. {1, 2}).  Default None -> every pol setting.
#
# To turn the overlay OFF, set enabled=False (or comment the dict).
# To restrict it back to e.g. setI/GMN_HMG only, set
#   only_keys={'GMN_HMG'}, only_pols={1}.
# ----------------------------------------------------------------------
OVERLAY_BOX = dict(
    enabled=True,
    xstart=346.0,
    xend_by_rs={'0.6': 370.0, '1': 420.0},
    ylow=-0.5,
    yhigh=0.5,
    kwargs=dict(ls='--', c='yellow'),
    only_keys={'dsigma_fbperGeV', 'GMN_HMG'},  # skip N1..N3, N12..N23
    only_pols=None,        # None = every pol setting
)


if OVERLAY_BOX and OVERLAY_BOX.get('enabled', True):
    import functools as _ft_ov

    _prev_build_quantity_figure = plot_qi.build_quantity_figure  # possibly
                                                                  # already
                                                                  # wrapped

    @_ft_ov.wraps(_prev_build_quantity_figure)
    def _build_with_overlay(data, key, x_key, y_key,
                            only_pol=None, fixed_str=''):
        fig = _prev_build_quantity_figure(
            data, key, x_key, y_key,
            only_pol=only_pol, fixed_str=fixed_str,
        )
        if fig is None:
            return fig

        only_keys = OVERLAY_BOX.get('only_keys')
        only_pols = OVERLAY_BOX.get('only_pols')
        if only_keys is not None and key not in only_keys:
            return fig
        if only_pols is not None and only_pol not in only_pols:
            return fig

        rs_arr = data.get('rs')
        rs_GeV = float(rs_arr[0]) if (rs_arr is not None
                                      and len(rs_arr) > 0) else float('nan')
        rs_label = _rs_label_from_GeV(rs_GeV)
        xend = OVERLAY_BOX.get('xend_by_rs', {}).get(rs_label)
        if xend is None:
            return fig

        xstart = OVERLAY_BOX.get('xstart', 346.0)
        ylow   = OVERLAY_BOX.get('ylow', -0.5)
        yhigh  = OVERLAY_BOX.get('yhigh',  0.5)
        kw     = OVERLAY_BOX.get('kwargs', dict(ls='--', c='yellow'))

        # fig.axes[:-1] are heatmap panels (the colour bar is appended
        # last by fig.colorbar).  Once set_xlim/set_ylim has been called
        # by the axis-limits wrapper, autoscale is off, so the overlay
        # lines won't disturb the visible range even though xstart=346
        # may fall outside the trimmed xlim.
        for ax in fig.axes[:-1]:
            ax.plot([xstart, xend], [yhigh, yhigh], **kw)
            ax.plot([xstart, xend], [ylow,  ylow ], **kw)
            ax.plot([xend,   xend], [ylow,  yhigh], **kw)
        return fig

    plot_qi.build_quantity_figure = _build_with_overlay


# ----------------------------------------------------------------------
# Per-quantity polyline overlays.
#
# Each entry maps a quantity key (e.g. 'N1') to a LIST of polylines.
# A polyline is dict(points=[(x1,y1), (x2,y2), ...], kwargs={...}); each
# adjacent pair (x_k, y_k)->(x_{k+1}, y_{k+1}) becomes one line segment.
# Points are in the PLOTTED coordinates of the heatmap:
#       x = m_{t tbar} [GeV],  y = cos theta_Z (dimensionless).
#
# This is the right place to add per-quantity reference shapes (triangles,
# rectangles, custom contours).  It runs AFTER the rectangular OVERLAY_BOX
# wrapper above, so the two compose naturally (a plot can have both an
# OVERLAY_BOX rectangle AND OVERLAY_POLYLINES segments).
#
# Coordinates are absolute (m_tt in GeV), so they may extend past the
# axis limits at smaller sqrt(s) (e.g. the N1 triangle's (700, 1) vertex
# is outside the rs=0.6 panel that only goes up to m_tt ~ 510 GeV).
# Matplotlib clips silently; if you need different geometry per rs,
# duplicate the entry into a {rs_label: [polylines]} mapping and adapt
# the wrapper.
# ----------------------------------------------------------------------
# N3-fiducial rectangle, three sides (left edge open), drawn as a
# reference shape on the N1, N2 and N3 plots.  Edit the four corner
# points in one place; all three quantities pick up the change.
_N3_FIDUCIAL_LINES = [
    # top edge:    (346, +0.7) -> (550, +0.7)
    dict(points=[(346.0,  0.7), (550.0,  0.7)],
         kwargs=dict(ls='--', c='orange')),
    # right edge:  (550, +0.7) -> (550, -0.7)
    dict(points=[(550.0,  0.7), (550.0, -0.7)],
         kwargs=dict(ls='--', c='orange')),
    # bottom edge: (346, -0.7) -> (550, -0.7)
    dict(points=[(346.0, -0.7), (550.0, -0.7)],
         kwargs=dict(ls='--', c='orange')),
]

OVERLAY_POLYLINES = {
    # All three N plots currently show ONLY the N3-fiducial rectangle.
    # To restore the N1- / N2-fiducial triangles, insert them BEFORE the
    # _N3_FIDUCIAL_LINES splat below, e.g.:
    #   'N1': [
    #       dict(points=[(346.0, 0.75), (450.0, 1.0),
    #                    (700.0, 1.0), (346.0, -0.75)],
    #            kwargs=dict(ls='--', c='lime')),
    #       *_N3_FIDUCIAL_LINES,
    #   ],
    'N1': list(_N3_FIDUCIAL_LINES),
    'N2': list(_N3_FIDUCIAL_LINES),
    'N3': list(_N3_FIDUCIAL_LINES),
    # Cross section in fb/GeV: same lime N3-fiducial reference box.  Note
    # the yellow OVERLAY_BOX (above) is ALSO drawn on this plot, so the
    # dsigma panel ends up with two boxes -- yellow (OVERLAY_BOX) and
    # lime (N3-fiducial).  Remove 'dsigma_fbperGeV' from OVERLAY_BOX's
    # only_keys if you want only the lime one.
    'dsigma_fbperGeV': list(_N3_FIDUCIAL_LINES),
}


if OVERLAY_POLYLINES:
    import functools as _ft_pl

    _prev_build_for_polylines = plot_qi.build_quantity_figure

    @_ft_pl.wraps(_prev_build_for_polylines)
    def _build_with_polylines(data, key, x_key, y_key,
                              only_pol=None, fixed_str=''):
        fig = _prev_build_for_polylines(
            data, key, x_key, y_key,
            only_pol=only_pol, fixed_str=fixed_str,
        )
        if fig is None:
            return fig

        polys = OVERLAY_POLYLINES.get(key)
        if not polys:
            return fig

        # Same axes-selection trick as the OVERLAY_BOX wrapper: the colour
        # bar is the last axes added by fig.colorbar, so heatmap panels
        # are fig.axes[:-1].  set_xlim/set_ylim have already been applied
        # by _build_with_axis_limits, so plotting outside the trimmed
        # window just gets clipped without disturbing the visible range.
        for ax in fig.axes[:-1]:
            for poly in polys:
                pts = poly.get('points', [])
                kw  = poly.get('kwargs', dict(ls='--', c='lime'))
                if len(pts) < 2:
                    continue
                xs = [p[0] for p in pts]
                ys = [p[1] for p in pts]
                ax.plot(xs, ys, **kw)
        return fig

    plot_qi.build_quantity_figure = _build_with_polylines


# ----------------------------------------------------------------------
# Polarisation settings to plot.  Each entry says:
#
#   'label'  : LaTeX-style panel title (drawn ABOVE the heatmap by
#              plot_qi, via the COL_TITLES override below).
#   'tag'    : short, filesystem-safe suffix used by the per-pol output
#              filename (e.g. 'setI' -> qi_psinteg_realpol_0.6_GMN_HMG_setI.pdf).
#   'source' : format string -- which scan_results_*.npz to read.  The
#              `{rs}` placeholder is filled with the rs label
#              ('0.5', '0.6', '1', ...).
#
# pol = 1 -> setting i  (P_e- = +0.8, P_e+ = -0.3)  [realpol npz]
# pol = 2 -> setting ii (P_e- = -0.8, P_e+ = +0.3)  [realpol npz]
# pol = 3 -> unpolarised                            [regular psinteg npz]
#
# The realpol npz is produced by qi_from_psinteg_realpol.py.  The
# regular psinteg npz is produced by qi_from_psinteg.py and ALREADY
# carries an 'unpolarised' row as pol=3, so the plot just consumes
# what's already on disk -- no extra Julia/Mosek work.
#
# Remove any entry to skip that plot (e.g. comment out pol=3 if you
# don't want the unpolarised heatmaps).
# ----------------------------------------------------------------------
POL_SETTINGS = {
    1: dict(label=r'$(\mathcal{P}_{e^-},\,\mathcal{P}_{e^+}) = (+0.8,\,-0.3)$',
            tag='setI',
            source='scan_results_psinteg_realpol_{rs}.npz'),
    2: dict(label=r'$(\mathcal{P}_{e^-},\,\mathcal{P}_{e^+}) = (-0.8,\,+0.3)$',
            tag='setII',
            source='scan_results_psinteg_realpol_{rs}.npz'),
    3: dict(label=r'Unpol',
            tag='unpol',
            source='scan_results_psinteg_{rs}.npz'),
}

# Quantities to draw.  Add/remove keys here to control what gets
# rendered.  Any key in plot_qi.LABELS is fair game; if a key is not
# present in the source npz, plot_qi.process_file raises SystemExit
# (so this set should match what qi_from_psinteg_realpol.py wrote).
#
# The N* keys are LINEAR negativities, built by plot_qi._add_derived()
# from the EN* log-negativities stored in the npz, via
#     N = (2^{E^N} - 1) / 2          (note eq 4.9)
# so no extra columns need to be saved to plot them.  Naming:
#   N1, N2, N3   : single-subsystem-vs-rest negativities
#                  N1 = N(t | tbar Z), N2 = N(tbar | t Z), N3 = N(Z | t tbar)
#   N12, N13, N23: pairwise bipartite negativities on the reduced 2-body rhos
#                  N12 = N_{t tbar}, N13 = N_{t Z}, N23 = N_{tbar Z}
PLOT_QUANTITIES = (
    'dsigma_fbperGeV', 'GMN_HMG',
    'N1', 'N2', 'N3',           # one-subsystem-vs-rest negativities
    'N12', 'N13', 'N23',        # pairwise negativities
)


# ----------------------------------------------------------------------
def _override_panel_titles():
    """Patch plot_qi.COL_TITLES so the (single) panel reads the
    realistic beam-setting label rather than plot_qi.py's default
    pure-helicity label, and patch plot_qi.POL_FILE_LABELS so any
    plot_qi-side default filename also picks up our tag (this script
    doesn't rely on it -- we pass an explicit out_path below -- but
    it keeps things consistent if you later wire in --split / -p).
    Idempotent."""
    for k, v in POL_SETTINGS.items():
        plot_qi.COL_TITLES[k]      = v['label']
        plot_qi.POL_FILE_LABELS[k] = v['tag']


def _discover_rs_labels(here):
    """Return every rs label for which AT LEAST ONE source npz exists
    next to this script (either scan_results_psinteg_<rs>.npz or
    scan_results_psinteg_realpol_<rs>.npz), sorted numerically when
    possible.

    The realpol regex is anchored against the longer prefix so a
    realpol file is NOT also picked up by the regular regex (which
    insists on a single non-underscore label segment).
    """
    rx_realpol = re.compile(
        r'^scan_results_psinteg_realpol_(?P<rs>[^_]+)\.npz$')
    rx_regular = re.compile(
        r'^scan_results_psinteg_(?P<rs>[^_]+)\.npz$')

    labels = set()
    for p in glob.glob(os.path.join(here, 'scan_results_psinteg*.npz')):
        base = os.path.basename(p)
        m = rx_realpol.match(base) or rx_regular.match(base)
        if m:
            labels.add(m.group('rs'))

    def _key(s):
        try:
            return float(s)
        except ValueError:
            return float('inf')
    return sorted(labels, key=_key)


def run_plots(rs_label, here, quantities):
    """For each (quantity, polarisation) pair, write a SINGLE-panel PDF.

    Filenames follow:
        qi_psinteg_realpol_<rs>_<quantity>_<tag>.pdf
    where <tag> comes from POL_SETTINGS[pol]['tag'] above.  Each PDF
    is one page with one panel.

    Each pol setting reads from the source file named in
    POL_SETTINGS[pol]['source'].format(rs=rs_label); a missing source
    is skipped with a warning instead of raising (so e.g. you can run
    this script on a directory that has only the realpol npz, or only
    the regular psinteg npz, and still get the matching plots).
    """
    print(f"==> rs label '{rs_label}'")
    any_plotted = False
    for pol_code, pol_info in POL_SETTINGS.items():
        src_name = pol_info['source'].format(rs=rs_label)
        src_path = os.path.join(here, src_name)
        if not os.path.isfile(src_path):
            print(f"  ! missing {src_name}, skipping pol={pol_code} "
                  f"({pol_info['tag']}).  Generate it with "
                  f"{'qi_from_psinteg_realpol.py' if 'realpol' in src_name else 'qi_from_psinteg.py'} "
                  f"{rs_label} first.")
            continue
        for q in quantities:
            out_pdf = os.path.join(
                here,
                f'qi_psinteg_realpol_{rs_label}_{q}_{pol_info["tag"]}.pdf',
            )
            plot_qi.process_file(
                src_path, out_pdf,
                only_quantity=q,
                only_pol=pol_code,           # restrict to ONE panel
            )
            any_plotted = True
    if not any_plotted:
        print(f"  !! no source npz found for rs={rs_label}; "
              f"nothing rendered.")
    print()


def parse_args(argv):
    parser = argparse.ArgumentParser(
        prog='plot_psinteg_realpol.py',
        description=(
            'Replot scan_results_psinteg_realpol_<rs>.npz with NO Julia '
            'recompute.  Use for fast cosmetic iteration on plot_qi.py.'
        ),
    )
    parser.add_argument(
        'rs_labels', nargs='*',
        help='One or more rs labels (e.g. 0.6 1).  Omit to process every '
             'scan_results_psinteg_realpol_*.npz next to this script.',
    )
    parser.add_argument(
        '-q', '--quantity', dest='quantity', default=None,
        metavar='NAME',
        help='Render only this single quantity (must be a key of '
             'plot_qi.LABELS).  Default: render every key listed in '
             'PLOT_QUANTITIES at the top of this script.',
    )
    return parser.parse_args(argv[1:])


def main(argv):
    args = parse_args(argv)
    here = os.path.dirname(os.path.abspath(__file__))

    if args.rs_labels:
        rs_labels = args.rs_labels
    else:
        rs_labels = _discover_rs_labels(here)
        if not rs_labels:
            raise SystemExit(
                f"no scan_results_psinteg_realpol_*.npz files next to "
                f"{os.path.basename(__file__)}; run "
                f"qi_from_psinteg_realpol.py first."
            )
        print(f"Auto-discovered {len(rs_labels)} rs label(s): "
              f"{rs_labels}\n")

    quantities = (
        (args.quantity,) if args.quantity is not None else PLOT_QUANTITIES
    )

    _override_panel_titles()
    for rs in rs_labels:
        run_plots(rs, here, quantities)


if __name__ == '__main__':
    main(sys.argv)
