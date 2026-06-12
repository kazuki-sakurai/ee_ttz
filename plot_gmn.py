"""
Plot GMN_HMG on the (m_tt, cos(theta_3)) plane for each polarisation,
using scan_results.npz produced by analysis.py. Output is a PDF.

Usage:
    python3 plot_gmn.py                                # writes gmn_scan.pdf
    python3 plot_gmn.py scan_results.npz               # explicit input
    python3 plot_gmn.py scan_results.npz gmn_scan.pdf  # explicit output too
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt

from load_scan import load_scan, select


POL_LABELS = {
    1: r"$e^-_R$ (pol 1)",
    2: r"$e^-_L$ (pol 2)",
    3: r"unpolarised (pol 3)",
}


def _grid(sub):
    """Reshape a 1-D scan on a regular (m12, cth3) grid into 2-D arrays."""
    m12 = sub['m12']
    cth3 = np.cos(sub['th3'])     # th3 was saved as acos(cth3)

    m12_axis = np.unique(m12)
    cth3_axis = np.unique(cth3)

    nx, ny = m12_axis.size, cth3_axis.size
    if nx * ny != m12.size:
        raise RuntimeError(
            f"non-regular grid: {m12.size} points but axes have sizes "
            f"{nx} (m12) x {ny} (cth3)"
        )

    ix = np.searchsorted(m12_axis, m12)
    iy = np.searchsorted(cth3_axis, cth3)

    Z = np.full((ny, nx), np.nan)
    Z[iy, ix] = sub['GMN_HMG']
    return m12_axis, cth3_axis, Z


def main(in_path='scan_results.npz', out_path='gmn_scan.pdf'):
    here = os.path.dirname(os.path.abspath(__file__))
    if not os.path.isabs(in_path):
        in_path = os.path.join(here, in_path)
    if not os.path.isabs(out_path):
        out_path = os.path.join(here, out_path)

    data = load_scan(in_path)
    pols = sorted(set(int(p) for p in data['pol']))

    # Common color scale across all panels.
    vmin = float(data['GMN_HMG'].min())
    vmax = float(data['GMN_HMG'].max())

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
        m12_axis, cth3_axis, Z = _grid(sub)

        mesh = ax.pcolormesh(
            m12_axis, cth3_axis, Z,
            cmap='viridis', shading='auto',
            vmin=vmin, vmax=vmax,
        )
        ax.set_xlabel(r'$m_{t\bar{t}}$  [GeV]')
        ax.set_title(POL_LABELS.get(p, f'pol {p}'))
        ax.grid(False)

    axes[0].set_ylabel(r'$\cos\theta_3$')

    cbar = fig.colorbar(mesh, ax=axes, shrink=0.95, pad=0.02)
    cbar.set_label(r'$\mathcal{N}_\mathrm{GMN}^\mathrm{HMG}(\rho)$')

    fig.suptitle(
        r'Genuine multipartite negativity (HMG 2014)'
        r' on the $(m_{t\bar t},\,\cos\theta_3)$ plane'
    )

    fig.savefig(out_path, format='pdf')
    print(f"Wrote {out_path}")


if __name__ == '__main__':
    in_path = sys.argv[1] if len(sys.argv) > 1 else 'scan_results.npz'
    out_path = sys.argv[2] if len(sys.argv) > 2 else 'gmn_scan.pdf'
    main(in_path, out_path)
