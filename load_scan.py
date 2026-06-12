"""
Loader for scan_results.npz produced by analysis.py.

Each scan point (one row of the flat arrays) corresponds to a fixed
combination of (pol, m12, cth3 -> th3, cth1, ph). The full 12x12 density
matrix for each point lives in `data['rho']`.

Run this file directly to print a summary, or import `load_scan` from your
own scripts.
"""

import os
import numpy as np


SCALAR_KEYS = [
    'rs', 'm12', 'th3', 'cth1', 'ph', 'pol',
    'pure', 'pure1', 'pure2', 'pure3',
    'c1', 'c2', 'c3', 'c12',
    'EN1', 'EN2', 'EN3', 'EN12', 'EN13', 'EN23',
    'GMN_HMG',
]


def load_scan(path='scan_results.npz'):
    """
    Load a scan file produced by analysis.py.

    Returns
    -------
    data : dict
        Dictionary mapping each saved key to its numpy array. Scalars are 1-D
        arrays of length N = number of scan points; `rho` has shape (N,12,12)
        complex.
    """
    if not os.path.isabs(path):
        path = os.path.join(os.path.dirname(os.path.abspath(__file__)), path)

    npz = np.load(path)
    data = { k: npz[k] for k in npz.files }
    npz.close()
    return data


def select(data, **filters):
    """
    Slice the loaded scan to the rows that match every filter, e.g.

        select(data, pol=1)
        select(data, pol=1, m12=lambda x: x > 500)

    Each filter is either a value to match exactly, or a callable that returns
    a boolean mask when applied to the column.
    """
    n = len(data['pol'])
    mask = np.ones(n, dtype=bool)
    for key, value in filters.items():
        col = data[key]
        if callable(value):
            mask &= value(col)
        else:
            mask &= (col == value)
    return { k: v[mask] for k, v in data.items() }


def summarise(data):
    """Print a short overview of what's in the loaded scan."""
    n = len(data['pol'])
    print(f"# of scan points : {n}")
    print(f"rho shape        : {data['rho'].shape}  (dtype {data['rho'].dtype})")
    print(f"polarisations    : {sorted(set(int(p) for p in data['pol']))}")
    print(f"m12 range        : [{data['m12'].min():.3f}, {data['m12'].max():.3f}]")
    print(f"th3 range  (rad) : [{data['th3'].min():.3f}, {data['th3'].max():.3f}]")
    print()

    print(f"{'quantity':<10} {'min':>12} {'max':>12} {'mean':>12}")
    print('-' * 50)
    for k in SCALAR_KEYS:
        if k in data and np.issubdtype(data[k].dtype, np.number):
            col = data[k]
            print(f"{k:<10} {col.min():12.5f} {col.max():12.5f} {col.mean():12.5f}")


if __name__ == '__main__':
    import sys
    path = sys.argv[1] if len(sys.argv) > 1 else 'scan_results.npz'
    data = load_scan(path)

    print(f"Loaded {path}\n")
    summarise(data)

    # Quick sanity checks on rho.
    rho = data['rho']
    traces = np.einsum('nii->n', rho)
    hermiticity = np.max(np.abs(rho - rho.conj().transpose(0, 2, 1)), axis=(1, 2))
    print()
    print(f"max |1 - tr(rho)|              : {np.max(np.abs(traces - 1)):.2e}")
    print(f"max |rho - rho^dagger|         : {np.max(hermiticity):.2e}")

    # Example: GMN by polarisation, averaged over the (m12, cth3) grid.
    print("\nMean GMN_HMG by polarisation:")
    for p in sorted(set(int(x) for x in data['pol'])):
        sub = select(data, pol=p)
        print(f"  pol = {p}:  <GMN_HMG> = {sub['GMN_HMG'].mean():.4f}"
              f"   (max = {sub['GMN_HMG'].max():.4f})")

    # Example: pull rho for the first pol=1 point with the largest GMN.
    sub1 = select(data, pol=1)
    if len(sub1['pol']):
        i = int(np.argmax(sub1['GMN_HMG']))
        print(f"\nLargest GMN_HMG at pol=1 : "
              f"m12={sub1['m12'][i]:.3f}, th3={sub1['th3'][i]:.3f},"
              f" GMN={sub1['GMN_HMG'][i]:.4f}")
        print("rho at that point (top-left 4x4 block):")
        print(np.array_str(sub1['rho'][i][:4, :4], precision=4, suppress_small=True))
