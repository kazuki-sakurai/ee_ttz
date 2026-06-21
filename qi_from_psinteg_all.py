"""
Compute every QI quantity from the (full phase space)-integrated R-matrices
written by psinteg_all.py, and produce 1-D line plots of each QI quantity
vs sqrt(s).

Beam polarisation
-----------------
For arbitrary beam polarisations (P_{e-}, P_{e+}) the unnormalised R-matrix
and cross section combine the two SM-allowed pure helicity states via
eq (2.9) of the draft note:

    R[P_e-, P_e+] = sum_{l1, l2} P_{l1, l2} R^{l1, l2}
                  = P_{+-} R^{+-} + P_{-+} R^{-+}
    P_{+-} = (1/4)(1 + P_{e-})(1 - P_{e+})
    P_{-+} = (1/4)(1 - P_{e-})(1 + P_{e+})

(the (+,+) and (-,-) terms vanish in the massless-electron SM).  This
script encodes the chosen settings in POL_SETTINGS at the top of the file;
edit there to scan other (P_{e-}, P_{e+}) configurations.

Pipeline:
    1. Load data/psinteg_all_res.npz  (R1 = R^{+-}, R2 = R^{-+}, rs and --
       if present -- sigma_eR_pb = sigma^{+-}, sigma_eL_pb = sigma^{-+}).
    2. For every sqrt(s) x beam-setting build the linear mixture
            R[setting] = P_{+-}(setting) R1 + P_{-+}(setting) R2
       and the spin density matrix rho = R / Tr(R).
    3. Compute every QI quantity using the same formulas as analysis.py.
    4. Save the per-(rs, pol) results to qi_from_psinteg_all_res.npz.  The
       per-row cross section is stored as `sigma_fb` (in femtobarns) so
       plot_qi_vs_rs.py can render it alongside the QI plots.
    5. Subprocess plot_qi_vs_rs.py on that file to produce
       qi_from_psinteg_all.pdf.

Usage:
    python3 qi_from_psinteg_all.py
    python3 qi_from_psinteg_all.py --no-plot     # only build the QI npz
    python3 qi_from_psinteg_all.py --force       # regenerate the QI npz
                                                 # even if it already exists
                                                 # (default: reuse it)

NOTE: if you already have qi_from_psinteg_all_res.npz from a prior run
(with the old pure-polarisation rows), pass --force to recompute with the
realistic beam settings configured below.
"""

import os
import sys
import subprocess
import numpy as np
from math import sqrt

sys.path.append('/Users/kazuki/Projects/pyHELAS')
from QI_functions import *           # normalise, purity, concurrence,
                                     # log_neg_bip, log_negativity
from ee_ttz import *                 # mt, mZ, ...
from ee_ttz_func import *
from ppt_cvxpy import get_GMN as _get_GMN
def gmn_hmg(rho):
    return _get_GMN(rho, dims=[2, 2, 3])        

# Same QI list as analysis.py / qi_from_psinteg.py.
ENTRIES    = ['pure', 'pure1', 'pure2', 'pure3',
              'c12', 'EN12', 'EN13', 'EN23',
              'c1', 'c2', 'c3', 'EN1', 'EN2', 'EN3',
              'GMN_HMG']
INPUT_KEYS = ['rs', 'pol']


# ----------------------------------------------------------------------
# Beam polarisation settings.  Each row is one curve in the downstream
# plots; the integer key is the `pol` column stored in the output npz, and
# downstream plot_qi_vs_rs.py keys its legend off the same integer.  The
# (Pe_minus, Pe_plus) values follow eq (2.8)-(2.9) of the draft note:
#
#     rho_{e+-} = (1/2)(1 + P_{e+-}) |+><+|_{e+-}
#               + (1/2)(1 - P_{e+-}) |-><-|_{e+-}
#
# so P_{e-} = +1 means a 100% right-handed e- beam; P_{e+} = 0 means an
# unpolarised positron beam, etc.  Settings 1 and 2 are the two realistic
# ILC-like configurations; setting 3 is the unpolarised baseline.
# ----------------------------------------------------------------------
POL_SETTINGS = {
    1: dict(label='setting i  (P_e-=+0.8, P_e+=-0.3)',
            Pe_minus= 0.8, Pe_plus=-0.3),
    2: dict(label='setting ii (P_e-=-0.8, P_e+=+0.3)',
            Pe_minus=-0.8, Pe_plus= 0.3),
    3: dict(label='unpolarised',
            Pe_minus= 0.0, Pe_plus= 0.0),
}


def _polarised_weights(Pe_minus, Pe_plus):
    """Joint initial-helicity probabilities P_{lambda_1, lambda_2} from
    eq (2.9) of the draft note, for the two SM-surviving combinations.

    Returns (P_pm, P_mp) where
        P_pm = P_{+,-} = (1/4)(1 + P_{e-})(1 - P_{e+})    # e_R e_L
        P_mp = P_{-,+} = (1/4)(1 - P_{e-})(1 + P_{e+})    # e_L e_R
    The (+,+) and (-,-) terms are zero in the massless-electron SM, hence
    not returned.
    """
    P_pm = 0.25 * (1.0 + Pe_minus) * (1.0 - Pe_plus)
    P_mp = 0.25 * (1.0 - Pe_minus) * (1.0 + Pe_plus)
    return P_pm, P_mp


# Conversion: 1 pb = 1000 fb.  Used to express the stored cross section in
# femtobarns (sigma_fb column of the output npz).
PB_TO_FB = 1000.0


def compute_qi(rho):
    """All QI quantities for one 12x12 rho on subsystems (t, tbar, Z) with
    Hilbert-space dimensions (2, 2, 3)."""
    q = {}
    q['pure'] = purity(rho)

    rho_t = rho.reshape(2, 2, 3, 2, 2, 3)
    rho1  = np.einsum('xijxab->ijab', rho_t).reshape(6, 6)   # rho_{tbar Z}
    rho2  = np.einsum('ixjaxb->ijab', rho_t).reshape(6, 6)   # rho_{t Z}
    rho3  = np.einsum('ijxabx->ijab', rho_t).reshape(4, 4)   # rho_{t tbar}

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


def build_qi_file():
    here = os.path.dirname(os.path.abspath(__file__))
    src  = os.path.join(here, 'data', 'psinteg_all_res.npz')
    d    = np.load(src)
    R1, R2, rs_arr = d['R1'], d['R2'], d['rs']
    Nrs = R1.shape[0]
    print(f"Loaded {src}: {Nrs} sqrt(s) points, R shape = {R1.shape}")

    # Pure-polarisation cross sections (in pb) written by psinteg_all.py:
    # sigma_eR_pb = sigma^{+-}, sigma_eL_pb = sigma^{-+}.  We mix these
    # with the realistic beam weights below.  Fall back to NaN if the
    # source npz pre-dates that addition.
    def _sigma(key):
        return d[key] if key in d.files else np.full(Nrs, np.nan)
    sigma_eR_pb = _sigma('sigma_eR_pb')     # sigma^{+-}  (in pb)
    sigma_eL_pb = _sigma('sigma_eL_pb')     # sigma^{-+}  (in pb)
    have_sigma  = ('sigma_eR_pb' in d.files) and ('sigma_eL_pb' in d.files)
    if not have_sigma:
        print("  pure-pol cross sections NOT in source npz; sigma_fb -> NaN.")
        print("  Regenerate data/psinteg_all_res.npz with the up-to-date "
              "psinteg_all.py to populate them.")

    # Build the per-setting R-matrix stacks and per-setting cross sections.
    # Each entry is an (Nrs, 12, 12) complex array for R and an (Nrs,)
    # float array for sigma (in fb).
    R_by_pol     = {}
    sigma_by_pol = {}
    for pol_code, setting in POL_SETTINGS.items():
        Pe_m, Pe_p = setting['Pe_minus'], setting['Pe_plus']
        P_pm, P_mp = _polarised_weights(Pe_m, Pe_p)

        R_by_pol[pol_code] = P_pm * R1 + P_mp * R2
        sigma_by_pol[pol_code] = (
            (P_pm * sigma_eR_pb + P_mp * sigma_eL_pb) * PB_TO_FB
        )
        print(f"  pol {pol_code}: {setting['label']}  "
              f"-> weights (P_+-, P_-+) = ({P_pm:.4f}, {P_mp:.4f})")

    records = {k: [] for k in INPUT_KEYS + ENTRIES + ['sigma_fb']}
    rho_all = []

    for pol_code in POL_SETTINGS:
        Rstack = R_by_pol[pol_code]
        sigmas = sigma_by_pol[pol_code]
        for i in range(Nrs):
            rho = normalise(Rstack[i])
            q   = compute_qi(rho)
            records['rs'].append(float(rs_arr[i]))
            records['pol'].append(pol_code)
            records['sigma_fb'].append(float(sigmas[i]))
            for e in ENTRIES:
                records[e].append(q[e])
            rho_all.append(rho)
        print(f"  pol = {pol_code} ({POL_SETTINGS[pol_code]['label']}): "
              f"done ({Nrs} points)")

    out_npz = os.path.join(here, 'qi_from_psinteg_all_res.npz')
    np.savez_compressed(
        out_npz,
        rho=np.array(rho_all, dtype=np.complex128),
        **{k: np.array(v) for k, v in records.items()},
    )
    print(f"Wrote {out_npz} with {len(rho_all)} (rs x pol) rows")
    return out_npz


def run_plot(npz_path):
    """Drive plot_qi_vs_rs.py on the QI npz."""
    here    = os.path.dirname(os.path.abspath(__file__))
    plot_py = os.path.join(here, 'plot_qi_vs_rs.py')
    out_pdf = os.path.join(here, 'qi_from_psinteg_all.pdf')
    subprocess.run(
        [sys.executable, plot_py, npz_path, out_pdf],
        check=True,
    )


def main(argv):
    flags   = [a for a in argv[1:] if a.startswith('-')]
    do_plot = '--no-plot' not in flags
    force   = ('--force' in flags) or ('-f' in flags)

    here     = os.path.dirname(os.path.abspath(__file__))
    npz_path = os.path.join(here, 'qi_from_psinteg_all_res.npz')

    # Skip the (Julia/Mosek-heavy) QI build step when the npz is already on
    # disk; pass --force / -f to override and recompute from scratch.
    if os.path.exists(npz_path) and not force:
        print(f"Using existing {npz_path}  (pass --force to regenerate)")
    else:
        npz_path = build_qi_file()

    if do_plot:
        run_plot(npz_path)


if __name__ == '__main__':
    main(sys.argv)
