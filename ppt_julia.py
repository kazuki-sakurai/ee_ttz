"""
Python wrapper around the Julia Entanglement module living in
quantum-correlations/src/entanglement.

Provides `ppt_visibilities(rho)`, which takes a 12x12 numpy density matrix on
a (qubit, qubit, qutrit) Hilbert space (dims [2, 2, 3], i.e. the ttZ spin
density matrix) and returns the PPT and PPT-mixture visibilities in four
partitions:

    - genuine multipartite (PPT / PPT_mixer applied to the full tripartite
      state; this matches the first block of particle_entanglement_example.jl)
    - A|BC     (dims [2, 6])
    - AB|C     (dims [4, 3])
    - AC|B     (dims [6, 2], obtained by permuting subsystems 2 and 3)

Requirements
------------
* `pip install juliacall`
* a working Julia install with the same packages used by
  particle_entanglement_example.jl: MultiStates (local), Entanglement (local),
  Convex, Mosek, MosekTools, IterTools, plus a valid Mosek license.

Julia is started lazily on the first call and reused, so the SDPs are solved
in-process (no per-call subprocess overhead).
"""

import os
import numpy as np

# -------------------------------------------------------------------
# Lazy Julia initialisation
# -------------------------------------------------------------------

_JL = None  # juliacall.Main once initialised

# By default look for quantum-correlations next to this file.
DEFAULT_QC_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "quantum-correlations",
)


def _init_julia(qc_dir=DEFAULT_QC_DIR):
    """Initialise Julia + load MultiStates / Entanglement only once."""
    global _JL
    if _JL is not None:
        return _JL

    from juliacall import Main as jl  # imported lazily so analysis.py can
                                      # still be imported without juliacall

    ent_dir = os.path.join(qc_dir, "src", "entanglement")
    if not os.path.isdir(ent_dir):
        raise RuntimeError(
            "Could not find quantum-correlations/src/entanglement at "
            f"{ent_dir!r}. Pass qc_dir=... to ppt_julia.ppt_visibilities."
        )

    # startup.jl uses pwd() to set LOAD_PATH, so cd into the entanglement dir
    # before including it -- exactly as particle_entanglement_example.jl does.
    jl.seval(f'cd(raw"{ent_dir}")')
    jl.seval('include("startup.jl")')
    jl.seval('include("Entanglement.jl")')
    jl.seval("using MultiStates")
    jl.seval("using LinearAlgebra")
    jl.seval("const Ent = Main.Entanglement")

    # Julia helpers. Doing the work Julia-side avoids round-tripping MultiState
    # objects through the Python boundary for every method.
    jl.seval(r"""
        # Run `f()` with stdout/stderr redirected to /dev/null so that Mosek
        # warnings (e.g. MSK_RES_WRN_ZEROS_IN_SPARSE_ROW) and the
        # PrimalRobustnessToPPTMixture status prints don't spam the terminal.
        function _silenced(f)
            open("/dev/null", "w") do dn
                redirect_stdout(dn) do
                    redirect_stderr(dn) do
                        f()
                    end
                end
            end
        end

        # Hofmann-Moroder-Guhne 2014 genuine multipartite negativity only,
        # for a (qubit, qubit, qutrit) state with dims [2, 2, 3].
        function _gmn_hmg(M_in)
            M = Matrix{ComplexF64}(M_in)
            return _silenced() do
                rho_full = MultiState(M, [2, 2, 3])
                Float64(Ent.EntanglementRobustness(rho_full, method="HMG"))
            end
        end

        function _ppt_visibilities(M_in)
            # juliacall passes numpy arrays as PyArray, so coerce to a
            # plain Matrix{ComplexF64} that the rest of the code expects.
            M = Matrix{ComplexF64}(M_in)

            return _silenced() do
                # full tripartite (matches the genuine-multipartite block in
                # particle_entanglement_example.jl)
                rho_full   = MultiState(M, [2, 2, 3])
                t_gm_mix   = Ent.EntanglementRobustness(rho_full,  method="PPT_mixer")
                t_gm_ppt   = Ent.EntanglementRobustness(rho_full,  method="PPT")

                # A | BC   (top qubit vs. qubit+qutrit)
                rhoA_BC    = MultiState(M, [2, 6])
                t_a_mix  = Ent.EntanglementRobustness(rhoA_BC,   method="PPT_mixer")
                t_a_ppt  = Ent.EntanglementRobustness(rhoA_BC,   method="PPT")

                # AB | C   (qubit+qubit vs. qutrit)
                rhoAB_C    = MultiState(M, [4, 3])
                t_c_mix = Ent.EntanglementRobustness(rhoAB_C,   method="PPT_mixer")
                t_c_ppt = Ent.EntanglementRobustness(rhoAB_C,   method="PPT")

                # AC | B   (qubit+qutrit vs. middle qubit) via PermuteSystems(2,3)
                rho_perm   = PermuteSystems(rho_full, 2, 3)
                rhoAC_B    = MultiState(rho_perm.mat, [6, 2])
                t_b_mix  = Ent.EntanglementRobustness(rhoAC_B,   method="PPT_mixer")
                t_b_ppt  = Ent.EntanglementRobustness(rhoAC_B,   method="PPT")

                # Hofmann-Moroder-Guhne 2014 genuine multipartite negativity:
                # the convex-roof-of-bipartite-negativity GME measure
                # (arXiv:1401.2424). Defined for the full tripartite state.
                gmn_hmg  = Ent.EntanglementRobustness(rho_full,   method="HMG")

                (
                    gm_mix   = Float64(t_gm_mix),    gm_ppt   = Float64(t_gm_ppt),
                    a_mix  = Float64(t_a_mix),   a_ppt  = Float64(t_a_ppt),
                    c_mix = Float64(t_c_mix),  c_ppt = Float64(t_c_ppt),
                    b_mix  = Float64(t_b_mix),   b_ppt  = Float64(t_b_ppt),
                    gmn_hmg = Float64(gmn_hmg),
                )
            end
        end
    """)

    _JL = jl
    return _JL


# -------------------------------------------------------------------
# Public API
# -------------------------------------------------------------------

def ppt_visibilities(rho, qc_dir=DEFAULT_QC_DIR):
    """
    Compute the PPT and PPT-mixture visibilities of a 12x12 (qubit, qubit,
    qutrit) density matrix in the four standard partitions.

    Parameters
    ----------
    rho : (12, 12) array_like, complex
        Density matrix with subsystem ordering [2, 2, 3] (e.g. (t, tbar, Z)).
    qc_dir : str, optional
        Path to the `quantum-correlations` package. Defaults to a sibling
        directory of this file.

    Returns
    -------
    dict
        Keys
          'PPT_GM',     'PPT_mix_GM'    : tripartite (genuine) values
          'PPT_A',    'PPT_mix_A'   : A|BC  (dims [2, 6])
          'PPT_C',   'PPT_mix_C'  : AB|C  (dims [4, 3])
          'PPT_B',    'PPT_mix_B'   : AC|B  (dims [6, 2])
    """
    jl = _init_julia(qc_dir=qc_dir)
    M = np.ascontiguousarray(rho, dtype=np.complex128)
    if M.shape != (12, 12):
        raise ValueError(f"rho must be 12x12, got shape {M.shape}")
    nt = jl._ppt_visibilities(M)
    return {
        "PPT_GM":       float(nt.gm_ppt),
        "PPT_mix_GM":   float(nt.gm_mix),
        "PPT_A":      float(nt.a_ppt),
        "PPT_mix_A":  float(nt.a_mix),
        "PPT_C":     float(nt.c_ppt),
        "PPT_mix_C": float(nt.c_mix),
        "PPT_B":      float(nt.b_ppt),
        "PPT_mix_B":  float(nt.b_mix),
        # Hofmann-Moroder-Guhne 2014 GMN, the convex-roof-of-bipartite-
        # negativity GME measure (arXiv:1401.2424).
        "GMN_HMG":    float(nt.gmn_hmg),
    }


# Convenient list of the keys ppt_visibilities() returns, in a stable order.
PPT_KEYS = [
    "PPT_GM",     "PPT_mix_GM",
    "PPT_A",    "PPT_mix_A",
    "PPT_C",   "PPT_mix_C",
    "PPT_B",    "PPT_mix_B",
    "GMN_HMG",
]


def gmn_hmg(rho, qc_dir=DEFAULT_QC_DIR):
    """
    Compute the Hofmann-Moroder-Guhne 2014 genuine multipartite negativity
    (arXiv:1401.2424) of a 12x12 (qubit, qubit, qutrit) density matrix.

    This calls only `EntanglementRobustness(rho, method="HMG")` on the full
    tripartite state, so it is roughly 8x cheaper than `ppt_visibilities`
    when only the HMG measure is needed.

    Parameters
    ----------
    rho : (12, 12) array_like, complex
        Density matrix with subsystem ordering [2, 2, 3].
    qc_dir : str, optional
        Path to the `quantum-correlations` package.

    Returns
    -------
    float
        E_HMG(rho) >= 0; 0 iff rho is a PPT mixture.
    """
    jl = _init_julia(qc_dir=qc_dir)
    M = np.ascontiguousarray(rho, dtype=np.complex128)
    if M.shape != (12, 12):
        raise ValueError(f"rho must be 12x12, got shape {M.shape}")
    return float(jl._gmn_hmg(M))
