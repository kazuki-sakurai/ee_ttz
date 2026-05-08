import sys
import numpy as np
from math import pi, sqrt, acos

def csqrt(x):
    """sqrt that returns complex values for negative real input."""
    return np.sqrt(np.asarray(x, dtype=complex))

def get_F3(C1, C2, C3):
    Q = (C1+C2+C3)/2
    return sqrt( (16/3)*Q*(Q-C1)*(Q-C2)*(Q-C3) )

def concurrence(rho):
    # Pauli-Y and the spin-flip operator Y ⊗ Y
    sigma_y = np.array([[0, -1j],
                        [1j,  0]])
    YY = np.kron(sigma_y, sigma_y)

    # Spin-flipped density matrix: rho_tilde = (Y⊗Y) rho* (Y⊗Y)
    rho_tilde = YY @ np.conjugate(rho) @ YY

    # Eigenvalues of rho * rho_tilde (these equal the squares of the
    # eigenvalues of sqrt(sqrt(rho) rho_tilde sqrt(rho)))
    eigvals = np.linalg.eigvals(rho @ rho_tilde)

    # Numerical noise can give tiny negative real parts; clip to >= 0
    eigvals = np.sort(np.real(eigvals))[::-1]
    eigvals = np.clip(eigvals, 0, None)
    sqrt_eigs = np.sqrt(eigvals)

    return max(0.0, sqrt_eigs[0] - sqrt_eigs[1] - sqrt_eigs[2] - sqrt_eigs[3])


def get_trace(rho):
    tr = np.trace(rho)
    return np.real_if_close(tr)

def normalise(rho):
    tr = np.real_if_close(np.trace(rho))
    return rho/tr

def purity(rho):
    pure = np.trace(rho@rho)
    return np.real_if_close(pure)

def log_neg_bip(rho0, dim):
    d1, d2 = dim
    rho = rho0.reshape(d1,d2,d1,d2)
    rho_PT = rho.transpose(2, 1, 0, 3)
    rho_PT = rho_PT.reshape(d1*d2,d1*d2)
    eigvals = np.linalg.eigvalsh(rho_PT)
    neg = np.sum(np.abs(eigvals[eigvals < 0]))
    EN = np.log2(1 + 2*neg)
    return EN

def partial_transpose_ABC(rho0, target):
    rho = rho0.reshape(2,2,3,2,2,3)
    if target == 'A':
        rho_PT = rho.transpose(3, 1, 2, 0, 4, 5)
    if target == 'B':
        rho_PT = rho.transpose(0, 4, 2, 3, 1, 5)
    if target == 'C':
        rho_PT = rho.transpose(0, 1, 5, 3, 4, 2)
    return rho_PT.reshape(12,12)

def negativity(rho, target):
    rho_PT = partial_transpose_ABC(rho, target)
    eigvals = np.linalg.eigvalsh(rho_PT)
    return np.sum(np.abs(eigvals[eigvals < 0]))

def log_negativity(rho, target):
    neg = negativity(rho, target)
    return np.log2(1 + 2*neg)

