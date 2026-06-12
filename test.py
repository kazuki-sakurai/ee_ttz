import sys, numpy as np
sys.path.append('/Users/kazuki/Projects/pyHELAS')
from QI_functions import log_negativity, normalise
from ppt_cvxpy import get_GMN
from ppt_julia  import gmn_hmg

def check(label, rho):
    N = [(2**log_negativity(rho, x) - 1) / 2 for x in 'ABC']
    j = gmn_hmg(rho)
    c = get_GMN(rho, dims=[2, 2, 3])
    print(f"  {label:<30s}  min(N)={min(N):.4f}  "
          f"Julia/Mosek={j:.4f}  CVXPY={c:.4f}  "
          f"|diff|={abs(j-c):.3e}")

np.random.seed(0)
A = np.random.randn(12, 12) + 1j * np.random.randn(12, 12)
check("Wishart random",   normalise(A @ A.conj().T))

psi = np.zeros(12, dtype=np.complex128); psi[0]=1/np.sqrt(2); psi[10]=1/np.sqrt(2)
check("GHZ |000>+|111>",  np.outer(psi, psi.conj()))

psi = np.zeros(12, dtype=np.complex128); psi[0] = 1.0
check("product |000>",    np.outer(psi, psi.conj()))
