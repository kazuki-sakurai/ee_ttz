import sys, numpy as np
sys.path.append('/Users/kazuki/Projects/pyHELAS')
from QI_functions import log_negativity, normalise
from ppt_julia import gmn_hmg

np.random.seed(0)
A   = np.random.randn(12, 12) + 1j * np.random.randn(12, 12)
rho = normalise(A @ A.conj().T)

EN1 = log_negativity(rho, 'A')
EN2 = log_negativity(rho, 'B')
EN3 = log_negativity(rho, 'C')
N1  = (2**EN1 - 1)/2
N2  = (2**EN2 - 1)/2
N3  = (2**EN3 - 1)/2
minN = min(N1, N2, N3)
g    = gmn_hmg(rho)

print(f"  N1 (t   | tbar Z)  = {N1:.6f}")
print(f"  N2 (tbar | t  Z)   = {N2:.6f}")
print(f"  N3 (Z   | t  tbar) = {N3:.6f}")
print(f"  min(N1, N2, N3)    = {minN:.6f}")
print(f"  GMN_HMG (SDP)      = {g:.6f}")
print()
if abs(g - N1) < 1e-4:
    print("  !! GMN_HMG == N1 to 1e-4 -- the bug is STILL LIVE")
    print("     (Julia compile cache or the wrong .jl file?)")
elif g <= minN + 1e-6:
    print("  ✓ GMN_HMG <= min(N_i) -- convex-roof bound holds")
    if minN - g > 1e-4:
        print(f"  ✓ AND strictly less by {minN - g:.4f}: convex roof is")
        print(f"    binding (mixed-state genuine multipartite entanglement)")
else:
    print(f"  ?? GMN > min(N) by {g - minN:.3e} -- unexpected, investigate")
