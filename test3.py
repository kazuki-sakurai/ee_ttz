import sys, os, numpy as np
sys.path.append('/Users/kazuki/Projects/pyHELAS')
import ppt_julia
from ppt_julia import gmn_hmg

print(f"ppt_julia loaded from : {ppt_julia.__file__}")
print(f"DEFAULT_QC_DIR        : {ppt_julia.DEFAULT_QC_DIR}")
print(f"Patched file path     : {os.path.join(ppt_julia.DEFAULT_QC_DIR, 'src', 'entanglement', 'RobustnessToPPTMixture.jl')}")

# Show the actual content of the constraint line that should be patched.
jl = os.path.join(ppt_julia.DEFAULT_QC_DIR, 'src', 'entanglement', 'RobustnessToPPTMixture.jl')
print(f"File exists           : {os.path.isfile(jl)}")
if os.path.isfile(jl):
    with open(jl) as f:
        lines = f.readlines()
    print()
    print("Lines 105-112 (HMG normalisation block):")
    for i in range(104, min(112, len(lines))):
        print(f"  {i+1}: {lines[i].rstrip()}")

# Take a known-violating point from the saved scan_results and re-evaluate.
d = np.load('scan_results_0_1.npz')
m = d['pol'] == 3
idx_all = np.arange(len(d['pol']))[m]
N1 = (2**d['EN1'][m] - 1)/2
N2 = (2**d['EN2'][m] - 1)/2
N3 = (2**d['EN3'][m] - 1)/2
minN = np.minimum.reduce([N1, N2, N3])
G    = d['GMN_HMG'][m]
viol_mask = G > minN + 1e-6
print()
print(f"\n{viol_mask.sum()} violating points (pol=3) in scan_results_0_1.npz")
if viol_mask.any():
    # Pick one and recompute live.
    j = int(np.where(viol_mask)[0][0])
    saved_G   = float(G[j])
    minN_here = float(minN[j])
    rho = d['rho'][idx_all[j]]
    print(f"\nSpot-check point (pol=3, row {idx_all[j]}):")
    print(f"  saved GMN_HMG  : {saved_G:.6f}")
    print(f"  min(N1,N2,N3)  : {minN_here:.6f}")
    print(f"  N1, N2, N3     : {N1[j]:.6f}, {N2[j]:.6f}, {N3[j]:.6f}")
    live_G = gmn_hmg(rho)
    print(f"  live gmn_hmg() : {live_G:.6f}")
    print()
    if abs(live_G - saved_G) < 1e-3:
        print("  -> live SDP REPRODUCES the saved (buggy) value.  This means")
        print("     the Julia code on disk that the rebuild used is the SAME")
        print("     the current process uses -- and BOTH are still buggy.")
        print("     Patch did not actually take effect.")
    elif live_G <= minN_here + 1e-6 and abs(live_G - saved_G) > 1e-3:
        print("  -> live SDP gives a DIFFERENT (correct) value than the saved")
        print("     one.  The rebuild was using stale code/cache; re-run after")
        print("     clearing the Julia compile cache.")
    else:
        print("  -> ?? live SDP gives a value I don't recognise; investigate.")
