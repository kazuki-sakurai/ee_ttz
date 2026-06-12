import numpy as np
from cp_check import cp_map
d = np.load('data/psinteg_all_beam_res.npz')
R1 = d['R1']; R2 = d['R2']
for i in range(len(R1)):
    R1_CP = cp_map(R1[i])
    rel = np.max(np.abs(R1_CP - R2[i])) / np.max(np.abs(R1[i]))
    print(f"{i:2d}  rs={d['rs'][i]:.0f}  |cp_map(R1) - R2|/|R1| = {rel:.3e}")
    