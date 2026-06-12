import numpy as np
d = np.load('data/psinteg_all_beam_res.npz')
R1 = d['R1']; R2 = d['R2']
for i in range(len(R1)):
    e1 = np.sort(np.linalg.eigvalsh(R1[i]))
    e2 = np.sort(np.linalg.eigvalsh(R2[i]))
    rel = np.max(np.abs(e1 - e2)) / np.max(np.abs(e1))
    print(f"{i:2d}  rs={d['rs'][i]:.0f}  |eig(R1)-eig(R2)|/|eig| = {rel:.3e}")
    