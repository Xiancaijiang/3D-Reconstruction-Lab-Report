import numpy as np
import submission as sub
from helper import camera2

# Load data
data = np.load('../data/some_corresp.npz')
pts1 = data['pts1']
pts2 = data['pts2']
K = np.load('../data/intrinsics.npz')
K1 = K['K1']
K2 = K['K2']
M = max(640, 480)

# Compute F and E
F = sub.eightpoint(pts1, pts2, M)
E = sub.essentialMatrix(F, K1, K2)

# Get possible M2 matrices
M2s = camera2(E)
M1 = np.hstack([np.eye(3), np.zeros((3, 1))])
C1 = K1 @ M1

best_M2 = None
best_P = None
best_err = float('inf')

for i in range(4):
    M2 = M2s[:, :, i]
    C2 = K2 @ M2
    P, err = sub.triangulate(C1, pts1, C2, pts2)
    
    # Check if points are in front of both cameras
    if np.all(P[:, 2] > 0):
        if err < best_err:
            best_err = err
            best_M2 = M2
            best_P = P

# Save results
np.savez('q3_3.npz', M2=best_M2, C2=K2 @ best_M2, P=best_P)