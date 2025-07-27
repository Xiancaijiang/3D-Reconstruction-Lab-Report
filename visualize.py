import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import submission as sub

# Load data
data = np.load('../data/some_corresp.npz')
pts1 = data['pts1']
pts2 = data['pts2']
K = np.load('../data/intrinsics.npz')
K1 = K['K1']
K2 = K['K2']
temple_coords = np.load('../data/templeCoords.npz')
x1 = temple_coords['x1']
y1 = temple_coords['y1']
M = max(640, 480)

# Compute F and E
F = sub.eightpoint(pts1, pts2, M)
E = sub.essentialMatrix(F, K1, K2)

# Get correct M2
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
    
    if np.all(P[:, 2] > 0):
        if err < best_err:
            best_err = err
            best_M2 = M2
            best_P = P

# Triangulate temple points
temple_pts1 = np.vstack([x1, y1]).T
temple_pts2 = np.zeros_like(temple_pts1)

for i in range(len(temple_pts1)):
    x2, y2 = sub.epipolarCorrespondence(im1, im2, F, temple_pts1[i, 0], temple_pts1[i, 1])
    temple_pts2[i] = [x2, y2]

temple_P, _ = sub.triangulate(C1, temple_pts1, C2, temple_pts2)

# Visualize
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(temple_P[:, 0], temple_P[:, 1], temple_P[:, 2], c='b', marker='o')
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
plt.show()

# Save results
np.savez('q4_2.npz', F=F, M1=M1, M2=best_M2, C1=C1, C2=C2 @ best_M2)