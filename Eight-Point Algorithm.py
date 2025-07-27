def eightpoint(pts1, pts2, M):
    # Normalize points
    pts1 = pts1 / M
    pts2 = pts2 / M
    
    # Construct A matrix
    A = np.zeros((len(pts1), 9))
    for i in range(len(pts1)):
        x1, y1 = pts1[i]
        x2, y2 = pts2[i]
        A[i] = [x2*x1, x2*y1, x2, y2*x1, y2*y1, y2, x1, y1, 1]
    
    # Solve for F using SVD
    U, S, V = np.linalg.svd(A)
    F = V[-1].reshape(3, 3)
    
    # Enforce rank-2 constraint
    U, S, V = np.linalg.svd(F)
    S[2] = 0
    F = U @ np.diag(S) @ V
    
    # Refine F
    F = refineF(F, pts1, pts2)
    
    # Unnormalize F
    T = np.diag([1/M, 1/M, 1])
    F = T.T @ F @ T
    
    return F