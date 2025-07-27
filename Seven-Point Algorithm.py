def sevenpoint(pts1, pts2, M):
    # Normalize points
    pts1 = pts1 / M
    pts2 = pts2 / M
    
    # Construct A matrix
    A = np.zeros((7, 9))
    for i in range(7):
        x1, y1 = pts1[i]
        x2, y2 = pts2[i]
        A[i] = [x2*x1, x2*y1, x2, y2*x1, y2*y1, y2, x1, y1, 1]
    
    # Solve for F using SVD
    U, S, V = np.linalg.svd(A)
    F1 = V[-1].reshape(3, 3)
    F2 = V[-2].reshape(3, 3)
    
    # Solve polynomial equation
    def f(alpha):
        return np.linalg.det(alpha * F1 + (1 - alpha) * F2)
    
    # Find roots
    coeffs = np.zeros(4)
    coeffs[0] = f(0)
    coeffs[1] = (f(1) - f(-1))/3 - (f(2) - f(-2))/12
    coeffs[2] = (f(1) + f(-1))/2 - f(0)
    coeffs[3] = (f(1) - f(-1))/2 - coeffs[1]
    roots = np.roots(coeffs)
    
    # Generate possible F matrices
    Farray = []
    T = np.diag([1/M, 1/M, 1])
    for alpha in roots:
        if np.isreal(alpha):
            F = np.real(alpha) * F1 + (1 - np.real(alpha)) * F2
            U, S, V = np.linalg.svd(F)
            S[2] = 0
            F = U @ np.diag(S) @ V
            F = T.T @ F @ T
            Farray.append(F)
    
    return Farray