def rodrigues(r):
    theta = np.linalg.norm(r)
    if theta < 1e-6:
        return np.eye(3)
    
    k = r / theta
    K = np.array([[0, -k[2], k[1]],
                  [k[2], 0, -k[0]],
                  [-k[1], k[0], 0]])
    
    R = np.eye(3) + np.sin(theta) * K + (1 - np.cos(theta)) * K @ K
    return R

def invRodrigues(R):
    A = (R - R.T) / 2
    rho = np.array([A[2, 1], A[0, 2], A[1, 0]])
    s = np.linalg.norm(rho)
    c = (np.trace(R) - 1) / 2
    
    if s == 0 and c == 1:
        return np.zeros(3)
    elif s == 0 and c == -1:
        # Special case
        pass  # Implementation omitted for brevity
    else:
        theta = np.arctan2(s, c)
        return rho / s * theta