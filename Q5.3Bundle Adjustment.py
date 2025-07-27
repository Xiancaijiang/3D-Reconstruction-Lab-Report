def rodriguesResidual(K1, M1, p1, K2, p2, x):
    N = len(p1)
    P = x[:-6].reshape(N, 3)
    r = x[-6:-3]
    t = x[-3:]
    
    R = rodrigues(r)
    M2 = np.hstack([R, t.reshape(3, 1)])
    
    C1 = K1 @ M1
    C2 = K2 @ M2
    
    residuals = []
    for i in range(N):
        # Project to camera 1
        p1_proj = C1 @ np.append(P[i], 1)
        p1_proj = p1_proj[:2] / p1_proj[2]
        
        # Project to camera 2
        p2_proj = C2 @ np.append(P[i], 1)
        p2_proj = p2_proj[:2] / p2_proj[2]
        
        residuals.extend(p1[i] - p1_proj)
        residuals.extend(p2[i] - p2_proj)
    
    return np.array(residuals)

def bundleAdjustment(K1, M1, p1, K2, M2_init, p2, P_init):
    # Extract rotation and translation
    R = M2_init[:, :3]
    t = M2_init[:, 3]
    r = invRodrigues(R)
    
    # Initial parameters
    x0 = np.concatenate([P_init.ravel(), r, t])
    
    # Optimize
    from scipy.optimize import least_squares
    res = least_squares(lambda x: rodriguesResidual(K1, M1, p1, K2, p2, x), x0)
    
    # Extract results
    x_opt = res.x
    N = len(p1)
    P_opt = x_opt[:-6].reshape(N, 3)
    r_opt = x_opt[-6:-3]
    t_opt = x_opt[-3:]
    R_opt = rodrigues(r_opt)
    M2_opt = np.hstack([R_opt, t_opt.reshape(3, 1)])
    
    return M2_opt, P_opt