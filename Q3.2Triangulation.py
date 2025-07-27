def triangulate(C1, pts1, C2, pts2):
    num_points = pts1.shape[0]
    P = np.zeros((num_points, 3))
    err = 0
    
    for i in range(num_points):
        # Construct A matrix
        A = np.zeros((4, 4))
        A[0] = pts1[i, 0] * C1[2] - C1[0]
        A[1] = pts1[i, 1] * C1[2] - C1[1]
        A[2] = pts2[i, 0] * C2[2] - C2[0]
        A[3] = pts2[i, 1] * C2[2] - C2[1]
        
        # Solve for P
        _, _, V = np.linalg.svd(A)
        P_hom = V[-1]
        P[i] = P_hom[:3] / P_hom[3]
        
        # Compute reprojection error
        p1_proj = C1 @ np.append(P[i], 1)
        p1_proj = p1_proj[:2] / p1_proj[2]
        p2_proj = C2 @ np.append(P[i], 1)
        p2_proj = p2_proj[:2] / p2_proj[2]
        
        err += np.sum((pts1[i] - p1_proj)**2 + (pts2[i] - p2_proj)**2)
    
    return P, err