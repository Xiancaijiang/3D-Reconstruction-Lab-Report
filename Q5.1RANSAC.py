def ransacF(pts1, pts2, M):
    max_iter = 1000
    threshold = 0.01
    best_inliers = None
    best_F = None
    
    for _ in range(max_iter):
        # Randomly select 7 points
        indices = np.random.choice(len(pts1), 7, replace=False)
        sample1 = pts1[indices]
        sample2 = pts2[indices]
        
        # Compute F using seven-point algorithm
        Farray = sevenpoint(sample1, sample2, M)
        
        for F in Farray:
            # Compute inliers
            inliers = np.zeros(len(pts1), dtype=bool)
            for i in range(len(pts1)):
                x1 = np.append(pts1[i], 1)
                x2 = np.append(pts2[i], 1)
                error = abs(x2.T @ F @ x1)
                if error < threshold:
                    inliers[i] = True
            
            # Update best F if better
            if np.sum(inliers) > np.sum(best_inliers if best_inliers is not None else 0):
                best_inliers = inliers
                best_F = F
    
    # Refine F using all inliers
    best_F = eightpoint(pts1[best_inliers], pts2[best_inliers], M)
    
    return best_F, best_inliers