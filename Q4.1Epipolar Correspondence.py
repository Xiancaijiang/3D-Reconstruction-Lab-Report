def epipolarCorrespondence(im1, im2, F, x1, y1):
    window_size = 10
    search_range = 50
    
    # Get epipolar line
    line = F @ np.array([x1, y1, 1])
    
    # Convert to slope-intercept form (y = mx + b)
    if abs(line[0]) > abs(line[1]):
        # More horizontal line
        y_start = max(0, int(y1) - search_range)
        y_end = min(im2.shape[0], int(y1) + search_range)
        best_dist = float('inf')
        best_x2, best_y2 = 0, 0
        
        for y2 in range(y_start, y_end):
            x2 = int(-(line[1]*y2 + line[2]) / line[0])
            
            # Compute window similarity
            dist = 0
            for i in range(-window_size, window_size):
                for j in range(-window_size, window_size):
                    if (0 <= y1+i < im1.shape[0] and 0 <= x1+j < im1.shape[1] and
                        0 <= y2+i < im2.shape[0] and 0 <= x2+j < im2.shape[1]):
                        dist += np.sum((im1[int(y1)+i, int(x1)+j] - im2[y2+i, x2+j])**2)
            
            if dist < best_dist:
                best_dist = dist
                best_x2, best_y2 = x2, y2
                
        return best_x2, best_y2
    else:
        # More vertical line
        x_start = max(0, int(x1) - search_range)
        x_end = min(im2.shape[1], int(x1) + search_range)
        best_dist = float('inf')
        best_x2, best_y2 = 0, 0
        
        for x2 in range(x_start, x_end):
            y2 = int(-(line[0]*x2 + line[2]) / line[1])
            
            # Compute window similarity
            dist = 0
            for i in range(-window_size, window_size):
                for j in range(-window_size, window_size):
                    if (0 <= y1+i < im1.shape[0] and 0 <= x1+j < im1.shape[1] and
                        0 <= y2+i < im2.shape[0] and 0 <= x2+j < im2.shape[1]):
                        dist += np.sum((im1[int(y1)+i, int(x1)+j] - im2[y2+i, x2+j])**2)
            
            if dist < best_dist:
                best_dist = dist
                best_x2, best_y2 = x2, y2
                
        return best_x2, best_y2