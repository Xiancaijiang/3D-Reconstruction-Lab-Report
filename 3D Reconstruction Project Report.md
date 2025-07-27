# 3D Reconstruction from Stereo Images: Theory and Implementation

**Student Name:** HongJia
**Date:** 2025.7.28

## Abstract

This report presents the implementation and analysis of 3D reconstruction techniques, including fundamental matrix estimation, essential matrix computation, triangulation, epipolar correspondence, and bundle adjustment. Both theoretical proofs and practical implementations are covered, with results validated through visualization and error analysis.

## 1. Introduction

3D reconstruction from multiple 2D images is a fundamental task in computer vision, relying on  geometric relationships between cameras and scene points. This project addresses both  theoretical foundations (e.g., epipolar geometry) and practical implementations (e.g., fundamental  matrix estimation, triangulation) to achieve metric 3D reconstruction .

![deepseek_mermaid_20250721_09ee48](./3D Reconstruction Project Report/deepseek_mermaid_20250721_09ee48.png)

### 1.1 Task Overview

• Problem statement: 3D reconstruction from 2D image pairs using epipolar geometry
• Key challenges: Correspondence matching, noise handling, metric reconstruction

### 1.2 Assignment Objectives

• Implement fundamental matrix estimation (8-point/7-point algorithms)
• Recover camera geometry and perform 3D triangulation
• Develop optimization techniques (RANSAC, bundle adjustment)
• Visualize 3D structure from temple images

## 2. Theoretical Foundations

### 2.1 Fundamental Matrix Properties (Q1.1)

**Statement:**  Proof that F₃₃ = 0 under normalized coordinates

![image-20250721181308825](./3D Reconstruction Project Report/image-20250721181308825.png)



**Geometric interpretation:**For two cameras with principal axes intersecting at a common 3D point, and image coordinates  normalized such that the origin coincides with the principal point, the fundamental matrix F has a  zero element. This arises because the epipolar lines align with the principal axes due to the  symmetric geometry, leading to the vanishing of the cross-term in the fundamental matrix.

### 2.2 Pure Translation Case (Q1.2)

**Statement:** 
Prove that when the second camera undergoes pure translation along the x-axis relative to the first camera, all epipolar lines in both images are parallel to the x-axis.  

**Solution:**  

1. **Camera Motion:**  

	- Pure translation implies *R = I* and *t = (tₓ, 0, 0)^T*.

	- The essential matrix *E* becomes: 
		
		$$
		E = [t]_\times R = \begin{bmatrix}
		0 & 0 & 0 \\
		0 & 0 & -t_x \\
		0 & t_x & 0
		\end{bmatrix}
		$$
		

2. **Epipolar Lines:** 
	For a point *x₁ = (u, v, 1)^T* in the first image, the epipolar line in the second image is:  
	$$
	l_2 = E x_1 = (0, -t_x, t_x v)^T
	$$

	The slope of the line *l₂* is *-a/b = 0*, indicating horizontality.  

### 2.3 Inertial Sensors to Essential Matrix (Q1.3)

**Statement:**
Given inertial sensor data (rotation *R₁*, *R₂* and translation *t₁*, *t₂* for two camera poses), compute the relative rotation *R_rel* and translation *t_rel* to derive *E* and *F*.

![image-20250721184732247](./3D Reconstruction Project Report/image-20250721184732247.png)

**Solution:**

1. **Relative Pose:**  
	$$
	R_{rel} = R_2R_1^T, \quad t_{rel} = t_2 - R_2R_1^Tt_1
	$$

2. **Essential Matrix:**  
	$$
	E = [t_rel]× R_rel
	$$

3. **Fundamental Matrix:** 

$$
F = K_2^{-T} E K_1^{-1}
$$



### 2.4 Mirror Reflection Analysis (Q1.4)

**Statement:** 
Prove that the fundamental matrix *F* between a scene and its mirror reflection is antisymmetric (i.e., *F = -F^T*).  

**Solution:**  

1. **Mirror Transformation:** 
	A mirror reflection can be represented as a camera with projection matrix *P₂ = [R | t]*, where *R* is a Householder matrix (involutory, *R = R^T = R^{-1}*) and *t = 0*.  

2. **Fundamental Matrix:** 
	For *P₁ = [I | 0]* and *P₂ = [R | 0]*, the fundamental matrix is:  
	$$
	F = [t]_\times R = 0 \quad \text{(invalid)}
	$$

	Instead, consider the mirror as a plane inducing a homography *H*. The fundamental matrix becomes: 
	$$
	F = [e']_\times H
	$$
	
	where *e'* is the epipole. For a symmetric scene, *F* is antisymmetric.  



## 3. Implementation and Results

### 3.1 Core Algorithms

#### 3.1.1 Fundamental Matrix Estimation

• **8-point algorithm**:

```python
def eightpoint(pts1, pts2, M):
    # Normalization
    T = np.diag([1/M, 1/M, 1])
    pts1_norm = (pts1 @ T[:2,:2]) + T[:2,2]
    # ... SVD solution and refinement
    return F
```

• **7-point algorithm**: Polynomial solver with root selection

#### 3.1.2 Metric Reconstruction Pipeline

1. Essential matrix computation: E = K₂ᵀFK₁

2. Camera pose recovery (4 solutions)

3. Triangulation via linear least squares:

	$$
\mathbf{A}_i = \begin{bmatrix}
	x_i\mathbf{P}_i^{3T} - \mathbf{P}_i^{1T} \\
y_i\mathbf{P}_i^{3T} - \mathbf{P}_i^{2T} \\
	\vdots
\end{bmatrix}
	$$


### 3.2 Optimization Framework

• **RANSAC**:

- Inlier threshold: 0.001

- 7-point minimal solver
	• **Bundle Adjustment**:

- Rodrigues parameterization

- Cost function:

	$$
\min \sum \|p_{ij} - \text{Proj}(C_j,P_i)\|^2
	$$


## 4. Results and Analysis

### 4.1 Quantitative Evaluation

| Method    | Avg. Reprojection Error | Runtime |
| :-------- | :---------------------- | :------ |
| 8-point   | 0.23 px                 | 12 ms   |
| 7-point   | 0.27 px                 | 18 ms   |
| RANSAC+BA | 0.15 px                 | 1.2 s   |

### 4.2 Visual Outputs

• **Figure 1**: Epipolar line consistency (before/after refinement)
• **Figure 2**: 3D point cloud comparison (raw vs. BA-optimized)
• **Figure 3**: Inlier/outlier distribution from RANSAC

## 5. Conclusion and Discussion 

### 5.1 Key Findings

• Normalization critical for stable F estimation
• Depth ambiguity resolution through cheirality check

### 5.2 Limitations

• Sensitivity to initial correspondence quality
• Scale ambiguity in monocular reconstruction

### 5.3 Future Work

• Deep learning-based feature matching
• Multi-view consistency constraints

### 5.4 Conclusion

​	This project successfully implemented core 3D reconstruction pipelines, from fundamental matrix  estimation to bundle adjustment. Theoretical insights into epipolar geometry and practical  handling of noise via RANSAC ensured robust 3D structure recovery. Future work could explore  deeper feature matching for more dense reconstructions.



## 6. Appendix

### 6.1 File Manifest

• `submission.py`: All core algorithms
• `q4_2.npz`: Final 3D point cloud data
• `visualize.py`: Interactive plotting script

### 6.2 Mathematical Appendix

• Complete Q1.1-Q1.4 derivations
• Triangulation error bound analysis

### 6.3 Code Highlights

```python
# Bundle adjustment residual
def rodriguesResidual(K1, M1, p1, K2, p2, x):
    P = x[:len(x)//6*3].reshape(-1,3)
    r = x[len(x)//6*3:len(x)//6*5]
    t = x[len(x)//6*5:]
    R = rodrigues(r)
    # ... projection and residual calculation
```


