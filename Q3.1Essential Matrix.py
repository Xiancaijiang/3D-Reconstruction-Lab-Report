def essentialMatrix(F, K1, K2):
    E = K2.T @ F @ K1
    U, S, V = np.linalg.svd(E)
    S = [1, 1, 0]  # Force singular values
    E = U @ np.diag(S) @ V
    return E