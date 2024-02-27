import numpy as np
from sklearn.decomposition import PCA

def NFINDR(HIM, p):
    """
    NFINDR algorithm for endmember extraction.
    
    Parameters:
    HIM : Hyperspectral image cube (nrows x ncols x nbands).
    p : Number of endmembers to extract.
    
    Returns:
    U : Extracted endmembers (nbands x p).
    P : Spatial coordinates of extracted endmembers.
    """
    # PCA for feature reduction
    nrows, ncols, nbands = HIM.shape
    HIM_reshaped = HIM.reshape((nrows*ncols, nbands))
    pca = PCA(n_components=p-1)
    pca.fit(HIM_reshaped)
    scores = pca.transform(HIM_reshaped)
    MNFHIM = scores.reshape((nrows, ncols, p-1))
    
    # Initial random pixels
    np.random.seed(0)
    P = np.zeros((2, p))
    for i in range(p):
        row = np.random.randint(0, nrows)
        col = np.random.randint(0, ncols)
        P[:, i] = [row, col]
    
    # Iterative optimization
    max_iter = 3 * p
    volume = 0
    for _ in range(max_iter):
        for i in range(p):
            for row in range(nrows):
                for col in range(ncols):
                    test_matrix = MNFHIM[row, col, :].reshape(1, p-1)
                    if i > 0:
                        test_matrix = np.vstack([P[:, :i], test_matrix])
                    new_volume = np.abs(np.linalg.det(test_matrix))
                    if new_volume > volume:
                        volume = new_volume
                        P[:, i] = [row, col]
    
    # Extracting endmembers
    U = np.zeros((nbands, p))
    for i in range(p):
        U[:, i] = HIM[int(P[0, i]), int(P[1, i]), :]
    
    return U, P.T

# Example usage
# HIM = np.random.rand(100, 100, 10)  # Hyperspectral image cube example
# p = 5  # Number of endmembers to extract
# U, P = NFINDR(HIM, p)
