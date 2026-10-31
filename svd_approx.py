import numpy as np

def validate_rank_and_dimensions(matrix, r) :

    """ Validates matrix dimensions and the requested rank r. Raises ValueError or TypeError if inputs are invalid. """

    if not isinstance(matrix, np.ndarray):
        matrix = np.array(matrix)
        
    if matrix.ndim != 2:
        raise ValueError("Input matrix must be 2-dimensional.")
        
    m, n = matrix.shape
    max_rank = min(m, n)
    
    if not isinstance(r, int) or r <= 0:
        raise ValueError(f"Rank r must be a positive integer. Got: {r}")
        
    if r > max_rank:
        raise ValueError( f"The requested rank r ({r}) cannot exceed the minimum matrix dimension min({m}, {n}) = {max_rank}.")
    

def compute_svd(matrix):

    """Computes the Singular Value Decomposition (SVD) of a given numerical matrix using a numpy."""

    U, S, Vt = np.linalg.svd(matrix, full_matrices=False)
    return U, S, Vt


def truncate_svd(U, S, Vt, r):

    """Constructs the rank-r truncation using the leading r singular values and corresponding singular vectors."""

    U_r = U[:, :r]
    S_r = S[:r]
    Vt_r = Vt[:r, :]
    return U_r, S_r, Vt_r


def low_rank_approximation(matrix, r):
    """ Validates inputs, computes the SVD, truncates to rank r, and reconstructs the low-rank matrix approximation."""

    # Perform input validation
    validate_rank_and_dimensions(matrix, r)
    
    # Compute full SVD
    U, S, Vt = compute_svd(matrix)
    
    # Truncate to rank r
    U_r, S_r, Vt_r = truncate_svd(U, S, Vt, r)
    
    # Reconstruct the low-rank matrix: A_approx = U_r * diag(S_r) * Vt_r
    matrix_approx = (U_r * S_r) @ Vt_r
    
    return matrix_approx