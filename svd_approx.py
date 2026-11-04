
import numpy as np

class SVD:

    def __init__(self, matrix, r):

        """Initializes the matrix and given rank."""

        self.matrix = np.asarray(matrix)
        self.r = r

        self.validate_rank_and_dimensions()

    def validate_rank_and_dimensions(self):

        """Validates matrix dimensions, numerical values, and rank."""

        if self.matrix.ndim != 2:
            raise ValueError("Input matrix must be 2-dimensional.")

        if not np.issubdtype(self.matrix.dtype, np.number):
            raise TypeError("Matrix must contain numerical values.")

        if not np.all(np.isfinite(self.matrix)):
            raise ValueError("Matrix must contain only finite values.")

        m, n = self.matrix.shape
        max_rank = min(m, n)

        if not isinstance(self.r, (int, np.integer)) or isinstance(self.r, bool) or self.r <= 0:
            raise ValueError(f"Rank r must be a positive integer. Got: {self.r}")

        if self.r > max_rank:
            raise ValueError(
                f"The provided rank r ({self.r}) cannot exceed "
                f"min({m}, {n}) = {max_rank}."
            )

    def compute_svd(self):
    
        """Computes the Singular Value Decomposition."""

        self.U, self.S, self.Vt = np.linalg.svd(self.matrix, full_matrices=False)

        return self.U, self.S, self.Vt

    def truncate_svd(self):
    
        """Retains only the first r singular values and vectors."""

        self.U_r = self.U[:, :self.r]
        self.S_r = self.S[:self.r]
        self.Vt_r = self.Vt[:self.r, :]

        return self.U_r, self.S_r, self.Vt_r

    def low_rank_approximation(self):
    
        """Computes the rank-r low-rank approximation."""

        self.compute_svd()
        self.truncate_svd()

        self.matrix_approx = (self.U_r * self.S_r) @ self.Vt_r

        return self.matrix_approx
