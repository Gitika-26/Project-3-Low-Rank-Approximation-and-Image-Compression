import numpy as np


class Metrics:

    # 1. Calculate Frobenius Error
    def frobenius_error(self, A, A_r):

        A = np.asarray(A)
        A_r = np.asarray(A_r)

        if A.ndim != 2 or A_r.ndim != 2:
            raise ValueError("Both inputs must be 2D matrices.")

        if A.shape != A_r.shape:
            raise ValueError("Both matrices must have the same dimensions.")

        if A.size == 0:
            raise ValueError("Matrices cannot be empty.")

        error = np.linalg.norm(A - A_r, ord='fro')

        return error


    # 2. Calculate Relative Error
    def relative_error(self, A, A_r):

        original_norm = np.linalg.norm(A, ord='fro')

        error = self.frobenius_error(A, A_r)

        if original_norm == 0:
            if error == 0:
                return 0.0
            raise ValueError("Relative error is undefined for a zero matrix.")

        rel_error = error / original_norm

        return rel_error


    # 3. Calculate Original Storage
    def storage_original(self, m, n):

        if not isinstance(m, (int, np.integer)) or isinstance(m, bool) or m <= 0:
            raise ValueError("m must be a positive integer.")

        if not isinstance(n, (int, np.integer)) or isinstance(n, bool) or n <= 0:
            raise ValueError("n must be a positive integer.")

        storage = m * n

        return storage


    # 4. Calculate SVD Storage
    def storage_svd(self, m, n, r):

        self.storage_original(m, n)

        if not isinstance(r, (int, np.integer)) or isinstance(r, bool):
            raise ValueError("Rank must be an integer.")

        if r <= 0 or r > min(m, n):
            raise ValueError("Invalid rank.")

        storage = r * (m + n + 1)

        return storage


    # 5. Calculate Compression Ratio
    def compression_ratio(self, m, n, r):

        original = self.storage_original(m, n)

        compressed = self.storage_svd(m, n, r)

        ratio = original / compressed

        return ratio
