import numpy as np

from src.metrics import Metrics


# Create an object of Metrics class
metrics = Metrics()


# Original matrix
A = np.array([
    [3, 0, 0],
    [0, 2, 0],
    [0, 0, 1]
], dtype=float)


# Reconstructed matrix
A_r = np.array([
    [3, 0, 0],
    [0, 2, 0],
    [0, 0, 0]
], dtype=float)


# Calculate errors
error = metrics.frobenius_error(A, A_r)

rel_error = metrics.relative_error(A, A_r)


# Calculate storage
m, n = A.shape

original = metrics.storage_original(m, n)

compressed = metrics.storage_svd(m, n, 2)

ratio = metrics.compression_ratio(m, n, 2)


# Display results
print("Original Matrix:")
print(A)

print("\nReconstructed Matrix:")
print(A_r)

print("\nFrobenius Error:", error)

print("Relative Error:", rel_error)

print("Original Storage:", original)

print("SVD Storage:", compressed)

print("Compression Ratio:", ratio)
