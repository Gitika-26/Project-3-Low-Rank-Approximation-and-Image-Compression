import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

# 1. Load the image and convert to a grayscale matrix
img = Image.open("test_image.jpg").convert('L')
A = np.array(img, dtype=float)

print("Original image shape:", A.shape)

# 2. Compute the SVD of the matrix
U, s, Vt = np.linalg.svd(A, full_matrices=False)

# 3. Choose a rank r for compression ( keeping top 20 singular values)
r = 20

# Slice the matrices to keep only the first r columns/rows
U_r = U[:, :r]
s_r = s[:r]
Vt_r = Vt[:r, :]

# 4. Reconstruct the image using matrix multiplication
# A_approx = U_r * S_r * Vt_r
A_approx = U_r @ np.diag(s_r) @ Vt_r

# Keep pixel values safely between 0 and 255
A_approx = np.clip(A_approx, 0, 255)

# 5. Display the original and compressed images side by side
plt.figure(figsize=(10, 5))

plt.subplot(1, 2, 1)
plt.imshow(A, cmap='gray')
plt.title("Original Image")
plt.axis('off')

plt.subplot(1, 2, 2)
plt.imshow(A_approx, cmap='gray')
plt.title(f"Compressed Image (Rank r = {r})")
plt.axis('off')

plt.show()
