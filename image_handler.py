import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from svd_approx import low_rank_approximation

# 1. Load the image and manually convert each RGB pixel to grayscale values.
img = Image.open("test_image.jpg").convert("RGB")
width, height = img.size
pixels = img.load()
A = np.empty((height, width), dtype=float)

for y in range(height):
    for x in range(width):
        red, green, blue = pixels[x, y]
        A[y, x] = 0.299 * red + 0.587 * green + 0.114 * blue

print("Original image shape:", A.shape)

# 2. Ask the user for an integer rank r.
r = int(input("Enter the rank r: "))

# 3. Use Gitika's SVD and low-rank reconstruction functions.
A_approx = low_rank_approximation(A, r)

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
