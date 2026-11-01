import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

def load_image_as_matrix(image_path: str, grayscale: bool = True) -> np.ndarray:
    """
    Load an image from disk and represent it as a numerical matrix.
    """
    try:
        img = Image.open(image_path)
    except FileNotFoundError:
        raise FileNotFoundError(f"Image not found at path: {image_path}")
        
    if grayscale:
        img = img.convert('L')
    
    # Convert image pixels to a floating-point numpy array
    return np.array(img, dtype=float)

def reconstruct_image_matrix(U: np.ndarray, s: np.ndarray, Vt: np.ndarray, r: int) -> np.ndarray:
    """
    Construct a rank-r approximation of a matrix using SVD components.
    """
    m, n = U.shape[0], Vt.shape[1]
    if r <= 0 or r > min(m, n):
        raise ValueError(f"Invalid rank r = {r}. Must be between 1 and {min(m, n)}.")
    
    # Slice the leading r components
    Ur = U[:, :r]
    sr = s[:r]
    Vtr = Vt[:r, :]
    
    # Reconstruct the low-rank matrix
    return Ur @ np.diag(sr) @ Vtr

def display_side_by_side(original: np.ndarray, reconstructed: np.ndarray, r: int):
    """
    Display original and rank-r reconstructed images side by side for comparison.

    """
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    
    axes[0].imshow(original, cmap='gray')
    axes[0].set_title("Original Image")
    axes[0].axis('off')
    
    # Clip pixel values to valid image range [0, 255] and cast to uint8
    rec_clipped = np.clip(reconstructed, 0, 255).astype(np.uint8)
    axes[1].imshow(rec_clipped, cmap='gray')
    axes[1].set_title(f"Rank-{r} Reconstructed Image")
    axes[1].axis('off')
    
    plt.tight_layout()
    plt.show()





