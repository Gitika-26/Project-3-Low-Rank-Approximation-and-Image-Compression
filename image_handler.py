import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

from svd_approx import SVD


class ImageHandler:
    """Load, reconstruct, and display grayscale images."""

    def __init__(self, image_path):
        self.image_path = image_path
        self.original_matrix = None
        self.reconstructed_matrix = None
        self.rank = None

    def load_image(self):
        """Load the image and manually convert its RGB pixels to grayscale."""
        with Image.open(self.image_path) as image_file:
            image = image_file.convert("RGB")
            width, height = image.size
            pixels = image.load()
            self.original_matrix = np.empty((height, width), dtype=float)

            for y in range(height):
                for x in range(width):
                    red, green, blue = pixels[x, y]
                    self.original_matrix[y, x] = (
                        0.299 * red + 0.587 * green + 0.114 * blue
                    )

        return self.original_matrix

    def reconstruct(self, rank):
        """Use Gitika's SVD class to create the rank-r reconstruction."""
        if self.original_matrix is None:
            self.load_image()

        self.rank = int(rank)
        svd = SVD(self.original_matrix, self.rank)
        self.reconstructed_matrix = svd.low_rank_approximation()
        return self.reconstructed_matrix

    def reconstruct_for_ranks(self, ranks):
        """Return a dictionary of reconstructed images, one for each rank."""
        reconstructions = {}

        for rank in ranks:
            reconstructions[rank] = self.reconstruct(rank)

        return reconstructions

    def get_singular_values(self):
        """Return singular values for plotting their decay."""
        if self.original_matrix is None:
            self.load_image()

        svd = SVD(self.original_matrix, 1)
        _, singular_values, _ = svd.compute_svd()
        return singular_values

    def display(self):
        """Show the original image beside its reconstruction."""
        if self.reconstructed_matrix is None:
            raise ValueError("Call reconstruct(rank) before display().")

        plt.figure(figsize=(10, 5))

        plt.subplot(1, 2, 1)
        plt.imshow(self.original_matrix, cmap="gray", vmin=0, vmax=255)
        plt.title("Original Image")
        plt.axis("off")

        plt.subplot(1, 2, 2)
        reconstructed = np.clip(self.reconstructed_matrix, 0, 255)
        plt.imshow(reconstructed, cmap="gray", vmin=0, vmax=255)
        plt.title(f"Reconstructed Image (rank = {self.rank})")
        plt.axis("off")

        plt.show()


if __name__ == "__main__":
    image = ImageHandler("images/test_image.jpg")
    image.load_image()
    print("Original image shape:", image.original_matrix.shape)

    rank = int(input("Enter rank r as an integer: "))
    image.reconstruct(rank)
    image.display()
