
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os

from image_handler import ImageHandler
from src.metrics import Metrics


class Experiments:
    """Conduct experiments for SVD image compression."""

    def __init__(self, image_paths, ranks=(5, 20, 100)):
        self.image_paths = image_paths
        self.ranks = ranks
        self.metrics = Metrics()

        os.makedirs("results", exist_ok=True)

    def experiment_rank_effect(self, image_path):

        image = ImageHandler(image_path)
        original = image.load_image()

        reconstructions = image.reconstruct_for_ranks(self.ranks)

        plt.figure(figsize=(4 * (len(self.ranks) + 1), 5))

        plt.subplot(1, len(self.ranks) + 1, 1)
        plt.imshow(original, cmap="gray", vmin=0, vmax=255)
        plt.title("Original Image")
        plt.axis("off")

        for i, rank in enumerate(self.ranks):

            reconstructed = np.clip(
                reconstructions[rank], 0, 255
            )

            plt.subplot(1, len(self.ranks) + 1, i + 2)
            plt.imshow(reconstructed, cmap="gray", vmin=0, vmax=255)
            plt.title(f"Rank = {rank}")
            plt.axis("off")

        plt.tight_layout()
        plt.savefig("results/rank_comparison.png", dpi=300)
        plt.show()
        plt.close()

    def experiment_error_storage(self, image_path):

        image = ImageHandler(image_path)
        A = image.load_image()

        m, n = A.shape
        rows = []

        for rank in self.ranks:

            A_r = image.reconstruct(rank)

            f_error = self.metrics.frobenius_error(A, A_r)
            r_error = self.metrics.relative_error(A, A_r)

            original_storage = self.metrics.storage_original(m, n)
            compressed_storage = self.metrics.storage_svd(m, n, rank)

            ratio = self.metrics.compression_ratio(m, n, rank)

            rows.append({
                "Rank": rank,
                "Frobenius Error": f_error,
                "Relative Error": r_error,
                "Original Storage": original_storage,
                "Compressed Storage": compressed_storage,
                "Compression Ratio": ratio})

        df = pd.DataFrame(rows)

        df.to_csv(
            "results/error_storage_results.csv",
            index=False)

        # Plot storage vs Frobenius error
        plt.figure(figsize=(8, 5))

        plt.plot(
            df["Compressed Storage"],
            df["Frobenius Error"],
            marker="o"
        )

        plt.xlabel("Compressed Storage (number of values)")
        plt.ylabel("Frobenius Error")
        plt.title("Error vs Storage")
        plt.grid(True)

        plt.savefig("results/error_vs_storage.png", dpi=300)
        plt.show()
        plt.close()

        print("\nError vs Storage Results:")
        print(df)

    def experiment_singular_values(self, image_path):

        image = ImageHandler(image_path)

        singular_values = image.get_singular_values()

       
        energy = singular_values ** 2

       
        cumulative_energy = (
            np.cumsum(energy) / np.sum(energy)
        )

        plt.figure(figsize=(12, 5))

        # Singular value decay
        plt.subplot(1, 2, 1)
        plt.plot(singular_values)
        plt.xlabel("Singular Value Index")
        plt.ylabel("Singular Value")
        plt.title("Singular Value Decay")
        plt.grid(True)

        # Cumulative energy
        plt.subplot(1, 2, 2)
        plt.plot(cumulative_energy)
        plt.xlabel("Rank")
        plt.ylabel("Cumulative Energy")
        plt.title("Cumulative Energy vs Rank")
        plt.grid(True)

        plt.tight_layout()

        plt.savefig("results/singular_values.png", dpi=300)
        plt.show()
        plt.close()

    def experiment_two_images(self):

        rows = []

        for image_path in self.image_paths:

            image = ImageHandler(image_path)
            A = image.load_image()

            m, n = A.shape

            reconstructions = image.reconstruct_for_ranks(
                self.ranks
            )

            # Plot original and reconstructions
            plt.figure(
                figsize=(4 * (len(self.ranks) + 1), 5)
            )

            plt.subplot(1, len(self.ranks) + 1, 1)
            plt.imshow(A, cmap="gray", vmin=0, vmax=255)
            plt.title("Original Image")
            plt.axis("off")

            for i, rank in enumerate(self.ranks):

                A_r = reconstructions[rank]
                A_r = np.clip(A_r, 0, 255)

                plt.subplot(1, len(self.ranks) + 1, i + 2)
                plt.imshow(A_r, cmap="gray", vmin=0, vmax=255)
                plt.title(f"Rank = {rank}")
                plt.axis("off")

                rows.append({
                    "Image": os.path.basename(image_path),
                    "Rank": rank,
                    "Frobenius Error":
                        self.metrics.frobenius_error(A, A_r),
                    "Relative Error":
                        self.metrics.relative_error(A, A_r),
                    "Compression Ratio":
                        self.metrics.compression_ratio(m, n, rank)
                })

            plt.tight_layout()

            filename = os.path.basename(image_path)
            filename = os.path.splitext(filename)[0]

            plt.savefig(
                f"results/{filename}_comparison.png",
                dpi=300
            )

            plt.show()
            plt.close()

        df = pd.DataFrame(rows)

        df.to_csv(
            "results/two_image_comparison.csv",
            index=False
        )

        print("\nTwo Image Comparison Results:")
        print(df)
#Running the experiments

image1 = "images/test_image.jpg"
image2 = "images/test_image_2.jpg"

ranks = (5, 20, 100)

my_experiments = Experiments(
    image_paths=[image1, image2],
    ranks=ranks
)

print("Experiment 1: Rank Effect")
my_experiments.experiment_rank_effect(image1)

print("Experiment 2: Error vs Storage")
my_experiments.experiment_error_storage(image1)

print("Experiment 3: Singular Values")
my_experiments.experiment_singular_values(image1)

print("Experiment 4: Comparing Two Images")
my_experiments.experiment_two_images()

print("\nALL EXPERIMENTS COMPLETED!")
