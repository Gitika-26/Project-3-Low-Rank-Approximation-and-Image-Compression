
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os

from src.image import load_image, compress_image, show_comparison
from src.svd_approx import compute_svd
from src.metrics import (
    frobenius_error,
    relative_error,
    storage_original,
    storage_svd,
    compression_ratio)

# Create results folder
os.makedirs("results", exist_ok=True)

def experiment_rank_effect(image_path, ranks=(5, 20, 100), save_dir="results"):

    A = load_image(image_path)

    reconstructions = []
    valid_ranks = []

    max_rank = min(A.shape)

    for r in ranks:

        if r <= 0 or r > max_rank:
            print(f"Skipping rank {r}")
            continue

        A_approx = compress_image(A, r)

        reconstructions.append(A_approx)
        valid_ranks.append(r)

    if len(valid_ranks) == 0:
        raise ValueError("No valid ranks available")

    os.makedirs(save_dir, exist_ok=True)

    save_path = os.path.join(save_dir, "rank_comparison.png")

    show_comparison(
        A,
        reconstructions,
        valid_ranks,
        save_path=save_path)

    print("Rank comparison saved:", save_path)

    return reconstructions
    
def experiment_error_storage(image_path, ranks, save_dir="results"):

    A = load_image(image_path)

    m, n = A.shape

    results = []

    for r in ranks:

        if r <= 0 or r > min(m, n):
            print(f"Skipping rank {r}")
            continue

        A_approx = compress_image(A, r)

        error = frobenius_error(A, A_approx)
        rel_error = relative_error(A, A_approx)

        original = storage_original(m, n)
        compressed = storage_svd(m, n, r)

        ratio = compression_ratio(m, n, r)

        results.append({
            "Rank": r,
            "Frobenius Error": error,
            "Relative Error": rel_error,
            "Original Storage": original,
            "Compressed Storage": compressed,
            "Compression Ratio": ratio})

    if len(results) == 0:
        raise ValueError("No valid ranks available")

    df = pd.DataFrame(results)

    os.makedirs(save_dir, exist_ok=True)

    df.to_csv(
        os.path.join(save_dir, "error_storage.csv"),
        index=False)

    print("\nError versus Storage Results:")
    print(df.to_string(index=False))

    plot_error_vs_storage(
        results,
        os.path.join(save_dir, "error_vs_storage.png"))

    return df

def experiment_singular_values(image_path, save_dir="results"):

    A = load_image(image_path)

    U, S, Vt = compute_svd(A)

    energy = S ** 2

    total_energy = np.sum(energy)

    if total_energy == 0:
        cumulative_energy = np.zeros(len(S))
    else:
        cumulative_energy = np.cumsum(energy) / total_energy

    os.makedirs(save_dir, exist_ok=True)

    # Plot singular values
    plot_singular_values(
        S,
        os.path.join(save_dir, "singular_values.png"))

    # Plot cumulative energy
    plt.figure(figsize=(8, 5))

    plt.plot(
        np.arange(1, len(S) + 1),
        cumulative_energy)

    plt.xlabel("Number of Singular Values")
    plt.ylabel("Cumulative Energy")
    plt.title("Cumulative Energy Contribution")
    plt.grid(True)

    plt.tight_layout()

    plt.savefig(
        os.path.join(save_dir, "cumulative_energy.png"))

    plt.show()

    return S, cumulative_energy

def experiment_second_image(image_paths, ranks, save_dir="results"):

    all_results = []

    for image_path in image_paths:

        A = load_image(image_path)

        m, n = A.shape

        reconstructions = []
        valid_ranks = []

        for r in ranks:

            if r <= 0 or r > min(m, n):
                print(f"Skipping rank {r} for {image_path}")
                continue

            A_approx = compress_image(A, r)

            error = frobenius_error(A, A_approx)
            rel_error = relative_error(A, A_approx)
            ratio = compression_ratio(m, n, r)

            all_results.append({
                "Image": os.path.basename(image_path),
                "Rank": r,
                "Frobenius Error": error,
                "Relative Error": rel_error,
                "Compression Ratio": ratio})

            reconstructions.append(A_approx)
            valid_ranks.append(r)

        if valid_ranks:

            filename = os.path.splitext(
                os.path.basename(image_path))[0]

            show_comparison(
                A,
                reconstructions,
                valid_ranks,
                save_path=os.path.join(
                    save_dir,
                    filename + "_comparison.png"))

    if len(all_results) == 0:
        raise ValueError("No valid image and rank combinations")

    df = pd.DataFrame(all_results)

    os.makedirs(save_dir, exist_ok=True)

    df.to_csv(
        os.path.join(save_dir, "second_image_comparison.csv"),
        index=False)

    print("\nComparison of Two Images:")
    print(df.to_string(index=False))

    return df

def plot_rank_comparison(original, reconstructions, ranks, save_path):

    show_comparison(
        original,
        reconstructions,
        ranks,
        save_path=save_path)

def plot_error_vs_storage(results, save_path):

    storage = [item["Compressed Storage"] for item in results]
    errors = [item["Frobenius Error"] for item in results]

    plt.figure(figsize=(8, 5))

    plt.plot(storage, errors, marker="o")

    plt.xlabel("Compressed Storage")
    plt.ylabel("Frobenius Error")
    plt.title("Error versus Storage")

    plt.grid(True)
    plt.tight_layout()

    plt.savefig(save_path)
    plt.show()

def plot_singular_values(s, save_path):

    plt.figure(figsize=(8, 5))

    plt.plot(
        np.arange(1, len(s) + 1),
        s,
        marker=".")

    plt.xlabel("Singular Value Index")
    plt.ylabel("Singular Value")
    plt.title("Singular Value Distribution")

    plt.grid(True)
    plt.tight_layout()

    plt.savefig(save_path)
    plt.show()
# image paths
image1 = "test_image.jpg"
image2 = "test_image_2.jpg"

ranks = (5, 20, 100)


print("\nEXPERIMENT 1: Rank Effect")

experiment_rank_effect(image1, ranks)


print("\nEXPERIMENT 2: Error versus Storage")

experiment_error_storage(image1, ranks)


print("\nEXPERIMENT 3: Singular Values")

experiment_singular_values(image1)


print("\nEXPERIMENT 4: Comparison of Two Images")

experiment_second_image(
    [image1, image2],
    ranks)
print("\nALL EXPERIMENTS COMPLETED!")