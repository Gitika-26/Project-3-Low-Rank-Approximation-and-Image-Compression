# Project 3: Low-Rank Approximation and Image Compression

A numerical implementation of **Singular Value Decomposition (SVD)** for constructing low-rank approximations of matrices and compressing images using truncated SVD. The project explores the relationship between matrix rank, singular values, reconstruction quality, and storage requirements.

For an $\(m \times n\)$ matrix A, the rank- $\(r\)$ approximation retains only the $\(r\)$ largest singular values and their corresponding singular vectors.

- **Original storage:** $\(m \times n\)$ values.
- **Low-rank SVD storage:** $\(r \times (m+n+1)\)$ values.

This demonstrates how SVD can reduce storage requirements while preserving the most significant information in a matrix.

## Team

| Member | GitHub | Role |
|--------|--------|------|
| MA26M009 (Lead) | [Gitika-26](https://github.com/Gitika-26) | SVD approximation and repository maintenance |
| MA26M010 | [jeetu7014](https://github.com/jeetu7014) | Image handling and reconstruction |
| MA26M011 | [ma26m011](https://github.com/ma26m011) | Error and storage analysis |
| MA26M012 | [msehgal77](https://github.com/msehgal77) | Experiments and visualisation |

## Repository Structure

```text
.
├── svd_approx.py      # SVD class: validation, decomposition and approximation
├── image_handler.py   # ImageHandler class: image loading and reconstruction
├── metrics.py         # Metrics class: error and storage calculations
├── experiments.py     # Experiments class and experiment execution
├── requirements.txt   # Project dependencies
├── images/            # Input images and reconstructed images
├── results/           # Generated plots and CSV files
└── README.md
└── Project_Report.pdf # Short Project Report
```

## Setup and Installation

**Requirements:** Python 3.8 or newer.

### 1. Clone the repository

```bash
git clone https://github.com/Gitika-26/Project-3-Low-Rank-Approximation-and-Image-Compression.git

cd Project-3-Low-Rank-Approximation-and-Image-Compression
```

### 2. Create and activate a virtual environment (optional)

### 3. Install the required dependencies

Install all project dependencies using the provided `requirements.txt` file:

```bash
pip install -r requirements.txt
```

**Note:** Make sure to install the requirements before running any of the project scripts.

## How to Run

Run all commands from the repository root directory, as the scripts use relative paths such as `images/test_image.jpg`.

### 1. Image Reconstruction

Reconstruct an image using a user-specified rank. The program prompts for the rank and displays the original image alongside its reconstruction.

```bash
python image_handler.py
```

### 2. Error and Storage Analysis

Calculate and display reconstruction error and storage requirements for rank 20 on `images/test_image.jpg`.

```bash
python metrics.py
```

### 3. Run All Experiments

Execute all four experiments and generate the corresponding plots and CSV files in the `results/` directory.

```bash
python experiments.py
```

**Note:** The experiments use `plt.show()` to display plots. Close each plot window to proceed to the next experiment.

## Description of the Implementation

### 1. `svd_approx.py` — Class `SVD`

Implements Singular Value Decomposition and low-rank matrix approximation.

| Method | Description |
|--------|-------------|
| `validate_rank_and_dimensions()` | Validates matrix dimensions, numerical values, NaN/inf values, and the requested rank. Raises appropriate errors for invalid inputs. |
| `compute_svd()` | Computes the thin SVD using `numpy.linalg.svd` and returns \(U\), \(S\), and \(V^T\). |
| `truncate_svd()` | Retains the first \(r\) singular values and their corresponding singular vectors. |
| `low_rank_approximation()` | Performs SVD and truncation to construct and return the rank-\(r\) approximation \(A_r\). |

### 2. `image_handler.py` — Class `ImageHandler`

Handles image loading, grayscale conversion, reconstruction, saving, and visualisation.

| Method | Description |
|--------|-------------|
| `load_image()` | Loads an image, converts it to grayscale using \(0.299R + 0.587G + 0.114B\), and returns a floating-point matrix of shape `(height, width)`. |
| `reconstruct(rank)` | Constructs the rank-\(r\) image approximation using the `SVD` class, saves the reconstructed image, and returns the reconstructed matrix. |
| `reconstruct_for_ranks(ranks)` | Reconstructs the image for multiple ranks and returns a dictionary containing the corresponding matrices. |
| `get_singular_values()` | Computes and returns the singular values of the image matrix. |
| `save_reconstructed_image()` | Clips pixel values to the range `[0, 255]` and saves the reconstruction as a PNG image. |
| `display()` | Displays the original image and its reconstruction side by side. |

### 3. `metrics.py` — Class `Metrics`

Provides functions to evaluate reconstruction quality and storage efficiency.

| Method | Description |
|--------|-------------|
| `frobenius_error(A, A_r)` | Calculates the Frobenius norm of the reconstruction error, $\|A-A_r\|_F$. Validates that both matrices have compatible dimensions. |
| `relative_error(A, A_r)` | Calculates the relative reconstruction error, $\frac{\|A-A_r\|_F}{\|A\|_F}$. |
| `storage_original(m, n)` | Calculates the storage required for the original matrix: $\(m times n\)$. |
| `storage_svd(m, n, r)` | Calculates the storage required for the truncated SVD representation: $\(r times (m+n+1)\)$. |
| `compression_ratio(m, n, r)` | Calculates the compression ratio by dividing original storage by SVD storage. |

### 4. `experiments.py` — Class `Experiments`

Runs four experiments to investigate the effect of rank on image reconstruction, error, and compression.

| Method | Experiment | Output in `results/` |
|--------|------------|----------------------|
| `experiment_rank_effect(image_path)` | **Experiment 1:** Compares the original image with reconstructions at different ranks. | `rank_comparison.png` |
| `experiment_error_storage(image_path)` | **Experiment 2:** Examines the relationship between reconstruction error and storage requirements for different ranks. | `error_storage_results.csv`, `error_vs_storage.png` |
| `experiment_singular_values(image_path)` | **Experiment 3:** Studies singular-value decay and cumulative energy. | `singular_values.png` |
| `experiment_two_images()` | **Experiment 4:** Performs the same reconstruction and analysis on two different images. | `test_image_comparison.png`, `test_image_2_comparison.png`, `two_image_comparison.csv` |

## Results and Analysis

The experiments were performed on two grayscale images using ranks 5, 20, and 100. The following values are obtained from `results/two_image_comparison.csv`.

| Image | Rank | Frobenius Error | Relative Error | Compression Ratio |
|-------|------|-----------------|----------------|-------------------|
| test_image.jpg | 5 | 45945.79 | 0.2714 | 119.5 |
| test_image.jpg | 20 | 37365.05 | 0.2207 | 29.9 |
| test_image.jpg | 100 | 26439.02 | 0.1562 | 6.0 |
| test_image_2.jpg | 5 | 49780.45 | 0.0964 | 341.7 |
| test_image_2.jpg | 20 | 23915.96 | 0.0463 | 85.4 |
| test_image_2.jpg | 100 | 8662.47 | 0.0168 | 17.1 |

### Observations

1. **Effect of rank:** Increasing the approximation rank reduces reconstruction error because more singular values and their corresponding components are retained.

2. **Storage requirements:** Increasing the rank also increases the amount of storage required. Therefore, there is a trade-off between reconstruction quality and compression.

3. **Comparison between images:** The second image achieves considerably lower relative reconstruction error and higher compression ratios at the same ranks.

4. **Singular-value decay:** The second image's singular values decay more rapidly. Consequently, a smaller number of singular components captures a larger proportion of its information, making it more suitable for low-rank compression.

### Conclusion

The experiments demonstrate that Singular Value Decomposition provides an effective method for low-rank matrix approximation and image compression. By retaining only the most significant singular values, substantial storage reduction can be achieved while preserving important image information.

However, the effectiveness of compression depends on the image itself and the rate at which its singular values decay. Selecting an appropriate rank is therefore essential to balancing reconstruction quality and storage efficiency.

---

**Technologies Used:** Python, NumPy, Pillow, Matplotlib, Pandas.
