# Project 3: Low-Rank Approximation and Image Compression

A numerical implementation that uses the singular value decomposition (SVD) to build low-rank approximations of matrices and to compress images with truncated SVD. It connects the idea of rank and singular values to measurable reconstruction quality and storage cost.

For an **m x n** matrix A with SVD A = U S Vt, the rank-r approximation keeps only the r largest singular values and their singular vectors:

**A_r = U[:, :r] @ diag(S[:r]) @ Vt[:r, :]**


Storing the original needs **m x n** values, while the rank- **r** factors need about **r x (m + n + 1)** values.

## Team

| Member | GitHub | Role |
|--------|--------|------|
| MA26M009 (Lead) | Gitika-26 | SVD approximation and repository maintenance |
| MA26M010 | jeetu7014 | Image handling and reconstruction |
| MA26M011 | ma26m011 | Error and storage analysis |
| MA26M012 | msehgal77 | Experiments and visualisation |

## Repository structure

```
.
├── svd_approx.py      # SVD class: validation, SVD, truncation, rank-r approximation
├── image_handler.py   # ImageHandler class: load, reconstruct, save, display images
├── metrics.py         # Metrics class: error, storage and compression-ratio formulas
├── experiments.py     # Experiments class + script that runs all four experiments
├── images/            # Input images and saved reconstructions
├── results/           # Generated plots and CSV tables
└── README.md
```

## Setup

Requires Python 3.8 or newer.

1. Clone the repository:

   ```bash
   git clone https://github.com/Gitika-26/Project-3-Low-Rank-Approximation-and-Image-Compression.git
   cd Project-3-Low-Rank-Approximation-and-Image-Compression
   ```

2. (Optional) Create and activate a virtual environment:

   ```bash
   python -m venv venv
   source venv/bin/activate        # Windows: venv\Scripts\activate
   ```

3. Install the dependencies:

   ```bash
   pip install numpy pillow matplotlib pandas
   ```

## How to run

Run all commands from the repository root, since the scripts use relative paths such as `images/test_image.jpg`.

**Reconstruct one image for a rank of your choice** (prompts for `r`, then shows the original next to the reconstruction):

```bash
python image_handler.py
```

**Print error and storage figures for one rank** (rank 20 on `images/test_image.jpg`):

```bash
python metrics.py
```

**Run all four experiments** (reproduces everything in `results/`):

```bash
python experiments.py
```

The experiments call `plt.show()`, so each plot window must be closed before the next experiment starts. 

## Description of the functions

### `svd_approx.py` - class `SVD(matrix, r)`

| Method | Description |
|--------|-------------|
| `validate_rank_and_dimensions()` | Called on construction. Raises an error if the input is not 2-D, not numeric, contains NaN/inf, or if `r` is not a positive integer or exceeds `min(m, n)`. |
| `compute_svd()` | Computes the thin SVD with `numpy.linalg.svd` and returns `U, S, Vt`. |
| `truncate_svd()` | Keeps the leading `r` singular values and the matching columns of `U` and rows of `Vt`. |
| `low_rank_approximation()` | Runs the two steps above and returns the rank-`r` matrix `A_r`. |

### `image_handler.py` - class `ImageHandler(image_path)`

| Method | Description |
|--------|-------------|
| `load_image()` | Opens the image, converts RGB to grayscale with `0.299 R + 0.587 G + 0.114 B`, and returns it as a float matrix of shape `(height, width)`. |
| `reconstruct(rank)` | Builds the rank-`r` reconstruction using the `SVD` class, saves it as `reconstructed_image_rank_<r>.png` next to the source image, and returns the matrix. |
| `reconstruct_for_ranks(ranks)` | Calls `reconstruct` for each rank and returns a `{rank: matrix}` dictionary. |
| `get_singular_values()` | Returns all singular values of the image matrix, for decay plots. |
| `save_reconstructed_image()` | Clips values to `[0, 255]` and writes the reconstruction as a PNG. |
| `display()` | Shows the original and the reconstruction side by side. |

### `metrics.py` - class `Metrics`

| Method | Description |
|--------|-------------|
| `frobenius_error(A, A_r)` | Returns `||A - A_r||_F`. Checks that both inputs are non-empty 2-D matrices of the same shape. |
| `relative_error(A, A_r)` | Returns `||A - A_r||_F / ||A||_F`. |
| `storage_original(m, n)` | Returns `m*n`. |
| `storage_svd(m, n, r)` | Returns `r*(m + n + 1)`. Rejects invalid ranks. |
| `compression_ratio(m, n, r)` | Returns `storage_original / storage_svd`. |

### `experiments.py` - class `Experiments(image_paths, ranks=(5, 20, 100))`

| Method | Experiment | Output in `results/` |
|--------|------------|----------------------|
| `experiment_rank_effect(image_path)` | 1. Effect of rank: original and reconstructions side by side | `rank_comparison.png` |
| `experiment_error_storage(image_path)` | 2. Error versus storage for each rank | `error_storage_results.csv`, `error_vs_storage.png` |
| `experiment_singular_values(image_path)` | 3. Singular-value decay and cumulative energy | `singular_values.png` |
| `experiment_two_images()` | 4. The same analysis on both images | `test_image_comparison.png`, `test_image_2_comparison.png`, `two_image_comparison.csv` |

## Results summary

Ranks 5, 20 and 100 were used on both images (grayscale). Values come from `results/two_image_comparison.csv`.

| Image | Rank | Frobenius error | Relative error | Compression ratio |
|-------|------|-----------------|----------------|-------------------|
| test_image.jpg | 5 | 45945.79 | 0.2714 | 119.5 |
| test_image.jpg | 20 | 37365.05 | 0.2207 | 29.9 |
| test_image.jpg | 100 | 26439.02 | 0.1562 | 6.0 |
| test_image_2.jpg | 5 | 49780.45 | 0.0964 | 341.7 |
| test_image_2.jpg | 20 | 23915.96 | 0.0463 | 85.4 |
| test_image_2.jpg | 100 | 8662.47 | 0.0168 | 17.1 |

Higher rank lowers the error but raises the storage cost. The second image compresses much better, because its singular values decay faster, so a few components capture most of its content.
