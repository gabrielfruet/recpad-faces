"""
Routines for opening face images and converting them to column vectors
by stacking the columns of the face matrix one beneath the other.

Last modification: 10/08/2021
Author: Guilherme Barreto (Python conversion by GitHub Copilot)
"""

from pathlib import Path

from PIL import Image
import numpy as np
import cv2
import os
import argparse

parser = argparse.ArgumentParser(
    description="Process face images and convert to column vectors."
)

parser.add_argument(
    "--input_dir", type=Path, required=True, help="Directory containing face images"
)
parser.add_argument(
    "--output_file",
    type=Path,
    default="recfaces.dat",
    help="Output file name for processed data",
)
parser.add_argument(
    "--resize",
    type=int,
    default=30,
    help="Resize images to this dimension (default: 30x30)",
)

parser.add_argument(
    "-pca",
    nargs="?",
    const=0.98,
    type=float,
    default=None,
    help="Apply PCA to reduce dimensionality and optionally set a variance threshold (e.g., 0.95)",
)

parser.add_argument(
    "--box_cox",
    action="store_true",
    help="Apply Box-Cox transformation",
)


args = parser.parse_args()

input_dir = Path(args.input_dir)
output_file = args.output_file
resize_size: int = args.resize

# Phase 1 -- Load available images
part1 = "subject0"
part2 = "subject"
part3 = [
    ".centerlight",
    ".glasses",
    ".happy",
    ".leftlight",
    ".noglasses",
    ".normal",
    ".rightlight",
    ".sad",
    ".sleepy",
    ".surprised",
    ".wink",
]

Nind = 15  # Number of individuals (classes)
Nexp = len(part3)  # Number of expressions

X = []  # Accumulate vectorized images
Y = []  # Accumulate label (individual identifier)
NAME = []  # Accumulate filenames

for i in range(1, Nind + 1):  # Index for individuals
    individuo = i
    for j in range(Nexp):  # Index for expressions
        if i < 10:
            nome = input_dir / f"{part1}{i}{part3[j]}"
        else:
            nome = input_dir / f"{part2}{i}{part3[j]}"

        # Append file extension if needed (assume .pgm; change if necessary)
        # For Yale Face Database, files often have no extension, so this is fine.
        filename = nome
        NAME.append(filename)

        # Read image (assumes grayscale)
        if not filename.exists():
            print(f"Warning: File not found: {filename}")
            continue

        Img = Image.open(filename)
        Img = Img.convert("L")

        if Img is None:
            print(f"Warning: Unable to read image: {filename}")
            continue

        # (Optional) Resize image to 30x30
        Ar = Img.resize((resize_size, resize_size))

        # (Optional) Add gaussian noise (disabled by default)
        # An = Ar + np.random.normal(0, 0.005, Ar.shape)
        An = Ar
        An = np.array(An)

        # Convert to double precision
        A = An.astype(np.float64) / 255.0

        # Vectorization: stack columns
        a = A.flatten(order="F")  # Fortran order = column stacking

        # Label = individual index
        ROT = i

        X.append(a)
        Y.append(ROT)

X = np.array(X).T  # Each image is a column in X
Y = np.array(Y).reshape(1, -1)  # Y is a row vector

if args.box_cox:
    from scipy.stats import boxcox

    X = X.clip(min=1e-4)  # Avoid log(0) issues

    for f in range(X.shape[0]):
        bc_result = boxcox(X[f], lmbda=None)
        assert len(bc_result) == 2, (
            "Box-Cox transformation failed, expected two outputs."
        )
        Xf = bc_result[0]
        assert isinstance(Xf, np.ndarray), (
            "Box-Cox transformation output is not a numpy array."
        )
        X[f] = Xf


if args.pca:
    Xc = X - np.mean(X, axis=1, keepdims=True)  # Center the data
    u, s, vh = np.linalg.svd(Xc.T)
    explained_variance_ratio = s**2 / np.sum(s**2)

    # This `-1` check is unreachable with the current argparse setup but left for consistency.
    if args.pca == -1:
        q = X.shape[0]
    else:
        # Correctly find the number of components (q) needed to reach the threshold.
        # argmax() finds the first index where the condition is True. Add 1 for the total count.
        q = (np.cumsum(explained_variance_ratio) >= args.pca).argmax() + 1

        # --- Scree Plot Generation ---
        import matplotlib.pyplot as plt

        plt.style.use("seaborn-v0_8-whitegrid")
        fig, ax1 = plt.subplots(figsize=(12, 7))
        plt.title(
            f"Scree Plot & Variância Cumulativa ({args.pca * 100:.0f}% Threshold)"
        )

        component_numbers = np.arange(1, len(explained_variance_ratio) + 1)
        cumulative_variance = np.cumsum(explained_variance_ratio)

        # Plot Individual Explained Variance (Bar Plot)
        ax1.bar(
            component_numbers,
            explained_variance_ratio,
            alpha=0.7,
            color="deepskyblue",
            label="Variância Individual",
        )
        ax1.set_xlabel("Número de componentes principais")
        ax1.set_ylabel("Razão de Variância Explicada", color="deepskyblue")
        ax1.tick_params(axis="y", labelcolor="deepskyblue")

        # Plot Cumulative Explained Variance (Line Plot on secondary axis)
        ax2 = ax1.twinx()
        ax2.plot(
            component_numbers,
            cumulative_variance,
            color="crimson",
            marker=".",
            linestyle="-",
            label="Variância Cumulativa",
        )
        ax2.set_ylabel("Razão da Variância Cumulativa Explicada", color="crimson")
        ax2.tick_params(axis="y", labelcolor="crimson")
        ax2.set_ylim(0, 1.05)

        # Highlight the threshold and selected point 'q'
        ax2.axhline(
            y=args.pca,
            color="grey",
            linestyle="--",
            label=f"{args.pca * 100:.0f}% Threshold",
        )
        ax1.axvline(
            x=q,
            color="darkgreen",
            linestyle="--",
            label=f"{q} Componentes Selecionadas",
        )

        # Mark the exact point on the cumulative curve
        ax2.plot(
            q, cumulative_variance[q - 1], "go", markersize=10, markeredgecolor="k"
        )

        # Combine legends from both axes
        lines, labels = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax2.legend(lines + lines2, labels + labels2, loc="center right")

        fig.tight_layout()

        # Construct the output path and save the plot
        plot_filename = output_file.stem + "_scree_plot.png"
        plot_path = output_file.parent / plot_filename
        plt.savefig(plot_path, dpi=300)
        plt.close(fig)
        print(f"Scree plot saved to: {plot_path}")

    # Project data onto the first q principal components
    # 1. Perform the standard PCA projection
    X_projected = vh @ Xc

    # # 2. Calculate the variance (eigenvalues) of each component
    # n_samples = Xc.shape[1]
    # # The variance of component k is s[k]**2 / (n_samples - 1)
    # variances = s**2 / (n_samples - 1)
    #
    # # 3. Divide each component by its standard deviation to "whiten" the data
    # # Add a small epsilon to the denominator for numerical stability
    # std_devs = np.sqrt(variances) + 1e-9
    # X_whitened = X_projected / std_devs[:, np.newaxis]
    #
    # # Now use the whitened data for the rest of the script
    # X = X_whitened
    X = X_projected

    X = X[:q, :]


# Combine X and Y: one attribute vector per row, label in last column
Z = np.vstack([X, Y])
Z = Z.T  # Each row: attributes + label

# Save to ASCII file ("recfaces.dat")
np.savetxt(output_file, Z, fmt="%.6f")

# Optionally, save X and Y separately as in original commented code
# np.savetxt("yale1_input20x20.txt", X, fmt='%.6f')
# np.savetxt("yale1_output20x20.txt", Y, fmt='%d')
