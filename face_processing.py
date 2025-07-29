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

# Optional: PCA (commented out, as in the original)
# from sklearn.decomposition import PCA
# q = 25
# pca = PCA(n_components=q)
# X_pca = pca.fit_transform(X.T).T  # X.T: shape (n_samples, n_features)
# X = X_pca

# Combine X and Y: one attribute vector per row, label in last column
Z = np.vstack([X, Y])
Z = Z.T  # Each row: attributes + label

# Save to ASCII file ("recfaces.dat")
np.savetxt("recfaces.dat", Z, fmt="%.6f")

# Optionally, save X and Y separately as in original commented code
# np.savetxt("yale1_input20x20.txt", X, fmt='%.6f')
# np.savetxt("yale1_output20x20.txt", Y, fmt='%d')
