from enum import Enum
import numpy as np
import logging
from typing import List, Tuple
import time

log = logging.getLogger(__name__)


class QdaMethods(Enum):
    FRIEDMAN = "friedman"
    POOLED = "pooled"
    DEFAULT = "default"
    DIAGONAL = "diagonal"
    TIKHONOV = "tikhonov"


class NormalizationMethods(Enum):
    Z_SCORE = "z_score"
    NO_NORMALIZATION = "none"
    SCALE_CHANGE = "scale_change"


def normalize(
    X: np.ndarray, method: NormalizationMethods = NormalizationMethods.Z_SCORE
):
    if method is NormalizationMethods.Z_SCORE:
        med = np.mean(X, axis=1, keepdims=True)
        dp = np.std(X, axis=1, keepdims=True)
        X = (X - med) / dp
    elif method is NormalizationMethods.SCALE_CHANGE:
        X = (X - X.min()) / np.ptp(X)
        X = (X - 0.5) * 2
    elif method is NormalizationMethods.NO_NORMALIZATION:
        pass

    return X


def estatisticas_classes(
    Xtrn, Ytrn, C, method: QdaMethods = QdaMethods.DEFAULT, **kwargs
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    F = Xtrn.shape[1]  # Number of features
    M = np.zeros((C, F))
    S_k = np.zeros((C, F, F))
    posto_k = np.zeros(C)
    for k in range(1, C + 1):
        Ic = np.where(Ytrn == k)[0]
        Xc = Xtrn[Ic]
        mu_k = np.mean(Xc, axis=0)
        cov_k = np.cov(Xc, rowvar=False, bias=True)
        if method is QdaMethods.TIKHONOV:
            λ = kwargs["λ"]
            cov_k += λ * np.eye(F)
        elif method is QdaMethods.DIAGONAL:
            cov_k = np.eye(F) * cov_k

        rank_k = np.linalg.matrix_rank(cov_k)
        M[k - 1] = mu_k
        S_k[k - 1] = cov_k
        posto_k[k - 1] = rank_k

    priors_occur = np.array([(np.sum(Ytrn == k + 1) - 1) for k in range(C)])
    f_wi = priors_occur / np.sum(priors_occur)

    if method is QdaMethods.FRIEDMAN or method is QdaMethods.POOLED:
        num_samples_train = len(Ytrn)
        num_classes = C
        priors_occur = priors_occur[..., np.newaxis, np.newaxis]
        sum_scatter_matrix = np.sum(priors_occur * S_k, axis=0) / (num_samples_train)
        C_pooled = sum_scatter_matrix / (num_samples_train)

        if method is QdaMethods.FRIEDMAN:
            λ = kwargs["λ"]
            for k in range(C):
                S_k[k] = ((1 - λ) * S_k[k] + λ * C_pooled) / (
                    (1 - λ) * sum_scatter_matrix[k] + λ * len(Ytrn)
                )
        else:
            for k in range(C):
                S_k[k] = C_pooled

    return M, S_k, posto_k, f_wi


def discriminant(X, means, inv_covs, f_wi=None) -> np.ndarray:
    """
    Vectorized Mahalanobis distance calculation for multiple samples and classes

    Parameters:
    -----------
    X : numpy.ndarray, shape (n_samples, n_features)
        Test samples
    means : numpy.ndarray, shape (n_classes, n_features)
        Class centroids
    inv_covs : numpy.ndarray, shape (n_classes, n_features, n_features)
        Inverse covariance matrices for each class
    f_wi: numpy.ndarray, optional, shape (n_classes,)
        Frequency of each class

    Returns:
    --------
    distances : numpy.ndarray, shape (n_samples, n_classes)
        Mahalanobis distances from each sample to each class centroid
    """
    n_samples, n_features = X.shape
    n_classes = means.shape[0]

    # Initialize distances matrix
    distances = np.zeros((n_samples, n_classes))

    # Compute distances for each class
    for k in range(n_classes):
        # Difference between samples and class mean
        # (n_samples, n_features)
        diff = X - means[k]

        # Compute Mahalanobis distance
        # (x-μ)ᵀΣ⁻¹(x-μ) for each sample
        mahal_term = np.einsum("ij,jk,ik->i", diff, inv_covs[k], diff)
        _, logdet = np.linalg.slogdet(inv_covs[k])
        distances[:, k] = mahal_term + 0.5 * logdet
        if f_wi is not None:
            distances[:, k] -= 2 * np.log(f_wi[k] + 1e-2)

    return distances


def quadratico(
    D: np.ndarray,
    Nr: int,
    Ptrain: int,
    λ: float = 0.01,
    method: QdaMethods | str = QdaMethods.DEFAULT,
) -> Tuple[
    np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray
]:
    """
    Quadratico classifier implementation in Python with optimized testing phase
    Classifier based on Mahalanobis distance to class centroids, using class-specific covariance matrices
    Returns:
        STATS: List[float] -- [mean, std, median, min, max] of accuracy over Nr repetitions
        TX_OK: np.ndarray -- List of accuracy rates for each run
        X: np.ndarray -- Normalized feature matrix used
        m: List[List[np.ndarray]] -- List of class centroids per repetition
        S: List[List[np.ndarray]] -- List of class covariance matrices per repetition
        posto: List[List[int]] -- List of ranks of covariance matrices per repetition
    """

    if isinstance(method, str):
        method = QdaMethods(method.lower())
    X = D[:, :-1].copy()
    Y = D[:, -1].astype(int).copy()
    Nrep = Nr
    Ptrn = Ptrain / 100.0

    log.info(f"Using method: {method}")

    # Z-score normalization
    med = np.mean(X, axis=1, keepdims=True)
    dp = np.std(X, axis=1, keepdims=True)
    X = (X - med) / dp

    N = len(Y)
    Nc = X.shape[0]
    C = np.max(Y)

    Ntrn = int(np.floor(Ptrn * N))

    Pacerto = []
    m = []
    S = []
    posto = []
    P_failed_inversions = []

    for r in range(Nrep):
        log.info(f"Repetition {r + 1}/{Nrep}")
        # Shuffle columns
        idx = np.random.permutation(N)
        X_shuf = X[idx]
        Y_shuf = Y[idx]

        # Training set
        Xtrn = X_shuf[:Ntrn]
        Ytrn = Y_shuf[:Ntrn]

        # Centroids and covariance matrices per class
        tic = time.perf_counter_ns()
        M, S_k, posto_k, f_wi = estatisticas_classes(Xtrn, Ytrn, C, method=method, λ=λ)
        toc = time.perf_counter_ns()
        log.info(
            f"Time for calculating centroids and covariance matrices: {(toc - tic) / 1e6:.2f} ms"
        )

        m.append(M)
        S.append(S_k)
        posto.append(posto_k)

        # Testing set
        Xtst = X_shuf[Ntrn:]
        Ytst = Y_shuf[Ntrn:]
        Ntst = Xtst.shape[0]

        # ============== OPTIMIZED TESTING PHASE ==============
        tic = time.perf_counter_ns()

        # Pre-compute inverse covariance matrices
        inv_covs = np.zeros_like(S_k)
        failed_inversions = 0
        for k in range(C):
            try:
                # regularization
                inv_covs[k] = np.linalg.inv(S_k[k])
                log.debug(
                    f"Max value of covariance matrix for class {k + 1}: {np.max(S_k[k])}"
                )
            except np.linalg.LinAlgError:
                failed_inversions += 1
                log.debug(
                    f"Covariance matrix for class {k + 1} is singular, using pseudo-inverse."
                )
                inv_covs[k] = np.linalg.pinv(S_k[k])

        log.info(f"Percentage of failed inversions: {100 * failed_inversions / C}%")

        # Calculate all distances at once
        distances = discriminant(Xtst, M, inv_covs, f_wi)

        # Find predicted classes (add 1 because class indices start at 1)
        predicted_classes = np.argmin(distances, axis=1) + 1

        # Count correct predictions
        acerto = np.sum(predicted_classes == Ytst)

        toc = time.perf_counter_ns()
        log.info(f"Time for testing : {(toc - tic) / 1e6:.2f} ms")
        Pacerto.append(100 * acerto / Ntst)
        P_failed_inversions.append(100 * failed_inversions / C)

    TX_OK = np.array(Pacerto)
    STATS = np.array(
        [np.mean(TX_OK), np.std(TX_OK), np.median(TX_OK), np.min(TX_OK), np.max(TX_OK)]
    )
    m = np.array(m)
    S = np.array(S)
    posto = np.array(posto)
    P_failed_inversions = np.array(P_failed_inversions)
    log.info(f"Posto: {posto}")
    return STATS, TX_OK, X, m, S, posto, P_failed_inversions


def classificador_1nn(
    D: np.ndarray,
    Nr: int,
    Ptrain: int,
    method: NormalizationMethods = NormalizationMethods.Z_SCORE,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    1-Nearest Neighbor classifier.
    Returns: (stats, accuracies, Xnorm)
    """
    X = D[:, :-1].copy()
    Y = D[:, -1].astype(int).copy()
    Nrep = Nr
    Ptrn = Ptrain / 100.0

    # Z-score normalization
    X = normalize(X, method)

    N = len(Y)
    Ntrn = int(np.floor(Ptrn * N))

    Pacerto = []

    for r in range(Nrep):
        idx = np.random.permutation(N)
        X_shuf = X[idx]
        Y_shuf = Y[idx]

        Xtrn = X_shuf[:Ntrn]
        Ytrn = Y_shuf[:Ntrn]
        Xtst = X_shuf[Ntrn:]
        Ytst = Y_shuf[Ntrn:]
        Ntst = Xtst.shape[0]

        # Compute distances from each test sample to all train samples
        dists = np.linalg.norm(Xtst[:, None, :] - Xtrn[None, :, :], axis=2)
        nn_indices = np.argmin(dists, axis=1)
        predicted_classes = Ytrn[nn_indices]
        acerto = np.sum(predicted_classes == Ytst)
        Pacerto.append(100 * acerto / Ntst)

    TX_OK = np.array(Pacerto)
    STATS = np.array(
        [np.mean(TX_OK), np.std(TX_OK), np.median(TX_OK), np.min(TX_OK), np.max(TX_OK)]
    )
    return STATS, TX_OK, X


def classificador_maxcorr(
    D: np.ndarray,
    Nr: int,
    Ptrain: int,
    method: NormalizationMethods = NormalizationMethods.Z_SCORE,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Maximum correlation classifier.
    Assigns each sample to the class whose centroid has the highest Pearson correlation coefficient.
    Returns: (stats, accuracies, Xnorm)
    """
    X = D[:, :-1].copy()
    Y = D[:, -1].astype(int).copy()
    Nrep = Nr
    Ptrn = Ptrain / 100.0

    X = normalize(X, method)

    N = len(Y)
    C = np.max(Y)
    Ntrn = int(np.floor(Ptrn * N))

    Pacerto = []

    for r in range(Nrep):
        idx = np.random.permutation(N)
        X_shuf = X[idx]
        Y_shuf = Y[idx]

        Xtrn = X_shuf[:Ntrn]
        Ytrn = Y_shuf[:Ntrn]
        Xtst = X_shuf[Ntrn:]
        Ytst = Y_shuf[Ntrn:]
        Ntst = Xtst.shape[0]

        # Class centroids
        M = np.zeros((C, X.shape[1]))
        for k in range(1, C + 1):
            Ic = np.where(Ytrn == k)[0]
            Xc = Xtrn[Ic]
            M[k - 1] = np.mean(Xc, axis=0)

        # Pearson correlation for each sample to each class centroid
        Xc = Xtst - Xtst.mean(axis=1, keepdims=True)
        Mc = M - M.mean(axis=1, keepdims=True)
        numer = np.dot(Xc, Mc.T)
        denom = np.linalg.norm(Xc, axis=1, keepdims=True) * np.linalg.norm(Mc, axis=1)
        corr = numer / (denom + 1e-12)
        predicted_classes = np.argmax(corr, axis=1) + 1
        acerto = np.sum(predicted_classes == Ytst)
        Pacerto.append(100 * acerto / Ntst)

    TX_OK = np.array(Pacerto)
    STATS = np.array(
        [np.mean(TX_OK), np.std(TX_OK), np.median(TX_OK), np.min(TX_OK), np.max(TX_OK)]
    )
    return STATS, TX_OK, X


def classificador_dmc(
    D: np.ndarray,
    Nr: int,
    Ptrain: int,
    method: NormalizationMethods = NormalizationMethods.Z_SCORE,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Distance to class mean (centroid) classifier.
    Assigns each sample to the class whose centroid is closest in Euclidean distance.
    Returns: (stats, accuracies, Xnorm)
    """
    X = D[:, :-1].copy()
    Y = D[:, -1].astype(int).copy()
    Nrep = Nr
    Ptrn = Ptrain / 100.0

    X = normalize(X, method)

    N = len(Y)
    C = np.max(Y)
    Ntrn = int(np.floor(Ptrn * N))

    Pacerto = []

    for r in range(Nrep):
        idx = np.random.permutation(N)
        X_shuf = X[idx]
        Y_shuf = Y[idx]

        Xtrn = X_shuf[:Ntrn]
        Ytrn = Y_shuf[:Ntrn]
        Xtst = X_shuf[Ntrn:]
        Ytst = Y_shuf[Ntrn:]
        Ntst = Xtst.shape[0]

        # Class centroids
        M = np.zeros((C, X.shape[1]))
        for k in range(1, C + 1):
            Ic = np.where(Ytrn == k)[0]
            Xc = Xtrn[Ic]
            M[k - 1] = np.mean(Xc, axis=0)

        # Compute distances to centroids
        dists = np.linalg.norm(Xtst[:, None, :] - M[None, :, :], axis=2)
        predicted_classes = np.argmin(dists, axis=1) + 1
        acerto = np.sum(predicted_classes == Ytst)
        Pacerto.append(100 * acerto / Ntst)

    TX_OK = np.array(Pacerto)
    STATS = np.array(
        [np.mean(TX_OK), np.std(TX_OK), np.median(TX_OK), np.min(TX_OK), np.max(TX_OK)]
    )
    return STATS, TX_OK, X
