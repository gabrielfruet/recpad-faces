import numpy as np
import logging
from typing import List, Tuple
import time

log = logging.getLogger(__name__)


def estatisticas_classes(Xtrn, Ytrn, C):
    F = Xtrn.shape[1]  # Number of features
    M = np.zeros((C, F))
    S_k = np.zeros((C, F, F))
    posto_k = np.zeros(C)
    for k in range(1, C + 1):
        Ic = np.where(Ytrn == k)[0]
        Xc = Xtrn[Ic]
        mu_k = np.mean(Xc, axis=0)
        cov_k = np.cov(Xc.T)
        rank_k = np.linalg.matrix_rank(cov_k)
        M[k - 1] = mu_k
        S_k[k - 1] = cov_k
        posto_k[k - 1] = rank_k

    return M, S_k, posto_k


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
        distances[:, k] = mahal_term
        if f_wi is not None:
            distances[:, k] -= 2 * np.log(f_wi[k] + 1e-2)

    return distances


def quadratico(
    D: np.ndarray, Nr: int, Ptrain: int
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
    X = D[:, :-1].copy()
    Y = D[:, -1].astype(int).copy()
    Nrep = Nr
    Ptrn = Ptrain / 100.0

    # Z-score normalization
    med = np.mean(X, axis=0, keepdims=True)
    dp = np.std(X, axis=0, keepdims=True)
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
        M, S_k, posto_k = estatisticas_classes(Xtrn, Ytrn, C)
        toc = time.perf_counter_ns()
        log.info(
            f"Time for calculating centroids and covariance matrices: {(toc - tic) / 1e6:.2f} ms"
        )

        m.append(M)
        S.append(S_k)
        posto.append(posto_k)

        count_result = np.unique_counts(Ytrn)
        sample_per_class = count_result.counts

        fw_i = sample_per_class / (sample_per_class.sum() - 1)

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
                inv_covs[k] = np.linalg.inv(S_k[k])
            except np.linalg.LinAlgError:
                failed_inversions += 1
                log.debug(
                    f"Covariance matrix for class {k + 1} is singular, using pseudo-inverse."
                )
                inv_covs[k] = np.linalg.pinv(S_k[k])

        log.info(f"Percentage of failed inversions: {100 * failed_inversions / C}%")

        # Calculate all distances at once
        distances = discriminant(Xtst, M, inv_covs)

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
    return STATS, TX_OK, X, m, S, posto, P_failed_inversions
