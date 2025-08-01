from quadratico import discriminant, estatisticas_classes
import time
from typing import List, Tuple
import numpy as np
import logging

log = logging.getLogger(__name__)


def variante2(
    D: np.ndarray,
    Nr: int,
    Ptrain: int,
) -> Tuple[
    np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray
]:
    """
    Quadratico variante 2 classifier implementation in Python with optimized testing phase
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

        C_pooled = np.sum(fw_i[:, np.newaxis, np.newaxis] * S_k, axis=0)

        # Testing set
        Xtst = X_shuf[Ntrn:]
        Ytst = Y_shuf[Ntrn:]
        Ntst = Xtst.shape[0]

        # ============== OPTIMIZED TESTING PHASE ==============
        tic = time.perf_counter_ns()

        # Pre-compute inverse covariance matrices
        failed_inversions = 0
        try:
            # regularization
            inv_C_pooled = np.linalg.inv(C_pooled)
            log.debug(
                f"Max value of inverse pooled covariance matrix: {np.max(inv_C_pooled)}"
            )
        except np.linalg.LinAlgError:
            failed_inversions += 1
            log.debug(
                f"Pooled covariance matrix inversion failed, using pseudo-inverse instead"
            )
            inv_C_pooled = np.linalg.pinv(C_pooled)

        log.info(f"Percentage of failed inversions: {100 * failed_inversions}%")

        # Calculate all distances at once
        distances = discriminant(Xtst, M, [inv_C_pooled] * C)

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
