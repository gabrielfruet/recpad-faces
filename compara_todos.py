"""
Python equivalent of the Octave script for running face recognition classifiers.

- Loads 'recfaces.dat'
- Runs various classification methods (quadratico, variante1, variante2, variante3, variante4, linearMQ)
- Collects stats and timing
- Plots boxplot of accuracies

Assumes you have implemented the corresponding Python functions:
    - quadratico
    - variante1
    - variante2
    - variante3
    - variante4
    - linearMQ

You may need to implement or translate these from your existing Octave/MATLAB code!
"""

import numpy as np
import matplotlib.pyplot as plt
import logging
import time

log = logging.getLogger(__name__)

def main():
    # Load data
    logging.basicConfig(level=logging.INFO)
    D = np.loadtxt('recfaces.dat')

    # Nr = 50       # Number of repetitions
    Nr = 5       # Number of repetitions
    Ptrain = 80   # Percentage for training

    # Run classifiers and record timing
    start = time.perf_counter_ns()
    from quadratico import quadratico
    STATS_0, TX_OK0, X0, m0, S0, posto0 = quadratico(D, Nr, Ptrain)
    log.info("Quadratico classifier executed in %d ns", time.perf_counter_ns() - start)
    log.info("STATS_0: %s", STATS_0)
    log.info("TX_OK0: %s", TX_OK0)
    log.info("X0 shape: %s", X0.shape)
    log.info("m0 shape: %s", m0.shape)
    log.info("S0 shape: %s", S0.shape)
    log.info("posto0: %s", posto0)
    Tempo0 = time.perf_counter_ns() - start

    start = time.perf_counter_ns()
    from variante1 import variante1
    STATS_1, TX_OK1, X1, m1, S1, posto1 = variante1(D, Nr, Ptrain, 0.01)
    Tempo1 = time.perf_counter_ns() - start

    start = time.perf_counter_ns()
    from variante2 import variante2
    STATS_2, TX_OK2, X2, m2, S2, posto2 = variante2(D, Nr, Ptrain)
    Tempo2 = time.perf_counter_ns() - start

    start = time.perf_counter_ns()
    from variante3 import variante3
    STATS_3, TX_OK3, X3, m3, S3, posto3 = variante3(D, Nr, Ptrain, 0.5)
    Tempo3 = time.perf_counter_ns() - start

    start = time.perf_counter_ns()
    from variante4 import variante4
    STATS_4, TX_OK4, X4, m4, S4, posto4 = variante4(D, Nr, Ptrain)
    Tempo4 = time.perf_counter_ns() - start

    start = time.perf_counter_ns()
    from linearMQ import linearMQ
    STATS_5, TX_OK5, W = linearMQ(D, Nr, Ptrain)
    Tempo5 = time.perf_counter_ns() - start

    # Print STATS
    print("STATS_0:", STATS_0)
    print("STATS_1:", STATS_1)
    print("STATS_2:", STATS_2)
    print("STATS_3:", STATS_3)
    print("STATS_4:", STATS_4)
    print("STATS_5:", STATS_5)

    TEMPOS = [Tempo0, Tempo1, Tempo2, Tempo3, Tempo4, Tempo5]
    print("TEMPOS:", TEMPOS)

    # Boxplot of success rates
    plt.boxplot([TX_OK0, TX_OK1, TX_OK2, TX_OK3, TX_OK4, TX_OK5])
    plt.xticks([1, 2, 3, 4, 5, 6], ["Quadratico", "Variante 1", "Variante 2", "Variante 3", "Variante 4", "MQ"])
    plt.title('Conjunto Coluna')
    plt.xlabel('Classificador')
    plt.ylabel('Taxas de acerto')
    plt.show()


if __name__ == "__main__":
    main()
