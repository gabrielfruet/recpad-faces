"""
Python equivalent of the Octave script for running face recognition classifiers.

- Loads 'recfaces.dat'
- Runs various classification methods (quadratico, variante1, variante2, variante3, variante4, linearMQ)
- Collects stats and timing
- Displays results in a formatted table using the 'rich' library
- Plots boxplot of accuracies

Assumes you have implemented the corresponding Python functions:
    - quadratico
    - variante1
    - variante2
    - variante3
    - variante4
    - linearMQ

You may need to implement or translate these from your existing Octave/MATLAB code!
To run this, you will need to install the 'rich' library:
pip install rich
"""

import argparse
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import logging
import time
from rich.console import Console
from rich.table import Table
from rich.logging import RichHandler

# It's assumed that the classifier functions (quadratico, variante1, etc.)
# are in separate .py files and return the described values.
# Since these files are not provided, we'll create mock functions
# for demonstration purposes so the script can run.
#
# In your actual use, you should REMOVE these mock functions
# and use your own implementations.


def mock_classifier(D, Nr, Ptrain, *args):
    """Mock function to simulate a classifier's output."""
    # Simulate some accuracy results between 70% and 95%
    tx_ok = np.random.uniform(0.70, 0.95, size=Nr)
    # The other return values are mocked but not used in the final table/plot
    stats = [
        np.mean(tx_ok),
        np.min(tx_ok),
        np.max(tx_ok),
        np.median(tx_ok),
        np.std(tx_ok),
    ]
    x = np.array([])
    m = np.array([])
    s = np.array([])
    P_failed_inversions = np.random.uniform(
        0.01, 0.1, size=Nr
    )  # Mock failed inversions
    posto = 0
    return stats, tx_ok, x, m, s, posto, P_failed_inversions


def mock_linearMQ(D, Nr, Ptrain):
    """Mock function for linearMQ."""
    tx_ok = np.random.uniform(0.75, 0.98, size=Nr)
    stats = [
        np.mean(tx_ok),
        np.min(tx_ok),
        np.max(tx_ok),
        np.median(tx_ok),
        np.std(tx_ok),
    ]
    w = np.array([])
    return stats, tx_ok, w


# --- Replace these imports with your actual classifier modules ---
# quadratico = mock_classifier
# variante1 = mock_classifier
# variante2 = mock_classifier
# variante3 = mock_classifier
variante4 = mock_classifier
linearMQ = mock_linearMQ
from quadratico import quadratico
from variante1 import variante1
from variante2 import variante2
from variante3 import variante3
from variante4 import variante4
# ... and so on
# ----------------------------------------------------------------

log = logging.getLogger(__name__)


def main():
    """
    Main function to run the classification and display results.
    """
    # Initialize Rich Console for beautiful printing
    console = Console()

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        type=str.upper,
        help="Set logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)",
    )

    parser.add_argument(
        "--data",
        "-d",
        default="recfaces.dat",
        type=Path,
        help="Path to the dataset file (default: recfaces.dat)",
    )

    args = parser.parse_args()

    # --- 1. Load Data ---
    logging.basicConfig(
        level=args.level, handlers=[RichHandler()], format="%(message)s"
    )
    try:
        # Create a dummy 'recfaces.dat' if it doesn't exist for demonstration
        if not args.data.exists():
            log.error(f"File '{args.data}' not found")
        D = np.loadtxt(args.data)
        console.print("[green]Loaded 'recfaces.dat' successfully.[/green]")

        # --- 2. Set Parameters ---
        # Nr = 50  # Number of repetitions
        Nr = 5  # Use a smaller number for quick tests
        Ptrain = 80  # Percentage for training

        # --- 3. Run Classifiers and Collect Results ---
        console.print(f"\nRunning {Nr} repetitions for each classifier...")

        # A list to hold all results for easier processing
        all_results = []

        classifiers_to_run = {
            "Quadrático": (quadratico, []),
            "Variante 1": (variante1, [0.01]),
            "Variante 2": (variante2, []),
            "Variante 3": (variante3, [0.0]),
            "Variante 4": (variante4, []),
            "Linear MQ": (linearMQ, []),
        }

        for name, (func, args) in classifiers_to_run.items():
            start_time = time.perf_counter()
            if name == "Linear MQ":
                _, tx_ok, _ = func(D, Nr, Ptrain, *args)
            else:
                _, tx_ok, _, _, _, _, P_failed_inversions = func(D, Nr, Ptrain, *args)
            end_time = time.perf_counter()
            exec_time = end_time - start_time
            all_results.append(
                {
                    "name": name,
                    "tx_ok": tx_ok,
                    "time": exec_time,
                    "failed_inv": P_failed_inversions,
                }
            )
            log.info(f"Classifier '{name}' executed in {exec_time:.4f} s")

        # --- 4. Display Results in a Table ---
        console.print("\n[bold cyan]Classifier Performance Summary[/bold cyan]")

        table = Table(
            show_header=True,
            header_style="bold magenta",
            title="Resultados dos Classificadores",
        )
        table.add_column("Classificador", style="cyan", no_wrap=True)
        table.add_column("Média", justify="right")
        table.add_column("Mínimo", justify="right")
        table.add_column("Máximo", justify="right")
        table.add_column("Mediana", justify="right")
        table.add_column("Desvio Padrão", justify="right", style="green")
        table.add_column("Tempo (s)", justify="right", style="yellow")
        table.add_column(
            "Porcentagem de inversões que falharam", justify="right", style="red"
        )

        for result in all_results:
            stats = result["tx_ok"]
            failed_inv = result.get("failed_inv", [])
            table.add_row(
                result["name"],
                f"{np.mean(stats):.4f}",
                f"{np.min(stats):.4f}",
                f"{np.max(stats):.4f}",
                f"{np.median(stats):.4f}",
                f"{np.std(stats):.4f}",
                f"{result['time']:.4f}",
                f"{np.mean(failed_inv) if len(failed_inv) > 0 else 'N/A':.4f}",
            )

        console.print(table)

        # --- 5. Generate Boxplot of Success Rates ---
        console.print("\nGenerating boxplot...")
        plt.style.use("seaborn-v0_8-whitegrid")
        fig, ax = plt.subplots(figsize=(10, 6))

        tx_ok_data = [res["tx_ok"] for res in all_results]
        classifier_names = [res["name"] for res in all_results]

        ax.boxplot(tx_ok_data)
        ax.set_xticklabels(classifier_names, rotation=45, ha="right")
        ax.set_title("Comparação das Taxas de Acerto dos Classificadores", fontsize=16)
        ax.set_xlabel("Classificador", fontsize=12)
        ax.set_ylabel("Taxa de Acerto (%)", fontsize=12)
        ax.yaxis.grid(True)
        plt.tight_layout()
        plt.show()
        console.print("[green]Done.[/green]")

    except Exception as e:
        console.print(f"[bold red]An error occurred: {e}[/bold red]")
        raise Exception("Error") from e


if __name__ == "__main__":
    main()
