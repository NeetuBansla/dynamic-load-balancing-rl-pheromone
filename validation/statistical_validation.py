# ============================================================
# 5-RUN STATISTICAL VALIDATION
# ============================================================

from pathlib import Path
import importlib.util
import os
import sys
import numpy as np
import pandas as pd
from scipy import stats


# ------------------------------------------------------------
# SETTINGS
# ------------------------------------------------------------

MAIN_FILE = Path(__file__).resolve().parent.parent / "main.py"

SEEDS = [42, 43, 44, 45, 46]

RESULT_FILE = Path(__file__).with_name("validation_results.csv")
SUMMARY_FILE = Path(__file__).with_name("validation_summary.csv")


if not MAIN_FILE.exists():
    raise FileNotFoundError(
        f"Cannot find main.py here:\n{MAIN_FILE}\n"
        "Keep main.py and statistical_validation.py in the same folder."
    )


# ------------------------------------------------------------
# RUN MAIN.PY FIVE TIMES
# ------------------------------------------------------------

results = []

print("=" * 70)
print("5-RUN INDEPENDENT STATISTICAL VALIDATION")
print("=" * 70)


for run_number, seed in enumerate(SEEDS, start=1):

    print("\n" + "=" * 70)
    print(f"RUN {run_number}/5 - SEED {seed}")
    print("=" * 70)

    # Give the seed to main.py
    os.environ["RUN_SEED"] = str(seed)

    # Unique module name forces a fresh execution of main.py
    module_name = f"main_validation_seed_{seed}"

    spec = importlib.util.spec_from_file_location(
        module_name,
        MAIN_FILE
    )

    main = importlib.util.module_from_spec(spec)

    # Execute main.py from the beginning:
    # dataset -> split -> fresh model -> training -> evaluation
    spec.loader.exec_module(main)

    # Collect classifier results
    result = {
        "seed": seed,
        "accuracy": float(main.accuracy),
        "precision": float(main.precision),
        "recall": float(main.recall),
        "f1": float(main.f1),
        "training_time_sec": float(main.training_time)
    }

    results.append(result)

    # Print result immediately
    print("\n----------------------------------------")
    print(f"RESULT FOR SEED {seed}")
    print("----------------------------------------")
    print(f"Accuracy : {main.accuracy:.4f}%")
    print(f"Precision: {main.precision:.4f}%")
    print(f"Recall   : {main.recall:.4f}%")
    print(f"F1 Score : {main.f1:.4f}%")
    print(f"Training : {main.training_time:.4f} sec")

    # Save after every run
    pd.DataFrame(results).to_csv(
        RESULT_FILE,
        index=False
    )

    # Remove module reference
    if module_name in sys.modules:
        del sys.modules[module_name]


# ------------------------------------------------------------
# ALL FIVE RUNS
# ------------------------------------------------------------

results_df = pd.DataFrame(results)

print("\n")
print("=" * 70)
print("ALL 5 RUN RESULTS")
print("=" * 70)

print(
    results_df[
        ["seed", "accuracy", "precision", "recall", "f1"]
    ].to_string(index=False)
)


# ------------------------------------------------------------
# CALCULATE MEAN, SD AND 95% CI
# ------------------------------------------------------------

metrics = [
    "accuracy",
    "precision",
    "recall",
    "f1"
]

summary = []


print("\n")
print("=" * 70)
print("FINAL STATISTICAL RESULTS")
print("=" * 70)


for metric in metrics:

    values = results_df[metric].to_numpy(dtype=float)

    n = len(values)

    mean_value = np.mean(values)

    # Sample standard deviation
    sd_value = np.std(values, ddof=1)

    # Standard error
    sem_value = stats.sem(values)

    # 95% confidence interval
    if sd_value == 0:
        ci_low = mean_value
        ci_high = mean_value
    else:
        ci_low, ci_high = stats.t.interval(
            confidence=0.95,
            df=n - 1,
            loc=mean_value,
            scale=sem_value
        )

    summary.append({
        "Metric": metric,
        "N": n,
        "Mean": mean_value,
        "SD": sd_value,
        "CI_95_Lower": ci_low,
        "CI_95_Upper": ci_high,
        "Mean_PlusMinus_SD":
            f"{mean_value:.4f} ± {sd_value:.4f}"
    })

    print(
        f"{metric.capitalize():10s}: "
        f"{mean_value:.4f} ± {sd_value:.4f}"
    )


# ------------------------------------------------------------
# SAVE SUMMARY
# ------------------------------------------------------------

summary_df = pd.DataFrame(summary)

summary_df.to_csv(
    SUMMARY_FILE,
    index=False
)


print("\n")
print("=" * 70)
print("95% CONFIDENCE INTERVALS")
print("=" * 70)

for _, row in summary_df.iterrows():

    print(
        f"{row['Metric'].capitalize():10s}: "
        f"[{row['CI_95_Lower']:.4f}, "
        f"{row['CI_95_Upper']:.4f}]"
    )


print("\nFiles created:")
print(f"1. {RESULT_FILE}")
print(f"2. {SUMMARY_FILE}")

print("\nValidation finished successfully.")