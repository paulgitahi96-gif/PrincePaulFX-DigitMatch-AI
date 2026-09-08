"""
============================================================
PRINCE PAUL FX
DIGITMATCH AI ANALYZER
TRAINING PIPELINE
VERSION 1.0
============================================================

Purpose:
    Complete historical-data training pipeline.

Pipeline:

    Deriv Historical Ticks
            ↓
       Last Digits
            ↓
       Feature Matrix
            ↓
    Chronological Split
            ↓
       LightGBM
            +
        XGBoost
            ↓
    Out-of-Sample Evaluation
            ↓
      Save Models
            ↓
     Training Report

Important:
    This system produces probabilistic digit candidates.

    It does NOT guarantee:
        - a correct next digit
        - a specific win rate
        - profitable trading

    Evaluation is chronological to reduce future-data leakage.

============================================================
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from deriv.ticks import (
    DerivHistoricalTicks,
    save_ticks_csv,
    ticks_to_digits,
)

from engine.features import (
    build_training_dataset,
    split_features_target,
    feature_columns,
)

from engine.lightgbm_engine import (
    LightGBMDigitEngine,
)

from engine.xgboost_engine import (
    XGBoostDigitEngine,
)


# ============================================================
# CONFIGURATION
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent

DATA_DIR = PROJECT_ROOT / "data"
MODEL_DIR = PROJECT_ROOT / "models"

HISTORICAL_DATA_FILE = (
    DATA_DIR / "historical_ticks.csv"
)

TRAINING_REPORT_FILE = (
    MODEL_DIR / "training_report.json"
)

DEFAULT_MARKET = "1HZ100V"

DEFAULT_TICK_COUNT = 5000

VALIDATION_FRACTION = 0.20

MIN_TRAINING_ROWS = 100

NUM_DIGITS = 10


# ============================================================
# DIRECTORY SETUP
# ============================================================

DATA_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

MODEL_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


# ============================================================
# LOGGING HELPERS
# ============================================================

def print_header(title: str) -> None:

    print()
    print("=" * 64)
    print(title)
    print("=" * 64)


def print_section(title: str) -> None:

    print()
    print("-" * 64)
    print(title)
    print("-" * 64)


# ============================================================
# DATA DOWNLOAD
# ============================================================

def download_historical_data(
    market: str,
    count: int,
) -> list:

    print_section(
        "STEP 1 — DOWNLOADING HISTORICAL TICKS"
    )

    print(
        f"Market: {market}"
    )

    print(
        f"Requested ticks: {count}"
    )

    client = DerivHistoricalTicks(
        market=market
    )

    ticks = client.fetch_sync(
        count=count
    )

    if not ticks:
        raise RuntimeError(
            "Deriv returned no historical ticks."
        )

    print(
        f"Received ticks: {len(ticks)}"
    )

    save_ticks_csv(
        ticks,
        HISTORICAL_DATA_FILE,
    )

    print(
        f"Saved dataset: "
        f"{HISTORICAL_DATA_FILE}"
    )

    return ticks


# ============================================================
# DIGIT VALIDATION
# ============================================================

def validate_digits(
    digits: list[int],
) -> None:

    print_section(
        "STEP 2 — VALIDATING DIGIT DATA"
    )

    if len(digits) < MIN_TRAINING_ROWS:
        raise ValueError(
            "Not enough historical digits for training. "
            f"Received {len(digits)}."
        )

    invalid = [
        digit
        for digit in digits
        if not isinstance(digit, int)
        or digit < 0
        or digit > 9
    ]

    if invalid:
        raise ValueError(
            "Invalid digit values detected."
        )

    distribution = pd.Series(
        digits
    ).value_counts().sort_index()

    print(
        f"Total valid digits: {len(digits)}"
    )

    print()
    print("Digit distribution:")

    for digit in range(NUM_DIGITS):

        count = int(
            distribution.get(
                digit,
                0,
            )
        )

        percentage = (
            count / len(digits) * 100
        )

        print(
            f"Digit {digit}: "
            f"{count:6d} "
            f"({percentage:6.2f}%)"
        )


# ============================================================
# FEATURE DATASET
# ============================================================

def create_training_dataset(
    digits: list[int],
) -> tuple[pd.DataFrame, pd.Series]:

    print_section(
        "STEP 3 — BUILDING FEATURE MATRIX"
    )

    dataset = build_training_dataset(
        digits
    )

    if dataset.empty:
        raise RuntimeError(
            "Feature dataset is empty."
        )

    X, y = split_features_target(
        dataset
    )

    columns = feature_columns(
        X
    )

    X = X[columns]

    print(
        f"Training examples: {len(X)}"
    )

    print(
        f"Feature columns: {len(columns)}"
    )

    print(
        f"Target classes: "
        f"{sorted(y.unique().tolist())}"
    )

    if len(X) < MIN_TRAINING_ROWS:
        raise ValueError(
            "Feature dataset is too small."
        )

    return X, y


# ============================================================
# CHRONOLOGICAL SPLIT
# ============================================================

def chronological_split(
    X: pd.DataFrame,
    y: pd.Series,
    validation_fraction: float = VALIDATION_FRACTION,
):
    """
    Preserve chronological ordering.

    Earlier observations:
        TRAINING

    Later observations:
        VALIDATION

    No random shuffle.
    """

    if not (
        0.05
        <= validation_fraction
        <= 0.40
    ):
        raise ValueError(
            "Validation fraction must be "
            "between 0.05 and 0.40."
        )

    total_rows = len(X)

    split_index = int(
        total_rows
        * (1.0 - validation_fraction)
    )

    if split_index <= 0:
        raise ValueError(
            "Invalid chronological split."
        )

    X_train = X.iloc[
        :split_index
    ].copy()

    X_valid = X.iloc[
        split_index:
    ].copy()

    y_train = y.iloc[
        :split_index
    ].copy()

    y_valid = y.iloc[
        split_index:
    ].copy()

    return (
        X_train,
        X_valid,
        y_train,
        y_valid,
    )


# ============================================================
# CLASS COVERAGE
# ============================================================

def validate_training_classes(
    y_train: pd.Series,
) -> None:

    print_section(
        "STEP 4 — CHECKING TRAINING CLASS COVERAGE"
    )

    classes = sorted(
        set(
            int(value)
            for value in y_train.tolist()
        )
    )

    missing = [
        digit
        for digit in range(NUM_DIGITS)
        if digit not in classes
    ]

    print(
        f"Classes present in training data: "
        f"{classes}"
    )

    if missing:

        raise ValueError(
            "The chronological training portion "
            "does not contain all 10 digit classes. "
            f"Missing classes: {missing}. "
            "Collect more historical ticks before training."
        )

    print(
        "All 10 digit classes are present."
    )


# ============================================================
# CLASS DISTRIBUTION
# ============================================================

def class_distribution(
    y: pd.Series,
) -> dict[str, int]:

    counts = (
        y.value_counts()
        .sort_index()
    )

    result = {}

    for digit in range(NUM_DIGITS):

        result[str(digit)] = int(
            counts.get(
                digit,
                0,
            )
        )

    return result


# ============================================================
# LIGHTGBM TRAINING
# ============================================================

def train_lightgbm(
    digits: list[int],
) -> dict:

    print_section(
        "STEP 5 — TRAINING LIGHTGBM"
    )

    engine = LightGBMDigitEngine(
        model_dir=MODEL_DIR
    )

    result = engine.train(
        digits,
        validation_fraction=(
            VALIDATION_FRACTION
        ),
    )

    model_path = engine.save()

    print(
        "LightGBM training complete."
    )

    print(
        f"Training rows: "
        f"{result['training_rows']}"
    )

    print(
        f"Validation rows: "
        f"{result['validation_rows']}"
    )

    print(
        f"Validation accuracy: "
        f"{result['accuracy']:.4f}"
    )

    print(
        f"Validation logloss: "
        f"{result['logloss']:.6f}"
    )

    print(
        f"Best iteration: "
        f"{result['best_iteration']}"
    )

    print(
        f"Model saved: {model_path}"
    )

    return {
        "engine": "LightGBM",
        "result": result,
        "model_path": str(
            model_path
        ),
    }


# ============================================================
# XGBOOST TRAINING
# ============================================================

def train_xgboost(
    digits: list[int],
) -> dict:

    print_section(
        "STEP 6 — TRAINING XGBOOST"
    )

    engine = XGBoostDigitEngine(
        model_dir=MODEL_DIR
    )

    result = engine.train(
        digits,
        validation_fraction=(
            VALIDATION_FRACTION
        ),
    )

    model_path = engine.save()

    print(
        "XGBoost training complete."
    )

    print(
        f"Training rows: "
        f"{result['training_rows']}"
    )

    print(
        f"Validation rows: "
        f"{result['validation_rows']}"
    )

    print(
        f"Validation accuracy: "
        f"{result['accuracy']:.4f}"
    )

    print(
        f"Validation logloss: "
        f"{result['logloss']:.6f}"
    )

    print(
        f"Best iteration: "
        f"{result['best_iteration']}"
    )

    print(
        f"Model saved: {model_path}"
    )

    return {
        "engine": "XGBoost",
        "result": result,
        "model_path": str(
            model_path
        ),
    }


# ============================================================
# BASELINE
# ============================================================

def calculate_uniform_baseline(
    y_valid: pd.Series,
) -> float:

    """
    A 10-class uniform/random baseline is approximately 10%
    when every digit is equally likely.
    """

    if len(y_valid) == 0:
        return 0.0

    return 1.0 / NUM_DIGITS


# ============================================================
# TRAINING REPORT
# ============================================================

def create_training_report(
    market: str,
    requested_ticks: int,
    received_ticks: int,
    digits: list[int],
    X: pd.DataFrame,
    y: pd.Series,
    y_train: pd.Series,
    y_valid: pd.Series,
    lightgbm_result: dict,
    xgboost_result: dict,
) -> dict:

    report = {
        "project": "PRINCE PAUL FX",
        "system": "DigitMatch AI Analyzer",
        "version": "1.0",
        "market": market,
        "requested_ticks": int(
            requested_ticks
        ),
        "received_ticks": int(
            received_ticks
        ),
        "usable_digits": int(
            len(digits)
        ),
        "feature_rows": int(
            len(X)
        ),
        "feature_count": int(
            len(X.columns)
        ),
        "training_rows": int(
            len(y_train)
        ),
        "validation_rows": int(
            len(y_valid)
        ),
        "validation_method": (
            "chronological"
        ),
        "validation_fraction": (
            VALIDATION_FRACTION
        ),
        "baseline_accuracy": (
            calculate_uniform_baseline(
                y_valid
            )
        ),
        "target_class_distribution": (
            class_distribution(y)
        ),
        "training_class_distribution": (
            class_distribution(
                y_train
            )
        ),
        "validation_class_distribution": (
            class_distribution(
                y_valid
            )
        ),
        "lightgbm": lightgbm_result,
        "xgboost": xgboost_result,
    }

    return report


def save_training_report(
    report: dict,
) -> Path:

    TRAINING_REPORT_FILE.write_text(
        json.dumps(
            report,
            indent=2,
        ),
        encoding="utf-8",
    )

    return TRAINING_REPORT_FILE


# ============================================================
# MAIN TRAINING PIPELINE
# ============================================================

def run_training(
    market: str = DEFAULT_MARKET,
    tick_count: int = DEFAULT_TICK_COUNT,
) -> dict:

    print_header(
        "PRINCE PAUL FX — DIGITMATCH AI TRAINING"
    )

    print(
        f"Market: {market}"
    )

    print(
        f"Historical ticks requested: "
        f"{tick_count}"
    )

    # --------------------------------------------------------
    # Download
    # --------------------------------------------------------

    ticks = download_historical_data(
        market=market,
        count=tick_count,
    )

    # --------------------------------------------------------
    # Convert to digits
    # --------------------------------------------------------

    digits = ticks_to_digits(
        ticks
    )

    # --------------------------------------------------------
    # Validate
    # --------------------------------------------------------

    validate_digits(
        digits
    )

    # --------------------------------------------------------
    # Build features
    # --------------------------------------------------------

    X, y = create_training_dataset(
        digits
    )

    # --------------------------------------------------------
    # Chronological split
    # --------------------------------------------------------

    print_section(
        "STEP 4B — CHRONOLOGICAL TRAIN/VALIDATION SPLIT"
    )

    (
        X_train,
        X_valid,
        y_train,
        y_valid,
    ) = chronological_split(
        X,
        y,
    )

    print(
        f"Training rows: "
        f"{len(X_train)}"
    )

    print(
        f"Validation rows: "
        f"{len(X_valid)}"
    )

    validate_training_classes(
        y_train
    )

    # --------------------------------------------------------
    # Train LightGBM
    # --------------------------------------------------------

    lightgbm_result = train_lightgbm(
        digits
    )

    # --------------------------------------------------------
    # Train XGBoost
    # --------------------------------------------------------

    xgboost_result = train_xgboost(
        digits
    )

    # --------------------------------------------------------
    # Report
    # --------------------------------------------------------

    print_section(
        "STEP 7 — CREATING TRAINING REPORT"
    )

    report = create_training_report(
        market=market,
        requested_ticks=tick_count,
        received_ticks=len(ticks),
        digits=digits,
        X=X,
        y=y,
        y_train=y_train,
        y_valid=y_valid,
        lightgbm_result=lightgbm_result,
        xgboost_result=xgboost_result,
    )

    report_path = save_training_report(
        report
    )

    print(
        f"Training report saved: "
        f"{report_path}"
    )

    # --------------------------------------------------------
    # Summary
    # --------------------------------------------------------

    print_header(
        "TRAINING COMPLETE"
    )

    print(
        f"Market: {market}"
    )

    print(
        f"Ticks: {len(ticks)}"
    )

    print(
        f"Feature rows: {len(X)}"
    )

    print(
        f"Features: {len(X.columns)}"
    )

    print()

    print(
        "LightGBM accuracy: "
        f"{lightgbm_result['result']['accuracy']:.4f}"
    )

    print(
        "XGBoost accuracy: "
        f"{xgboost_result['result']['accuracy']:.4f}"
    )

    print(
        "Uniform 10-class baseline: "
        f"{calculate_uniform_baseline(y_valid):.4f}"
    )

    print()

    print(
        "Models are ready for the prediction layer."
    )

    return report


# ============================================================
# CLI
# ============================================================

def parse_arguments():

    import argparse

    parser = argparse.ArgumentParser(
        description=(
            "PRINCE PAUL FX "
            "DigitMatch AI training pipeline"
        )
    )

    parser.add_argument(
        "--market",
        default=DEFAULT_MARKET,
        help=(
            "Deriv market symbol. "
            f"Default: {DEFAULT_MARKET}"
        ),
    )

    parser.add_argument(
        "--ticks",
        type=int,
        default=DEFAULT_TICK_COUNT,
        help=(
            "Number of historical ticks "
            f"to download. "
            f"Default: {DEFAULT_TICK_COUNT}"
        ),
    )

    return parser.parse_args()


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":

    args = parse_arguments()

    try:

        run_training(
            market=args.market,
            tick_count=args.ticks,
        )

    except KeyboardInterrupt:

        print()
        print(
            "Training interrupted by user."
        )

        sys.exit(1)

    except Exception as error:

        print()
        print("=" * 64)
        print("TRAINING FAILED")
        print("=" * 64)

        print(
            f"Error: {error}"
        )

        sys.exit(1)
