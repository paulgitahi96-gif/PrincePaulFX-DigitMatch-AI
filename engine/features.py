"""
============================================================
PRINCE PAUL FX
DIGITMATCH AI ANALYZER
FEATURE ENGINE
VERSION 1.0
============================================================

Purpose:
    Convert incoming Deriv tick digits into machine-learning
    features for LightGBM and XGBoost.

Input:
    Rolling sequence of last digits.

Output:
    Feature row describing the current market state.

Important:
    The target is NOT created here.

    The prediction target is:

        NEXT_DIGIT = digit on the next tick

This keeps feature generation separated from model training.

============================================================
"""

from __future__ import annotations

from collections import Counter
from typing import Iterable, Optional

import numpy as np
import pandas as pd


# ==========================================================
# CONFIGURATION
# ==========================================================

MIN_HISTORY = 20
MAX_LAG = 20

ROLLING_WINDOWS = (
    5,
    10,
    20,
    50,
)


# ==========================================================
# BASIC VALIDATION
# ==========================================================

def clean_digits(
    digits: Iterable[int],
) -> list[int]:
    """
    Validate and normalize a digit sequence.

    Only digits 0-9 are accepted.
    """

    cleaned = []

    for value in digits:

        try:
            digit = int(value)
        except (TypeError, ValueError):
            continue

        if 0 <= digit <= 9:
            cleaned.append(digit)

    return cleaned


# ==========================================================
# FREQUENCY FEATURES
# ==========================================================

def frequency_features(
    digits: list[int],
    window: int,
) -> dict[str, float]:
    """
    Calculate frequency of every digit inside a rolling window.
    """

    values = digits[-window:]

    total = len(values)

    features = {}

    if total == 0:

        for digit in range(10):
            features[f"freq_{window}_d{digit}"] = 0.0

        return features

    counts = Counter(values)

    for digit in range(10):

        count = counts.get(digit, 0)

        features[
            f"freq_{window}_d{digit}"
        ] = count / total

    return features


# ==========================================================
# GAP FEATURES
# ==========================================================

def digit_gap(
    digits: list[int],
    target_digit: int,
) -> int:
    """
    Number of ticks since target_digit last appeared.

    If the digit does not exist in the supplied history,
    the gap equals the history length.
    """

    for index in range(
        len(digits) - 1,
        -1,
        -1,
    ):

        if digits[index] == target_digit:

            return (
                len(digits)
                - 1
                - index
            )

    return len(digits)


def gap_features(
    digits: list[int],
) -> dict[str, float]:
    """
    Create gap features for digits 0-9.
    """

    features = {}

    for digit in range(10):

        gap = digit_gap(
            digits,
            digit,
        )

        features[
            f"gap_d{digit}"
        ] = float(gap)

    return features


# ==========================================================
# STREAK FEATURES
# ==========================================================

def current_streak(
    digits: list[int],
) -> int:
    """
    Length of the current identical-digit streak.
    """

    if not digits:
        return 0

    latest = digits[-1]

    streak = 0

    for digit in reversed(digits):

        if digit != latest:
            break

        streak += 1

    return streak


def streak_features(
    digits: list[int],
) -> dict[str, float]:
    """
    Current streak-related features.
    """

    if not digits:

        return {
            "current_streak": 0.0,
            "last_digit": -1.0,
        }

    last_digit = digits[-1]

    return {
        "current_streak": float(
            current_streak(digits)
        ),
        "last_digit": float(
            last_digit
        ),
    }


# ==========================================================
# TRANSITION FEATURES
# ==========================================================

def transition_features(
    digits: list[int],
) -> dict[str, float]:
    """
    Encode the transition from the previous digit
    to the current digit.

    Example:

        4 -> 7

    creates:

        transition_4_7 = 1
    """

    features = {}

    for previous in range(10):

        for current in range(10):

            features[
                f"transition_{previous}_{current}"
            ] = 0.0

    if len(digits) < 2:
        return features

    previous = digits[-2]
    current = digits[-1]

    features[
        f"transition_{previous}_{current}"
    ] = 1.0

    return features


# ==========================================================
# RECENCY FEATURES
# ==========================================================

def recency_features(
    digits: list[int],
) -> dict[str, float]:
    """
    Encode the most recent digits as categorical/numeric
    lag features.

    lag_1 = most recent digit
    lag_2 = previous digit
    etc.
    """

    features = {}

    for lag in range(
        1,
        MAX_LAG + 1,
    ):

        if len(digits) >= lag:

            value = digits[-lag]

        else:

            value = -1

        features[
            f"lag_{lag}"
        ] = float(value)

    return features


# ==========================================================
# DIGIT STATISTICS
# ==========================================================

def statistical_features(
    digits: list[int],
) -> dict[str, float]:
    """
    General statistics for the recent digit sequence.
    """

    if not digits:

        return {
            "mean_digit": 0.0,
            "std_digit": 0.0,
            "min_digit": 0.0,
            "max_digit": 0.0,
            "unique_digits": 0.0,
        }

    values = np.asarray(
        digits,
        dtype=float,
    )

    return {
        "mean_digit": float(
            np.mean(values)
        ),
        "std_digit": float(
            np.std(values)
        ),
        "min_digit": float(
            np.min(values)
        ),
        "max_digit": float(
            np.max(values)
        ),
        "unique_digits": float(
            len(set(digits))
        ),
    }


# ==========================================================
# PARITY FEATURES
# ==========================================================

def parity_features(
    digits: list[int],
) -> dict[str, float]:
    """
    Recent odd/even statistics.
    """

    if not digits:

        return {
            "odd_ratio_5": 0.0,
            "odd_ratio_10": 0.0,
            "odd_ratio_20": 0.0,
        }

    features = {}

    for window in (
        5,
        10,
        20,
    ):

        values = digits[-window:]

        if not values:

            ratio = 0.0

        else:

            odd_count = sum(
                digit % 2 == 1
                for digit in values
            )

            ratio = (
                odd_count
                / len(values)
            )

        features[
            f"odd_ratio_{window}"
        ] = float(ratio)

    return features


# ==========================================================
# BUILD ONE FEATURE ROW
# ==========================================================

def build_feature_row(
    digits: Iterable[int],
) -> Optional[dict[str, float]]:
    """
    Build one ML feature row from the current history.

    Returns None if there is insufficient history.
    """

    cleaned = clean_digits(digits)

    if len(cleaned) < MIN_HISTORY:
        return None

    features = {}

    # ------------------------------------------------------
    # Rolling frequency
    # ------------------------------------------------------

    for window in ROLLING_WINDOWS:

        features.update(
            frequency_features(
                cleaned,
                window,
            )
        )

    # ------------------------------------------------------
    # Gap analysis
    # ------------------------------------------------------

    features.update(
        gap_features(cleaned)
    )

    # ------------------------------------------------------
    # Streak
    # ------------------------------------------------------

    features.update(
        streak_features(cleaned)
    )

    # ------------------------------------------------------
    # Previous digit transitions
    # ------------------------------------------------------

    features.update(
        transition_features(cleaned)
    )

    # ------------------------------------------------------
    # Recent digit lags
    # ------------------------------------------------------

    features.update(
        recency_features(cleaned)
    )

    # ------------------------------------------------------
    # Statistics
    # ------------------------------------------------------

    features.update(
        statistical_features(cleaned)
    )

    # ------------------------------------------------------
    # Odd / Even
    # ------------------------------------------------------

    features.update(
        parity_features(cleaned)
    )

    return features


# ==========================================================
# BUILD FEATURE DATASET
# ==========================================================

def build_training_dataset(
    digits: Iterable[int],
) -> pd.DataFrame:
    """
    Build a chronological supervised-learning dataset.

    For each historical position:

        features = information available BEFORE next tick

        target = NEXT digit

    Example:

        History:
            1 4 7 2 8

        Feature row:
            1 4 7 2 8

        Target:
            next digit

    No future information is intentionally included.
    """

    cleaned = clean_digits(digits)

    rows = []

    if len(cleaned) <= MIN_HISTORY:
        return pd.DataFrame()

    for position in range(
        MIN_HISTORY,
        len(cleaned),
    ):

        history = cleaned[
            :position
        ]

        target = cleaned[
            position
        ]

        features = build_feature_row(
            history
        )

        if features is None:
            continue

        features["target"] = int(
            target
        )

        rows.append(features)

    if not rows:
        return pd.DataFrame()

    dataframe = pd.DataFrame(
        rows
    )

    return dataframe


# ==========================================================
# SPLIT FEATURES AND TARGET
# ==========================================================

def split_features_target(
    dataframe: pd.DataFrame,
):
    """
    Separate X features from y target.
    """

    if dataframe.empty:

        return (
            pd.DataFrame(),
            pd.Series(
                dtype=int
            ),
        )

    if "target" not in dataframe.columns:

        raise ValueError(
            "Training dataset must contain "
            "a 'target' column."
        )

    X = dataframe.drop(
        columns=["target"]
    )

    y = dataframe["target"].astype(
        int
    )

    return X, y


# ==========================================================
# FEATURE COLUMN ORDER
# ==========================================================

def feature_columns(
    dataframe: pd.DataFrame,
) -> list[str]:
    """
    Return deterministic feature-column order.
    """

    return sorted(
        column
        for column in dataframe.columns
        if column != "target"
    )


# ==========================================================
# LIVE FEATURE VECTOR
# ==========================================================

def build_live_features(
    digits: Iterable[int],
) -> Optional[pd.DataFrame]:
    """
    Build a one-row DataFrame suitable for model.predict()
    or model.predict_proba().
    """

    features = build_feature_row(
        digits
    )

    if features is None:
        return None

    dataframe = pd.DataFrame(
        [features]
    )

    return dataframe


# ==========================================================
# QUICK DIAGNOSTIC
# ==========================================================

if __name__ == "__main__":

    example_digits = [
        1, 4, 7, 2, 8,
        3, 5, 5, 9, 0,
        4, 6, 1, 8, 2,
        7, 3, 9, 1, 6,
        4, 8, 0, 5, 2,
        9, 7, 1, 3, 6,
        8, 4, 2, 0, 5,
    ]

    dataset = build_training_dataset(
        example_digits
    )

    print(
        "Feature dataset shape:",
        dataset.shape,
    )

    if not dataset.empty:

        X, y = split_features_target(
            dataset
        )

        print(
            "Feature columns:",
            len(X.columns),
        )

        print(
            "Target:",
            y.tolist(),
        )

        print(
            "\nLatest feature row:"
        )

        print(
            X.tail(1).to_string(
                index=False
            )
        )
