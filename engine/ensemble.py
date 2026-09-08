"""
============================================================
PRINCE PAUL FX
DIGITMATCH AI ANALYZER
ENSEMBLE ENGINE
VERSION 1.0
============================================================

Purpose:
    Combine LightGBM and XGBoost probability distributions
    into one ensemble probability distribution.

Pipeline:

        Tick History
             |
             v
        Feature Engine
             |
       +-----+------+
       |            |
       v            v
   LightGBM      XGBoost
       |            |
       +-----+------+
             |
             v
       ENSEMBLE ENGINE
             |
             v
       Ranked Digits 0-9
             |
             v
       Scanner / Confirmation

Important:
    Ensemble probability is an analytical estimate.
    It is NOT a guarantee of the next tick.

============================================================
"""

from __future__ import annotations

from typing import Iterable, Optional

import numpy as np


# ==========================================================
# CONFIGURATION
# ==========================================================

NUM_CLASSES = 10

DEFAULT_LIGHTGBM_WEIGHT = 0.50
DEFAULT_XGBOOST_WEIGHT = 0.50


# ==========================================================
# ENSEMBLE ENGINE
# ==========================================================

class DigitEnsembleEngine:
    """
    Combines LightGBM and XGBoost probability outputs.
    """

    def __init__(
        self,
        lightgbm_engine=None,
        xgboost_engine=None,
        lightgbm_weight: float = DEFAULT_LIGHTGBM_WEIGHT,
        xgboost_weight: float = DEFAULT_XGBOOST_WEIGHT,
    ):

        self.lightgbm_engine = lightgbm_engine
        self.xgboost_engine = xgboost_engine

        self.lightgbm_weight = float(
            lightgbm_weight
        )

        self.xgboost_weight = float(
            xgboost_weight
        )

        self._validate_weights()

        self.last_result: Optional[dict] = None


    # ======================================================
    # VALIDATE WEIGHTS
    # ======================================================

    def _validate_weights(
        self,
    ) -> None:

        if self.lightgbm_weight < 0:
            raise ValueError(
                "LightGBM weight cannot be negative."
            )

        if self.xgboost_weight < 0:
            raise ValueError(
                "XGBoost weight cannot be negative."
            )

        total = (
            self.lightgbm_weight
            + self.xgboost_weight
        )

        if total <= 0:

            raise ValueError(
                "At least one ensemble weight "
                "must be greater than zero."
            )


    # ======================================================
    # NORMALIZE PROBABILITIES
    # ======================================================

    @staticmethod
    def normalize_probabilities(
        probabilities: Iterable[float],
    ) -> np.ndarray:
        """
        Validate and normalize a ten-class probability vector.
        """

        values = np.asarray(
            list(probabilities),
            dtype=float,
        )

        if values.size != NUM_CLASSES:

            raise ValueError(
                "Probability vector must contain "
                "exactly 10 values."
            )

        values = np.nan_to_num(
            values,
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
        )

        values = np.maximum(
            values,
            0.0,
        )

        total = float(
            values.sum()
        )

        if total <= 0:

            return np.full(
                NUM_CLASSES,
                1.0 / NUM_CLASSES,
            )

        return values / total


    # ======================================================
    # COMBINE TWO MODELS
    # ======================================================

    def combine_probabilities(
        self,
        lightgbm_probabilities,
        xgboost_probabilities,
    ) -> np.ndarray:
        """
        Weighted probability averaging.

        The weights are normalized automatically.
        """

        lgb_prob = (
            self.normalize_probabilities(
                lightgbm_probabilities
            )
        )

        xgb_prob = (
            self.normalize_probabilities(
                xgboost_probabilities
            )
        )

        total_weight = (
            self.lightgbm_weight
            + self.xgboost_weight
        )

        lgb_weight = (
            self.lightgbm_weight
            / total_weight
        )

        xgb_weight = (
            self.xgboost_weight
            / total_weight
        )

        ensemble = (
            lgb_prob * lgb_weight
            + xgb_prob * xgb_weight
        )

        return self.normalize_probabilities(
            ensemble
        )


    # ======================================================
    # MODEL AGREEMENT
    # ======================================================

    @staticmethod
    def calculate_agreement(
        lightgbm_probabilities,
        xgboost_probabilities,
    ) -> dict:
        """
        Measure whether the two models select the same
        leading digit.
        """

        lgb_prob = (
            DigitEnsembleEngine
            .normalize_probabilities(
                lightgbm_probabilities
            )
        )

        xgb_prob = (
            DigitEnsembleEngine
            .normalize_probabilities(
                xgboost_probabilities
            )
        )

        lgb_digit = int(
            np.argmax(lgb_prob)
        )

        xgb_digit = int(
            np.argmax(xgb_prob)
        )

        same_digit = (
            lgb_digit == xgb_digit
        )

        difference = float(
            abs(
                lgb_prob[lgb_digit]
                - xgb_prob[xgb_digit]
            )
        )

        return {
            "lightgbm_digit": lgb_digit,
            "xgboost_digit": xgb_digit,
            "agreement": same_digit,
            "probability_difference": difference,
        }


    # ======================================================
    # RANK DIGITS
    # ======================================================

    @staticmethod
    def rank_digits(
        probabilities,
    ) -> list[dict]:
        """
        Rank all digits from highest probability to lowest.
        """

        probabilities = (
            DigitEnsembleEngine
            .normalize_probabilities(
                probabilities
            )
        )

        ranked = sorted(
            [
                (
                    digit,
                    float(
                        probabilities[digit]
                    ),
                )
                for digit in range(
                    NUM_CLASSES
                )
            ],
            key=lambda item: item[1],
            reverse=True,
        )

        return [
            {
                "rank": index + 1,
                "digit": digit,
                "probability": probability,
                "percentage": (
                    probability * 100.0
                ),
            }
            for index, (
                digit,
                probability,
            ) in enumerate(ranked)
        ]


    # ======================================================
    # PREDICT FROM TWO MODELS
    # ======================================================

    def predict_from_probabilities(
        self,
        lightgbm_probabilities,
        xgboost_probabilities,
    ) -> dict:
        """
        Create the complete ensemble result.
        """

        ensemble_probabilities = (
            self.combine_probabilities(
                lightgbm_probabilities,
                xgboost_probabilities,
            )
        )

        ranked = self.rank_digits(
            ensemble_probabilities
        )

        agreement = (
            self.calculate_agreement(
                lightgbm_probabilities,
                xgboost_probabilities,
            )
        )

        top = ranked[0]

        second = ranked[1]

        probability_gap = (
            top["probability"]
            - second["probability"]
        )

        result = {
            "predicted_digit": int(
                top["digit"]
            ),
            "probability": float(
                top["probability"]
            ),
            "percentage": float(
                top["percentage"]
            ),
            "second_digit": int(
                second["digit"]
            ),
            "second_probability": float(
                second["probability"]
            ),
            "probability_gap": float(
                probability_gap
            ),
            "ranked_digits": ranked,
            "probabilities": {
                digit: float(
                    ensemble_probabilities[
                        digit
                    ]
                )
                for digit in range(
                    NUM_CLASSES
                )
            },
            "model_agreement": agreement[
                "agreement"
            ],
            "lightgbm_digit": agreement[
                "lightgbm_digit"
            ],
            "xgboost_digit": agreement[
                "xgboost_digit"
            ],
        }

        self.last_result = result

        return result


    # ======================================================
    # PREDICT USING ENGINE OBJECTS
    # ======================================================

    def predict(
        self,
        digits,
    ) -> Optional[dict]:
        """
        Ask both model engines for probabilities and combine
        them.
        """

        if self.lightgbm_engine is None:

            raise RuntimeError(
                "LightGBM engine has not been supplied."
            )

        if self.xgboost_engine is None:

            raise RuntimeError(
                "XGBoost engine has not been supplied."
            )

        if not self.lightgbm_engine.is_trained:

            raise RuntimeError(
                "LightGBM engine is not trained."
            )

        if not self.xgboost_engine.is_trained:

            raise RuntimeError(
                "XGBoost engine is not trained."
            )

        lgb_probabilities = (
            self.lightgbm_engine
            .predict_proba(
                digits
            )
        )

        xgb_probabilities = (
            self.xgboost_engine
            .predict_proba(
                digits
            )
        )

        if lgb_probabilities is None:

            return None

        if xgb_probabilities is None:

            return None

        return self.predict_from_probabilities(
            lgb_probabilities,
            xgb_probabilities,
        )


    # ======================================================
    # TOP CANDIDATES
    # ======================================================

    def top_candidates(
        self,
        digits,
        count: int = 3,
    ) -> list[dict]:
        """
        Return the strongest ensemble candidates.
        """

        count = max(
            1,
            min(
                int(count),
                NUM_CLASSES,
            ),
        )

        result = self.predict(
            digits
        )

        if result is None:

            return []

        return result[
            "ranked_digits"
        ][:count]


    # ======================================================
    # SCANNER CANDIDATE
    # ======================================================

    def get_candidate(
        self,
        digits,
    ) -> Optional[dict]:
        """
        Return the current highest-ranked digit as the
        scanner candidate.
        """

        result = self.predict(
            digits
        )

        if result is None:

            return None

        return {
            "digit": result[
                "predicted_digit"
            ],
            "probability": result[
                "probability"
            ],
            "percentage": result[
                "percentage"
            ],
            "model_agreement": result[
                "model_agreement"
            ],
        }


    # ======================================================
    # ENGINE STATUS
    # ======================================================

    def status(
        self,
    ) -> dict:
        """
        Return current ensemble status.
        """

        lightgbm_trained = bool(
            self.lightgbm_engine
            and self.lightgbm_engine.is_trained
        )

        xgboost_trained = bool(
            self.xgboost_engine
            and self.xgboost_engine.is_trained
        )

        return {
            "engine": "Ensemble",
            "lightgbm_trained": lightgbm_trained,
            "xgboost_trained": xgboost_trained,
            "lightgbm_weight": (
                self.lightgbm_weight
            ),
            "xgboost_weight": (
                self.xgboost_weight
            ),
            "ready": (
                lightgbm_trained
                and xgboost_trained
            ),
            "last_prediction": (
                self.last_result
            ),
        }


# ==========================================================
# DIAGNOSTIC
# ==========================================================

if __name__ == "__main__":

    print(
        "PRINCE PAUL FX"
    )

    print(
        "DigitMatch Ensemble Engine"
    )

    # Example probabilities only.
    # These are NOT real market predictions.

    lightgbm_example = [
        0.08,
        0.09,
        0.10,
        0.07,
        0.11,
        0.09,
        0.10,
        0.14,
        0.11,
        0.11,
    ]

    xgboost_example = [
        0.07,
        0.10,
        0.09,
        0.08,
        0.10,
        0.11,
        0.09,
        0.15,
        0.10,
        0.11,
    ]

    engine = DigitEnsembleEngine()

    result = (
        engine.predict_from_probabilities(
            lightgbm_example,
            xgboost_example,
        )
    )

    print(
        "\nPredicted candidate:",
        result["predicted_digit"],
    )

    print(
        "Ensemble probability:",
        f"{result['percentage']:.2f}%",
    )

    print(
        "Model agreement:",
        result["model_agreement"],
    )

    print(
        "\nRanked digits:"
    )

    for item in result[
        "ranked_digits"
    ]:

        print(
            f"{item['rank']}. "
            f"Digit {item['digit']} "
            f"-> "
            f"{item['percentage']:.2f}%"
        )
