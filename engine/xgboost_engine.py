"""
============================================================
PRINCE PAUL FX
DIGITMATCH AI ANALYZER
XGBOOST ENGINE
VERSION 1.0
============================================================

Purpose:
    Train an XGBoost multiclass classifier to estimate the
    probability of the next Deriv tick digit (0-9).

Role:
    Secondary model in the PRINCE PAUL FX ensemble.

The final scanner will NOT rely on XGBoost alone.
LightGBM and XGBoost probabilities will later be combined
by engine/ensemble.py.

Important:
    This is probabilistic analysis, not a guarantee of the
    next tick outcome.

============================================================
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from xgboost import XGBClassifier

from engine.features import (
    build_live_features,
    build_training_dataset,
    feature_columns,
    split_features_target,
)


# ==========================================================
# CONFIGURATION
# ==========================================================

NUM_CLASSES = 10

DEFAULT_MODEL_DIR = Path("models")

MODEL_FILENAME = (
    "xgboost_digitmatch.json"
)

METADATA_FILENAME = (
    "xgboost_metadata.json"
)

RANDOM_SEED = 42


# ==========================================================
# XGBOOST ENGINE
# ==========================================================

class XGBoostDigitEngine:
    """
    XGBoost multiclass engine for next-digit analysis.
    """

    def __init__(
        self,
        model_dir: str | Path = DEFAULT_MODEL_DIR,
    ):

        self.model_dir = Path(
            model_dir
        )

        self.model_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        self.model: Optional[
            XGBClassifier
        ] = None

        self.columns: list[str] = []

        self.is_trained = False

        self.training_rows = 0

        self.validation_accuracy = None

        self.validation_logloss = None

        self.best_iteration = None


    # ======================================================
    # MODEL CREATION
    # ======================================================

    def _create_model(
        self,
    ) -> XGBClassifier:
        """
        Create the XGBoost multiclass classifier.
        """

        return XGBClassifier(

            objective="multi:softprob",

            num_class=NUM_CLASSES,

            n_estimators=400,

            learning_rate=0.03,

            max_depth=6,

            min_child_weight=3,

            subsample=0.90,

            colsample_bytree=0.90,

            gamma=0.0,

            reg_alpha=0.10,

            reg_lambda=1.0,

            eval_metric="mlogloss",

            random_state=RANDOM_SEED,

            n_jobs=-1,

            tree_method="hist",

        )


    # ======================================================
    # PREPARE DATA
    # ======================================================

    def prepare_data(
        self,
        digits,
    ):
        """
        Convert a digit sequence into supervised-learning
        features and targets.
        """

        dataset = build_training_dataset(
            digits
        )

        if dataset.empty:

            raise ValueError(
                "Not enough tick history to create "
                "an XGBoost training dataset."
            )

        X, y = split_features_target(
            dataset
        )

        columns = feature_columns(
            X
        )

        X = X[columns]

        return X, y


    # ======================================================
    # CHRONOLOGICAL SPLIT
    # ======================================================

    @staticmethod
    def chronological_split(
        X: pd.DataFrame,
        y: pd.Series,
        validation_fraction: float = 0.20,
    ):
        """
        Chronological train/validation split.

        The newest portion is reserved for validation.

        No random shuffling is performed.
        """

        if not 0.05 <= validation_fraction <= 0.40:

            raise ValueError(
                "validation_fraction must be between "
                "0.05 and 0.40."
            )

        total_rows = len(X)

        if total_rows < 20:

            raise ValueError(
                "At least 20 examples are recommended "
                "for validation."
            )

        split_index = int(
            total_rows
            * (1.0 - validation_fraction)
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


    # ======================================================
    # TRAIN
    # ======================================================

    def train(
        self,
        digits,
        validation_fraction: float = 0.20,
    ) -> dict:
        """
        Train XGBoost and evaluate it on later,
        unseen observations.
        """

        X, y = self.prepare_data(
            digits
        )

        (
            X_train,
            X_valid,
            y_train,
            y_valid,
        ) = self.chronological_split(
            X,
            y,
            validation_fraction,
        )

        model = self._create_model()

        model.fit(
            X_train,
            y_train,
            eval_set=[
                (
                    X_valid,
                    y_valid,
                )
            ],
            verbose=False,
        )

        self.model = model

        self.columns = list(
            X_train.columns
        )

        self.is_trained = True

        self.training_rows = len(
            X_train
        )

        if hasattr(
            model,
            "best_iteration",
        ):

            self.best_iteration = (
                int(
                    model.best_iteration
                )
                if model.best_iteration
                is not None
                else None
            )

        probabilities = (
            model.predict_proba(
                X_valid
            )
        )

        predictions = np.argmax(
            probabilities,
            axis=1,
        )

        accuracy = float(
            np.mean(
                predictions
                == y_valid.to_numpy()
            )
        )

        logloss = self.multiclass_logloss(
            y_valid.to_numpy(),
            probabilities,
        )

        self.validation_accuracy = (
            accuracy
        )

        self.validation_logloss = (
            logloss
        )

        return {
            "training_rows": int(
                len(X_train)
            ),
            "validation_rows": int(
                len(X_valid)
            ),
            "accuracy": accuracy,
            "logloss": logloss,
            "best_iteration": (
                self.best_iteration
            ),
        }


    # ======================================================
    # LOG LOSS
    # ======================================================

    @staticmethod
    def multiclass_logloss(
        y_true,
        probabilities,
    ) -> float:
        """
        Calculate multiclass logarithmic loss.
        """

        y_true = np.asarray(
            y_true,
            dtype=int,
        )

        probabilities = np.asarray(
            probabilities,
            dtype=float,
        )

        probabilities = np.clip(
            probabilities,
            1e-15,
            1.0,
        )

        probabilities = (
            probabilities
            / probabilities.sum(
                axis=1,
                keepdims=True,
            )
        )

        rows = np.arange(
            len(y_true)
        )

        correct_probability = (
            probabilities[
                rows,
                y_true,
            ]
        )

        return float(
            -np.mean(
                np.log(
                    correct_probability
                )
            )
        )


    # ======================================================
    # PREDICT PROBABILITIES
    # ======================================================

    def predict_proba(
        self,
        digits,
    ) -> Optional[np.ndarray]:
        """
        Return probability for all ten digits.
        """

        if not self.is_trained:

            raise RuntimeError(
                "XGBoost model has not been trained."
            )

        features = build_live_features(
            digits
        )

        if features is None:

            return None

        features = features.reindex(
            columns=self.columns,
            fill_value=0.0,
        )

        probabilities = (
            self.model.predict_proba(
                features
            )
        )

        probabilities = np.asarray(
            probabilities[0],
            dtype=float,
        )

        return probabilities


    # ======================================================
    # PREDICT
    # ======================================================

    def predict(
        self,
        digits,
    ) -> Optional[dict]:
        """
        Predict the most probable next digit.
        """

        probabilities = (
            self.predict_proba(
                digits
            )
        )

        if probabilities is None:

            return None

        predicted_digit = int(
            np.argmax(
                probabilities
            )
        )

        probability = float(
            probabilities[
                predicted_digit
            ]
        )

        probability_map = {
            digit: float(
                probabilities[digit]
            )
            for digit in range(
                NUM_CLASSES
            )
        }

        return {
            "digit": predicted_digit,
            "probability": probability,
            "probabilities": probability_map,
        }


    # ======================================================
    # TOP CANDIDATES
    # ======================================================

    def top_candidates(
        self,
        digits,
        count: int = 3,
    ) -> list[dict]:
        """
        Return the strongest XGBoost digit candidates.
        """

        if count < 1:

            return []

        count = min(
            count,
            NUM_CLASSES,
        )

        result = self.predict(
            digits
        )

        if result is None:

            return []

        ranked = sorted(
            result[
                "probabilities"
            ].items(),
            key=lambda item: item[1],
            reverse=True,
        )

        return [
            {
                "digit": int(digit),
                "probability": float(
                    probability
                ),
            }
            for digit, probability
            in ranked[:count]
        ]


    # ======================================================
    # SAVE
    # ======================================================

    def save(
        self,
    ) -> Path:
        """
        Save the trained XGBoost model and metadata.
        """

        if not self.is_trained:

            raise RuntimeError(
                "Cannot save an untrained model."
            )

        model_path = (
            self.model_dir
            / MODEL_FILENAME
        )

        metadata_path = (
            self.model_dir
            / METADATA_FILENAME
        )

        self.model.save_model(
            str(model_path)
        )

        metadata = {
            "model": "XGBoost",
            "task": "multiclass",
            "num_classes": NUM_CLASSES,
            "classes": list(
                range(NUM_CLASSES)
            ),
            "feature_columns": self.columns,
            "training_rows": self.training_rows,
            "validation_accuracy": (
                self.validation_accuracy
            ),
            "validation_logloss": (
                self.validation_logloss
            ),
            "best_iteration": (
                self.best_iteration
            ),
        }

        metadata_path.write_text(
            json.dumps(
                metadata,
                indent=2,
            ),
            encoding="utf-8",
        )

        return model_path


    # ======================================================
    # LOAD
    # ======================================================

    def load(
        self,
        model_path: str | Path | None = None,
        metadata_path: str | Path | None = None,
    ) -> None:
        """
        Load a previously trained XGBoost model.
        """

        if model_path is None:

            model_path = (
                self.model_dir
                / MODEL_FILENAME
            )

        if metadata_path is None:

            metadata_path = (
                self.model_dir
                / METADATA_FILENAME
            )

        model_path = Path(
            model_path
        )

        metadata_path = Path(
            metadata_path
        )

        if not model_path.exists():

            raise FileNotFoundError(
                f"XGBoost model not found: "
                f"{model_path}"
            )

        if not metadata_path.exists():

            raise FileNotFoundError(
                f"XGBoost metadata not found: "
                f"{metadata_path}"
            )

        model = self._create_model()

        model.load_model(
            str(model_path)
        )

        self.model = model

        metadata = json.loads(
            metadata_path.read_text(
                encoding="utf-8"
            )
        )

        self.columns = metadata.get(
            "feature_columns",
            [],
        )

        self.training_rows = int(
            metadata.get(
                "training_rows",
                0,
            )
        )

        self.validation_accuracy = (
            metadata.get(
                "validation_accuracy"
            )
        )

        self.validation_logloss = (
            metadata.get(
                "validation_logloss"
            )
        )

        best_iteration = metadata.get(
            "best_iteration"
        )

        self.best_iteration = (
            int(best_iteration)
            if best_iteration is not None
            else None
        )

        self.is_trained = True


    # ======================================================
    # STATUS
    # ======================================================

    def status(
        self,
    ) -> dict:
        """
        Return engine status for the dashboard.
        """

        return {
            "engine": "XGBoost",
            "trained": self.is_trained,
            "training_rows": self.training_rows,
            "validation_accuracy": (
                self.validation_accuracy
            ),
            "validation_logloss": (
                self.validation_logloss
            ),
            "feature_count": len(
                self.columns
            ),
            "best_iteration": (
                self.best_iteration
            ),
        }


# ==========================================================
# STANDALONE DIAGNOSTIC
# ==========================================================

if __name__ == "__main__":

    print(
        "PRINCE PAUL FX"
    )

    print(
        "XGBoost DigitMatch Engine"
    )

    print(
        "Engine loaded successfully."
    )

    print(
        f"Classes: {NUM_CLASSES}"
    )
