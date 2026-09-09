"""
============================================================
PRINCE PAUL FX
DIGITMATCH AI ANALYZER
LIGHTGBM ENGINE
VERSION 2.0
============================================================

Purpose:
    Train, validate, save, load and serve LightGBM
    multiclass digit prediction models.

Target:
    Predict the NEXT digit (0-9).

Important:
    This engine does not execute trades.

Validation:
    Chronological validation is used to reduce
    look-ahead leakage.

Model persistence:
    Uses joblib to save/load the sklearn LightGBM
    classifier consistently.

============================================================
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Optional

import joblib
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier


# ============================================================
# CONFIGURATION
# ============================================================

NUM_CLASSES = 10

DEFAULT_MODEL_PATH = (
    "models/lightgbm_digitmatch.joblib"
)

DEFAULT_METADATA_PATH = (
    "models/lightgbm_metadata.json"
)


# ============================================================
# LIGHTGBM ENGINE
# ============================================================

class LightGBMEngine:
    """
    LightGBM multiclass digit prediction engine.

    Input:
        Feature matrix containing historical tick features.

    Target:
        Next digit, values 0-9.
    """

    def __init__(
        self,
        model_path: str | Path = DEFAULT_MODEL_PATH,
        metadata_path: str | Path = DEFAULT_METADATA_PATH,
    ):

        self.model_path = Path(
            model_path
        )

        self.metadata_path = Path(
            metadata_path
        )

        self.model: Optional[
            LGBMClassifier
        ] = None

        self.feature_names: list[str] = []

        self.training_rows: int = 0

        self.validation_rows: int = 0

        self.validation_accuracy: Optional[
            float
        ] = None

        self.validation_logloss: Optional[
            float
        ] = None

        self.best_iteration: Optional[
            int
        ] = None


    # ========================================================
    # MODEL CREATION
    # ========================================================

    def _create_model(
        self,
    ) -> LGBMClassifier:
        """
        Create the LightGBM classifier.

        The parameters are intentionally conservative.
        """

        return LGBMClassifier(

            objective="multiclass",

            num_class=NUM_CLASSES,

            n_estimators=300,

            learning_rate=0.03,

            num_leaves=31,

            max_depth=-1,

            min_child_samples=20,

            subsample=0.90,

            colsample_bytree=0.90,

            reg_alpha=0.10,

            reg_lambda=1.0,

            random_state=42,

            n_jobs=-1,

            verbosity=-1,

        )


    # ========================================================
    # FEATURE PREPARATION
    # ========================================================

    @staticmethod
    def _prepare_features(
        X: pd.DataFrame | np.ndarray,
    ) -> pd.DataFrame:
        """
        Normalize feature input into a DataFrame.
        """

        if isinstance(
            X,
            pd.DataFrame,
        ):

            dataframe = X.copy()

        else:

            dataframe = pd.DataFrame(
                X
            )

        # Replace invalid numerical values.

        dataframe = dataframe.replace(
            [np.inf, -np.inf],
            np.nan,
        )

        dataframe = dataframe.fillna(
            0
        )

        return dataframe


    # ========================================================
    # TARGET PREPARATION
    # ========================================================

    @staticmethod
    def _prepare_target(
        y,
    ) -> np.ndarray:
        """
        Validate target digits.
        """

        values = np.asarray(
            y,
            dtype=int,
        )

        if values.ndim != 1:

            values = values.reshape(
                -1
            )

        invalid = values[
            (values < 0)
            | (values >= NUM_CLASSES)
        ]

        if len(invalid) > 0:

            raise ValueError(
                "Target contains invalid "
                "digit classes: "
                f"{sorted(set(invalid.tolist()))}"
            )

        return values


    # ========================================================
    # TRAIN
    # ========================================================

    def train(
        self,
        X_train: pd.DataFrame | np.ndarray,
        y_train,
        X_validation: Optional[
            pd.DataFrame | np.ndarray
        ] = None,
        y_validation=None,
    ) -> dict[str, Any]:
        """
        Train LightGBM.

        If validation data is supplied, it is used only
        for evaluation and early stopping.
        """

        X_train_df = (
            self._prepare_features(
                X_train
            )
        )

        y_train_array = (
            self._prepare_target(
                y_train
            )
        )

        if len(X_train_df) != len(
            y_train_array
        ):

            raise ValueError(
                "X_train and y_train "
                "have different lengths."
            )

        # ----------------------------------------------------
        # Check all classes.
        # ----------------------------------------------------

        present_classes = sorted(
            set(
                y_train_array.tolist()
            )
        )

        missing_classes = [
            digit
            for digit in range(
                NUM_CLASSES
            )
            if digit not in present_classes
        ]

        if missing_classes:

            raise ValueError(
                "Training data is missing "
                f"digit classes: "
                f"{missing_classes}"
            )

        # ----------------------------------------------------
        # Store feature names.
        # ----------------------------------------------------

        self.feature_names = [
            str(column)
            for column in X_train_df.columns
        ]

        # ----------------------------------------------------
        # Create model.
        # ----------------------------------------------------

        self.model = (
            self._create_model()
        )

        # ----------------------------------------------------
        # Prepare validation.
        # ----------------------------------------------------

        validation_set = None

        if (
            X_validation is not None
            and y_validation is not None
        ):

            X_validation_df = (
                self._prepare_features(
                    X_validation
                )
            )

            y_validation_array = (
                self._prepare_target(
                    y_validation
                )
            )

            if len(
                X_validation_df
            ) != len(
                y_validation_array
            ):

                raise ValueError(
                    "X_validation and "
                    "y_validation have "
                    "different lengths."
                )

            # Make sure validation columns
            # match training columns.

            X_validation_df = (
                X_validation_df.reindex(
                    columns=self.feature_names,
                    fill_value=0,
                )
            )

            validation_set = (
                X_validation_df,
                y_validation_array,
            )

        # ----------------------------------------------------
        # Train.
        # ----------------------------------------------------

        if validation_set is not None:

            X_val, y_val = (
                validation_set
            )

            self.model.fit(

                X_train_df,

                y_train_array,

                eval_set=[
                    (
                        X_val,
                        y_val,
                    )
                ],

                callbacks=[],

            )

        else:

            self.model.fit(

                X_train_df,

                y_train_array,

            )

        # ----------------------------------------------------
        # Store metadata.
        # ----------------------------------------------------

        self.training_rows = (
            len(X_train_df)
        )

        if validation_set is not None:

            self.validation_rows = (
                len(X_val)
            )

            predictions = (
                self.model.predict(
                    X_val
                )
            )

            self.validation_accuracy = (
                float(
                    np.mean(
                        predictions
                        == y_val
                    )
                )
            )

            try:

                from sklearn.metrics import (
                    log_loss
                )

                probabilities = (
                    self.model.predict_proba(
                        X_val
                    )
                )

                self.validation_logloss = (
                    float(
                        log_loss(
                            y_val,
                            probabilities,
                            labels=list(
                                range(
                                    NUM_CLASSES
                                )
                            ),
                        )
                    )
                )

            except Exception:

                self.validation_logloss = (
                    None
                )

        # ----------------------------------------------------
        # Best iteration.
        # ----------------------------------------------------

        best_iteration = getattr(
            self.model,
            "best_iteration_",
            None,
        )

        if best_iteration is not None:

            try:

                self.best_iteration = (
                    int(
                        best_iteration
                    )
                )

            except (
                TypeError,
                ValueError,
            ):

                self.best_iteration = (
                    None
                )

        report = {
            "training_rows": (
                self.training_rows
            ),
            "validation_rows": (
                self.validation_rows
            ),
            "feature_count": len(
                self.feature_names
            ),
            "validation_accuracy": (
                self.validation_accuracy
            ),
            "validation_logloss": (
                self.validation_logloss
            ),
            "best_iteration": (
                self.best_iteration
            ),
            "num_classes": (
                NUM_CLASSES
            ),
        }

        return report


    # ========================================================
    # PREDICT PROBABILITIES
    # ========================================================

    def predict_proba(
        self,
        X: pd.DataFrame | np.ndarray,
    ) -> np.ndarray:
        """
        Return probability for every digit 0-9.
        """

        if self.model is None:

            raise RuntimeError(
                "LightGBM model is not loaded "
                "or trained."
            )

        dataframe = (
            self._prepare_features(
                X
            )
        )

        if self.feature_names:

            dataframe = (
                dataframe.reindex(
                    columns=self.feature_names,
                    fill_value=0,
                )
            )

        probabilities = (
            self.model.predict_proba(
                dataframe
            )
        )

        probabilities = np.asarray(
            probabilities,
            dtype=float,
        )

        # ----------------------------------------------------
        # Defensive normalization.
        # ----------------------------------------------------

        row_sums = (
            probabilities.sum(
                axis=1,
                keepdims=True,
            )
        )

        row_sums[
            row_sums == 0
        ] = 1.0

        probabilities = (
            probabilities
            / row_sums
        )

        return probabilities


    # ========================================================
    # PREDICT DIGIT
    # ========================================================

    def predict(
        self,
        X: pd.DataFrame | np.ndarray,
    ) -> np.ndarray:
        """
        Return the highest-probability digit.
        """

        probabilities = (
            self.predict_proba(
                X
            )
        )

        return np.argmax(
            probabilities,
            axis=1,
        )


    # ========================================================
    # SINGLE PREDICTION
    # ========================================================

    def predict_single(
        self,
        X: pd.DataFrame | np.ndarray,
    ) -> dict[str, Any]:
        """
        Produce a detailed prediction for one
        feature row.
        """

        probabilities = (
            self.predict_proba(
                X
            )
        )

        if probabilities.shape[0] != 1:

            raise ValueError(
                "predict_single expects "
                "exactly one feature row."
            )

        probability_row = (
            probabilities[0]
        )

        ranked_indices = np.argsort(
            probability_row
        )[::-1]

        predicted_digit = int(
            ranked_indices[0]
        )

        predicted_probability = float(
            probability_row[
                predicted_digit
            ]
        )

        second_digit = int(
            ranked_indices[1]
        )

        second_probability = float(
            probability_row[
                second_digit
            ]
        )

        return {
            "predicted_digit": (
                predicted_digit
            ),
            "probability": (
                predicted_probability
            ),
            "second_digit": (
                second_digit
            ),
            "second_probability": (
                second_probability
            ),
            "gap": (
                predicted_probability
                - second_probability
            ),
            "probabilities": {
                str(digit): float(
                    probability_row[
                        digit
                    ]
                )
                for digit in range(
                    NUM_CLASSES
                )
            },
            "ranked_digits": [
                int(index)
                for index in ranked_indices
            ],
        }


    # ========================================================
    # TOP CANDIDATES
    # ========================================================

    def top_candidates(
        self,
        X: pd.DataFrame | np.ndarray,
        top_n: int = 3,
    ) -> list[dict[str, Any]]:
        """
        Return the strongest candidate digits.
        """

        if top_n < 1:

            raise ValueError(
                "top_n must be at least 1."
            )

        top_n = min(
            top_n,
            NUM_CLASSES,
        )

        probabilities = (
            self.predict_proba(
                X
            )
        )

        if probabilities.shape[0] != 1:

            raise ValueError(
                "top_candidates expects "
                "exactly one feature row."
            )

        row = probabilities[0]

        indices = np.argsort(
            row
        )[::-1][:top_n]

        return [
            {
                "digit": int(
                    index
                ),
                "probability": float(
                    row[index]
                ),
                "rank": rank + 1,
            }
            for rank, index in enumerate(
                indices
            )
        ]


    # ========================================================
    # SAVE
    # ========================================================

    def save(
        self,
        model_path: Optional[
            str | Path
        ] = None,
        metadata_path: Optional[
            str | Path
        ] = None,
    ) -> tuple[Path, Path]:
        """
        Save the complete sklearn LightGBM
        classifier using joblib.
        """

        if self.model is None:

            raise RuntimeError(
                "Cannot save an empty model."
            )

        model_destination = Path(
            model_path
            if model_path is not None
            else self.model_path
        )

        metadata_destination = Path(
            metadata_path
            if metadata_path is not None
            else self.metadata_path
        )

        model_destination.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        metadata_destination.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        # ----------------------------------------------------
        # Save sklearn wrapper.
        # ----------------------------------------------------

        joblib.dump(
            self.model,
            model_destination,
        )

        # ----------------------------------------------------
        # Save metadata separately.
        # ----------------------------------------------------

        metadata = {
            "engine": "LightGBM",
            "version": "2.0",
            "num_classes": (
                NUM_CLASSES
            ),
            "feature_count": len(
                self.feature_names
            ),
            "feature_names": (
                self.feature_names
            ),
            "training_rows": (
                self.training_rows
            ),
            "validation_rows": (
                self.validation_rows
            ),
            "validation_accuracy": (
                self.validation_accuracy
            ),
            "validation_logloss": (
                self.validation_logloss
            ),
            "best_iteration": (
                self.best_iteration
            ),
            "model_format": "joblib",
        }

        with open(
            metadata_destination,
            "w",
            encoding="utf-8",
        ) as file:

            json.dump(
                metadata,
                file,
                indent=2,
            )

        # ----------------------------------------------------
        # Update paths.
        # ----------------------------------------------------

        self.model_path = (
            model_destination
        )

        self.metadata_path = (
            metadata_destination
        )

        return (
            model_destination,
            metadata_destination,
        )


    # ========================================================
    # LOAD
    # ========================================================

    def load(
        self,
        model_path: Optional[
            str | Path
        ] = None,
        metadata_path: Optional[
            str | Path
        ] = None,
    ) -> None:
        """
        Load the sklearn LightGBM classifier
        and its metadata.
        """

        model_source = Path(
            model_path
            if model_path is not None
            else self.model_path
        )

        metadata_source = Path(
            metadata_path
            if metadata_path is not None
            else self.metadata_path
        )

        if not model_source.exists():

            raise FileNotFoundError(
                "LightGBM model not found: "
                f"{model_source}"
            )

        # ----------------------------------------------------
        # Load model.
        # ----------------------------------------------------

        loaded_model = joblib.load(
            model_source
        )

        if not isinstance(
            loaded_model,
            LGBMClassifier,
        ):

            raise TypeError(
                "Loaded file is not a "
                "LightGBM sklearn classifier."
            )

        self.model = (
            loaded_model
        )

        # ----------------------------------------------------
        # Load metadata if available.
        # ----------------------------------------------------

        if metadata_source.exists():

            with open(
                metadata_source,
                "r",
                encoding="utf-8",
            ) as file:

                metadata = json.load(
                    file
                )

            self.feature_names = [
                str(name)
                for name in metadata.get(
                    "feature_names",
                    [],
                )
            ]

            self.training_rows = int(
                metadata.get(
                    "training_rows",
                    0,
                )
            )

            self.validation_rows = int(
                metadata.get(
                    "validation_rows",
                    0,
                )
            )

            accuracy = metadata.get(
                "validation_accuracy"
            )

            if accuracy is not None:

                self.validation_accuracy = (
                    float(
                        accuracy
                    )
                )

            logloss = metadata.get(
                "validation_logloss"
            )

            if logloss is not None:

                self.validation_logloss = (
                    float(
                        logloss
                    )
                )

            best_iteration = metadata.get(
                "best_iteration"
            )

            if best_iteration is not None:

                self.best_iteration = (
                    int(
                        best_iteration
                    )
                )

        self.model_path = (
            model_source
        )

        self.metadata_path = (
            metadata_source
        )


    # ========================================================
    # STATUS
    # ========================================================

    def status(self) -> dict[str, Any]:
        """
        Return engine status for the dashboard.
        """

        loaded = (
            self.model is not None
        )

        return {
            "engine": "LightGBM",
            "loaded": loaded,
            "trained": loaded,
            "model_path": str(
                self.model_path
            ),
            "metadata_path": str(
                self.metadata_path
            ),
            "feature_count": len(
                self.feature_names
            ),
            "training_rows": (
                self.training_rows
            ),
            "validation_rows": (
                self.validation_rows
            ),
            "validation_accuracy": (
                self.validation_accuracy
            ),
            "validation_logloss": (
                self.validation_logloss
            ),
            "best_iteration": (
                self.best_iteration
            ),
            "num_classes": (
                NUM_CLASSES
            ),
        }


# ============================================================
# COMPATIBILITY HELPERS
# ============================================================

def train_lightgbm(
    X_train,
    y_train,
    X_validation=None,
    y_validation=None,
) -> LightGBMEngine:
    """
    Convenience function compatible with a simple
    training workflow.
    """

    engine = LightGBMEngine()

    engine.train(
        X_train=X_train,
        y_train=y_train,
        X_validation=X_validation,
        y_validation=y_validation,
    )

    return engine


# ============================================================
# TEST
# ============================================================

if __name__ == "__main__":

    print("=" * 64)
    print(
        "PRINCE PAUL FX"
    )
    print(
        "LIGHTGBM ENGINE"
    )
    print(
        "VERSION 2.0"
    )
    print("=" * 64)

    engine = LightGBMEngine()

    print()
    print(
        "Engine initialized successfully."
    )

    print()
    print(
        "Status:"
    )

    print(
        json.dumps(
            engine.status(),
            indent=2,
        )
    )
