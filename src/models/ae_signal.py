Now the very last model file:

Stay inside the models folder
Click "Add file" → "Create new file"
Type in the filename box:

ae_signal.py

Paste this code:

python"""
models/ae_signal.py
────────────────────
Adverse Event signal detection using deep learning.

Implements an autoencoder-based anomaly detector that identifies
unexpected adverse event patterns in clinical trial safety data.

Use cases:
    - Early detection of unexpected AE clustering by body system
    - Identification of subjects with unusual AE burden profiles
    - Signal detection for pharmacovigilance review
    - Data quality flagging for outlier AE records

Architecture:
    ADAE feature matrix → Autoencoder → Reconstruction error
    High reconstruction error = anomalous AE pattern

Author : Girish Rajeev
         Clinical Data Scientist | Regulatory Standards Leader | AI/ML Solution Engineer
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)


# ── Signal result ─────────────────────────────────────────────────────────────

@dataclass
class AESignalResult:
    """Results from AE signal detection."""

    usubjid: list[str]
    reconstruction_error: list[float]
    anomaly_flag: list[bool]
    threshold: float
    n_anomalies: int
    feature_names: list[str] = field(default_factory=list)

    def to_dataframe(self) -> pd.DataFrame:
        """Export results as a DataFrame."""
        return pd.DataFrame({
            "USUBJID":              self.usubjid,
            "RECONSTRUCTION_ERROR": self.reconstruction_error,
            "ANOMALY_FLAG":         self.anomaly_flag,
        }).sort_values("RECONSTRUCTION_ERROR", ascending=False)

    def summary(self) -> dict:
        return {
            "n_subjects":   len(self.usubjid),
            "n_anomalies":  self.n_anomalies,
            "anomaly_rate": round(100 * self.n_anomalies / len(self.usubjid), 1),
            "threshold":    round(self.threshold, 4),
            "mean_error":   round(float(np.mean(self.reconstruction_error)), 4),
            "max_error":    round(float(np.max(self.reconstruction_error)), 4),
        }


# ── AE Signal Detector ────────────────────────────────────────────────────────

class AESignalDetector:
    """
    Autoencoder-based adverse event signal detector.

    Builds a subject-level AE feature matrix from ADAE data
    (AE counts per body system, severity distribution, relatedness
    profile) then trains a denoising autoencoder to learn the
    expected AE pattern distribution.

    Subjects with high reconstruction error deviate from the
    expected pattern and are flagged for pharmacovigilance review.

    Parameters
    ----------
    encoding_dim : int
        Bottleneck layer size. Default 16.
    epochs : int
        Training epochs. Default 100.
    batch_size : int
        Training batch size. Default 16.
    contamination : float
        Expected fraction of anomalies (used to set threshold).
        Default 0.05 (5%).
    random_state : int
        Random seed. Default 42.

    Examples
    --------
    >>> detector = AESignalDetector(contamination=0.05)
    >>> result = detector.fit_detect(adae_df, adsl_df)
    >>> flagged = result.to_dataframe()[result.to_dataframe()["ANOMALY_FLAG"]]
    >>> print(flagged)
    """

    def __init__(
        self,
        encoding_dim: int = 16,
        epochs: int = 100,
        batch_size: int = 16,
        contamination: float = 0.05,
        random_state: int = 42,
        verbose: int = 0,
    ) -> None:
        self.encoding_dim  = encoding_dim
        self.epochs        = epochs
        self.batch_size    = batch_size
        self.contamination = contamination
        self.random_state  = random_state
        self.verbose       = verbose
        self._scaler       = StandardScaler()
        self._model        = None
        self._feature_names: list[str] = []
        np.random.seed(random_state)

    # ── Public API ────────────────────────────────────────────────────────────

    def fit_detect(
        self,
        adae_df: pd.DataFrame,
        adsl_df: pd.DataFrame,
    ) -> AESignalResult:
        """
        Build feature matrix, train autoencoder, and detect anomalies.

        Parameters
        ----------
        adae_df : pd.DataFrame
            ADAE dataset with treatment-emergent AEs.
        adsl_df : pd.DataFrame
            ADSL dataset (for subject list and denominator).

        Returns
        -------
        AESignalResult
            Detection results with reconstruction errors and flags.
        """
        logger.info("Building AE feature matrix...")
        feature_df = self._build_feature_matrix(adae_df, adsl_df)
        self._feature_names = feature_df.columns.tolist()

        X = self._scaler.fit_transform(feature_df.values).astype(np.float32)

        logger.info("Training AE signal autoencoder (%d subjects, %d features)...",
                    X.shape[0], X.shape[1])
        self._train(X)

        errors = self._reconstruction_error(X)
        threshold = float(np.percentile(errors, 100 * (1 - self.contamination)))
        flags = errors > threshold

        result = AESignalResult(
            usubjid=feature_df.index.tolist(),
            reconstruction_error=errors.tolist(),
            anomaly_flag=flags.tolist(),
            threshold=threshold,
            n_anomalies=int(flags.sum()),
            feature_names=self._feature_names,
        )
        logger.info(
            "Signal detection complete: %d anomalies detected (%.1f%% of %d subjects)",
            result.n_anomalies,
            100 * result.n_anomalies / len(result.usubjid),
            len(result.usubjid),
        )
        return result

    def reconstruct(self, adae_df: pd.DataFrame, adsl_df: pd.DataFrame) -> pd.DataFrame:
        """
        Return reconstruction errors for new subjects (post-training).

        Parameters
        ----------
        adae_df : pd.DataFrame
            New ADAE data to score.
        adsl_df : pd.DataFrame
            Corresponding ADSL data.

        Returns
        -------
        pd.DataFrame
            DataFrame with USUBJID and reconstruction error.
        """
        if self._model is None:
            raise RuntimeError("Model must be trained first. Call fit_detect().")
        feature_df = self._build_feature_matrix(adae_df, adsl_df)
        X = self._scaler.transform(feature_df.values).astype(np.float32)
        errors = self._reconstruction_error(X)
        return pd.DataFrame({
            "USUBJID":              feature_df.index,
            "RECONSTRUCTION_ERROR": errors,
        })

    # ── Feature engineering ───────────────────────────────────────────────────

    @staticmethod
    def _build_feature_matrix(
        adae_df: pd.DataFrame,
        adsl_df: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Build subject-level AE feature matrix from ADAE data.

        Features per subject:
        - Total TEAE count
        - AE count per body system (AEBODSYS)
        - Count by severity grade (Grade 1-5)
        - Count of serious AEs
        - Count of related AEs
        - Count of Grade 3+ AEs
        """
        teae = adae_df[
            adae_df.get("TRTEMFL", pd.Series("Y", index=adae_df.index)) == "Y"
        ].copy()

        all_subjects = adsl_df["USUBJID"].unique()
        features = pd.DataFrame(index=all_subjects)
        features.index.name = "USUBJID"

        # Total TEAE count
        total = teae.groupby("USUBJID").size().rename("TOTAL_TEAE")
        features = features.join(total).fillna(0)

        # AE count by body system
        if "AEBODSYS" in teae.columns:
            bodsys_counts = (
                teae.groupby(["USUBJID", "AEBODSYS"])
                .size()
                .unstack(fill_value=0)
            )
            bodsys_counts.columns = [
                f"SOC_{c.replace(' ', '_').upper()[:20]}"
                for c in bodsys_counts.columns
            ]
            features = features.join(bodsys_counts).fillna(0)

        # AE count by severity grade
        if "ATOXGR" in teae.columns:
            for grade in [1, 2, 3, 4, 5]:
                col = f"GRADE_{grade}_N"
                grade_counts = (
                    teae[teae["ATOXGR"] == grade]
                    .groupby("USUBJID").size()
                    .rename(col)
                )
                features = features.join(grade_counts).fillna(0)

        # Serious AEs
        if "ASER" in teae.columns:
            serious = (
                teae[teae["ASER"] == "Y"]
                .groupby("USUBJID").size()
                .rename("SERIOUS_N")
            )
            features = features.join(serious).fillna(0)

        # Related AEs
        if "AEREL" in teae.columns:
            related = (
                teae[teae["AEREL"] == "Y"]
                .groupby("USUBJID").size()
                .rename("RELATED_N")
            )
            features = features.join(related).fillna(0)

        return features.astype(float)

    # ── Model training ────────────────────────────────────────────────────────

    def _train(self, X: np.ndarray) -> None:
        """Train the denoising autoencoder."""
        import tensorflow as tf
        from tensorflow import keras

        tf.random.set_seed(self.random_state)
        n_features = X.shape[1]

        self._model = self._build_autoencoder(n_features, keras)

        # Add noise for denoising training
        noise = np.random.normal(0, 0.1, X.shape).astype(np.float32)
        X_noisy = X + noise

        self._model.fit(
            X_noisy, X,
            epochs=self.epochs,
            batch_size=self.batch_size,
            validation_split=0.1,
            verbose=self.verbose,
            callbacks=[
                keras.callbacks.EarlyStopping(
                    patience=10,
                    restore_best_weights=True,
                    verbose=0,
                )
            ],
        )
        logger.info("Autoencoder training complete.")

    def _build_autoencoder(self, n_features: int, keras) -> "keras.Model":
        """Build symmetric denoising autoencoder for AE signals."""
        inputs = keras.Input(shape=(n_features,))

        # Encoder
        x = keras.layers.Dense(
            max(32, n_features * 2), activation="relu"
        )(inputs)
        x = keras.layers.Dropout(0.2)(x)
        encoded = keras.layers.Dense(
            self.encoding_dim, activation="relu"
        )(x)

        # Decoder
        x = keras.layers.Dense(
            max(32, n_features * 2), activation="relu"
        )(encoded)
        outputs = keras.layers.Dense(n_features, activation="linear")(x)

        model = keras.Model(inputs, outputs, name="ae_signal_detector")
        model.compile(optimizer="adam", loss="mse")
        return model

    def _reconstruction_error(self, X: np.ndarray) -> np.ndarray:
        """Compute mean squared reconstruction error per subject."""
        X_pred = self._model.predict(X, verbose=0)
        return np.mean(np.square(X - X_pred), axis=1)
