Now the exciting ML piece! This is what makes your repo truly stand out:

Stay inside the adam folder
Click "Add file" → "Create new file"
Type in the filename box:

imputation.py

Paste this code:

python"""
adam/imputation.py
───────────────────
ML-powered missing value imputation for ADaM datasets.

Implements three imputation strategies of increasing sophistication:

    1. SimpleImputer      — mean/median/mode (scikit-learn baseline)
    2. IterativeImputer   — MICE-style multivariate imputation (scikit-learn)
    3. NeuralImputer      — Autoencoder-based imputation (TensorFlow/Keras)

The NeuralImputer is particularly well-suited for:
    - High-dimensional lab panels with correlated analytes
    - Longitudinal data with structured missingness patterns
    - Datasets where MCAR/MAR assumptions may not hold

All imputers follow a consistent fit/transform/fit_transform API
compatible with scikit-learn pipelines.

Author : Girish Rajeev
         Clinical Data Scientist | Regulatory Standards Leader | AI/ML Solution Engineer
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.experimental import enable_iterative_imputer  # noqa: F401
from sklearn.impute import IterativeImputer, SimpleImputer
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)


# ── Simple imputer wrapper ────────────────────────────────────────────────────

class ClinicalSimpleImputer:
    """
    Simple mean/median/mode imputation for ADaM numeric columns.

    Wraps sklearn SimpleImputer with clinical data conventions:
    - Imputes only specified numeric columns
    - Preserves non-numeric columns unchanged
    - Logs imputation statistics for audit trail

    Parameters
    ----------
    strategy : str
        Imputation strategy: 'mean', 'median', or 'most_frequent'.
    numeric_cols : list[str], optional
        Columns to impute. If None, all numeric columns are imputed.

    Examples
    --------
    >>> imputer = ClinicalSimpleImputer(strategy="median")
    >>> adlb_imputed = imputer.fit_transform(adlb_df)
    """

    def __init__(
        self,
        strategy: str = "median",
        numeric_cols: Optional[list[str]] = None,
    ) -> None:
        self.strategy = strategy
        self.numeric_cols = numeric_cols
        self._imputer = SimpleImputer(strategy=strategy)
        self._fitted_cols: list[str] = []

    def fit(self, df: pd.DataFrame) -> "ClinicalSimpleImputer":
        cols = self._resolve_cols(df)
        self._imputer.fit(df[cols])
        self._fitted_cols = cols
        logger.info("ClinicalSimpleImputer fitted on %d columns", len(cols))
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        result = df.copy()
        imputed = self._imputer.transform(result[self._fitted_cols])
        result[self._fitted_cols] = imputed
        missing_before = df[self._fitted_cols].isna().sum().sum()
        missing_after  = result[self._fitted_cols].isna().sum().sum()
        logger.info(
            "SimpleImputer: %d missing values imputed (strategy=%s)",
            missing_before - missing_after,
            self.strategy,
        )
        return result

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        return self.fit(df).transform(df)

    def _resolve_cols(self, df: pd.DataFrame) -> list[str]:
        if self.numeric_cols:
            return [c for c in self.numeric_cols if c in df.columns]
        return df.select_dtypes(include=[np.number]).columns.tolist()


# ── MICE-style iterative imputer ──────────────────────────────────────────────

class ClinicalMICEImputer:
    """
    MICE-style multivariate imputation for correlated clinical variables.

    Uses sklearn IterativeImputer which models each feature with missing
    values as a function of other features — similar to the MICE algorithm
    (Multiple Imputation by Chained Equations) widely used in clinical trials.

    Particularly effective for correlated lab panels where, for example,
    a missing creatinine value can be predicted from BUN, eGFR, and age.

    Parameters
    ----------
    max_iter : int
        Maximum number of imputation rounds. Default 10.
    random_state : int
        Random seed for reproducibility.
    numeric_cols : list[str], optional
        Columns to impute. Defaults to all numeric columns.

    Examples
    --------
    >>> imputer = ClinicalMICEImputer(max_iter=10, random_state=42)
    >>> adlb_imputed = imputer.fit_transform(adlb_df)
    """

    def __init__(
        self,
        max_iter: int = 10,
        random_state: int = 42,
        numeric_cols: Optional[list[str]] = None,
    ) -> None:
        self.max_iter = max_iter
        self.random_state = random_state
        self.numeric_cols = numeric_cols
        self._imputer = IterativeImputer(
            max_iter=max_iter,
            random_state=random_state,
            skip_complete=True,
        )
        self._fitted_cols: list[str] = []

    def fit(self, df: pd.DataFrame) -> "ClinicalMICEImputer":
        cols = self._resolve_cols(df)
        self._imputer.fit(df[cols])
        self._fitted_cols = cols
        logger.info("ClinicalMICEImputer fitted: %d cols, %d iterations", len(cols), self.max_iter)
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        result = df.copy()
        imputed = self._imputer.transform(result[self._fitted_cols])
        result[self._fitted_cols] = imputed
        missing_before = df[self._fitted_cols].isna().sum().sum()
        logger.info("MICEImputer: %d missing values imputed", missing_before)
        return result

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        return self.fit(df).transform(df)

    def _resolve_cols(self, df: pd.DataFrame) -> list[str]:
        if self.numeric_cols:
            return [c for c in self.numeric_cols if c in df.columns]
        return df.select_dtypes(include=[np.number]).columns.tolist()


# ── Neural autoencoder imputer ────────────────────────────────────────────────

class NeuralImputer:
    """
    Autoencoder-based missing value imputation using TensorFlow/Keras.

    Architecture:
        Input (n_features) → Encoder [256 → 128 → 64] → Bottleneck
        → Decoder [64 → 128 → 256] → Output (n_features)

    Training strategy:
        1. Randomly mask additional values in complete rows (denoising)
        2. Train autoencoder to reconstruct original values
        3. At inference, pass rows with real missing values through
           the trained autoencoder to impute them

    This approach captures non-linear correlations between clinical
    variables that linear MICE cannot model — particularly useful for
    complex lab panels and biomarker data.

    Parameters
    ----------
    encoding_dim : int
        Size of the bottleneck layer. Default 64.
    epochs : int
        Training epochs. Default 50.
    batch_size : int
        Training batch size. Default 32.
    masking_fraction : float
        Fraction of values to randomly mask during training. Default 0.2.
    random_state : int
        Random seed. Default 42.
    numeric_cols : list[str], optional
        Columns to impute. Defaults to all numeric columns.

    Examples
    --------
    >>> imputer = NeuralImputer(epochs=50, encoding_dim=64)
    >>> adlb_imputed = imputer.fit_transform(adlb_df)
    """

    def __init__(
        self,
        encoding_dim: int = 64,
        epochs: int = 50,
        batch_size: int = 32,
        masking_fraction: float = 0.2,
        random_state: int = 42,
        numeric_cols: Optional[list[str]] = None,
        verbose: int = 0,
    ) -> None:
        self.encoding_dim    = encoding_dim
        self.epochs          = epochs
        self.batch_size      = batch_size
        self.masking_fraction = masking_fraction
        self.random_state    = random_state
        self.numeric_cols    = numeric_cols
        self.verbose         = verbose
        self._scaler         = StandardScaler()
        self._model          = None
        self._fitted_cols: list[str] = []
        np.random.seed(random_state)

    def fit(self, df: pd.DataFrame) -> "NeuralImputer":
        """
        Train the denoising autoencoder on complete-case rows.

        Parameters
        ----------
        df : pd.DataFrame
            Training data. Only rows with no missing values are used.

        Returns
        -------
        NeuralImputer
            Fitted imputer.
        """
        import tensorflow as tf
        from tensorflow import keras

        tf.random.set_seed(self.random_state)

        cols = self._resolve_cols(df)
        self._fitted_cols = cols

        # Use complete cases for training
        train_df = df[cols].dropna()
        if len(train_df) < 10:
            raise ValueError(
                f"NeuralImputer requires at least 10 complete-case rows. "
                f"Found {len(train_df)}. Consider SimpleImputer or MICEImputer."
            )

        X = self._scaler.fit_transform(train_df.values).astype(np.float32)
        n_features = X.shape[1]

        # ── Denoising: randomly mask values during training ───────────────────
        mask = np.random.binomial(1, 1 - self.masking_fraction, X.shape).astype(np.float32)
        X_masked = X * mask

        # ── Build autoencoder ─────────────────────────────────────────────────
        self._model = self._build_autoencoder(n_features, keras)

        self._model.fit(
            X_masked, X,
            epochs=self.epochs,
            batch_size=self.batch_size,
            validation_split=0.1,
            verbose=self.verbose,
            callbacks=[
                keras.callbacks.EarlyStopping(
                    patience=5,
                    restore_best_weights=True,
                    verbose=0,
                )
            ],
        )
        logger.info(
            "NeuralImputer trained: %d features, %d training rows, %d epochs",
            n_features, len(X), self.epochs,
        )
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Impute missing values using the trained autoencoder.

        Parameters
        ----------
        df : pd.DataFrame
            Dataset with missing values to impute.

        Returns
        -------
        pd.DataFrame
            Dataset with missing values replaced by autoencoder predictions.
        """
        if self._model is None:
            raise RuntimeError("NeuralImputer must be fitted before calling transform().")

        result = df.copy()
        cols   = self._fitted_cols
        data   = result[cols].values.astype(np.float32)

        # Temporarily fill NaN with 0 for autoencoder input
        nan_mask  = np.isnan(data)
        data_filled = np.where(nan_mask, 0.0, data)

        # Scale → reconstruct → inverse scale
        data_scaled      = self._scaler.transform(data_filled).astype(np.float32)
        reconstructed    = self._model.predict(data_scaled, verbose=0)
        reconstructed_inv = self._scaler.inverse_transform(reconstructed)

        # Only fill positions that were originally NaN
        imputed = np.where(nan_mask, reconstructed_inv, data)
        result[cols] = imputed

        n_imputed = nan_mask.sum()
        logger.info("NeuralImputer: %d missing values imputed via autoencoder", n_imputed)
        return result

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        return self.fit(df).transform(df)

    def _build_autoencoder(self, n_features: int, keras) -> "keras.Model":
        """Build a symmetric denoising autoencoder."""
        inputs = keras.Input(shape=(n_features,), name="input")

        # Encoder
        x = keras.layers.Dense(min(256, n_features * 4), activation="relu", name="enc1")(inputs)
        x = keras.layers.BatchNormalization()(x)
        x = keras.layers.Dropout(0.2)(x)
        x = keras.layers.Dense(min(128, n_features * 2), activation="relu", name="enc2")(x)
        x = keras.layers.BatchNormalization()(x)
        encoded = keras.layers.Dense(self.encoding_dim, activation="relu", name="bottleneck")(x)

        # Decoder
        x = keras.layers.Dense(min(128, n_features * 2), activation="relu", name="dec1")(encoded)
        x = keras.layers.BatchNormalization()(x)
        x = keras.layers.Dense(min(256, n_features * 4), activation="relu", name="dec2")(x)
        outputs = keras.layers.Dense(n_features, activation="linear", name="output")(x)

        model = keras.Model(inputs=inputs, outputs=outputs, name="clinical_autoencoder")
        model.compile(optimizer="adam", loss="mse", metrics=["mae"])
        return model

    def _resolve_cols(self, df: pd.DataFrame) -> list[str]:
        if self.numeric_cols:
            return [c for c in self.numeric_cols if c in df.columns]
        return df.select_dtypes(include=[np.number]).columns.tolist()


# ── Imputer factory ───────────────────────────────────────────────────────────

def get_imputer(
    method: str = "mice",
    **kwargs,
) -> ClinicalSimpleImputer | ClinicalMICEImputer | NeuralImputer:
    """
    Factory function to instantiate the appropriate imputer.

    Parameters
    ----------
    method : str
        One of 'simple', 'mice', or 'neural'.
    **kwargs
        Additional keyword arguments passed to the imputer constructor.

    Returns
    -------
    Imputer instance.

    Examples
    --------
    >>> imputer = get_imputer("neural", epochs=100, encoding_dim=32)
    >>> adlb_imputed = imputer.fit_transform(adlb_df)
    """
    methods = {
        "simple": ClinicalSimpleImputer,
        "mice":   ClinicalMICEImputer,
        "neural": NeuralImputer,
    }
    if method not in methods:
        raise ValueError(f"Unknown method '{method}'. Choose from: {list(methods.keys())}")

    imputer = methods[method](**kwargs)
    logger.info("Instantiated %s imputer", method)
    return imputer
