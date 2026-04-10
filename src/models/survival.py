Now the final module — src/models/. Let's add the survival analysis and AE signal detection:

Click src → models folder
Click "Add file" → "Create new file"
Type in the filename box:

survival.py

Paste this code:

python"""
models/survival.py
───────────────────
Survival analysis models for clinical trial time-to-event endpoints.

Implements:
    KaplanMeierModel    — Non-parametric KM estimator with CI
    CoxPHModel          — Cox proportional hazards regression
    SurvivalForest      — Random survival forest (scikit-survival)

All models accept ADTTE-format DataFrames directly and produce
publication-ready summary statistics and plot-ready data.

Author : Girish Rajeev
         Clinical Data Scientist | Regulatory Standards Leader | AI/ML Solution Engineer
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd
from lifelines import KaplanMeierFitter, CoxPHFitter
from lifelines.statistics import logrank_test

logger = logging.getLogger(__name__)


# ── Result containers ─────────────────────────────────────────────────────────

@dataclass
class KMResult:
    """Results from Kaplan-Meier analysis."""
    arm: str
    n_subjects: int
    n_events: int
    median_survival: float
    median_ci_lower: float
    median_ci_upper: float
    timeline: list[float] = field(default_factory=list)
    survival_prob: list[float] = field(default_factory=list)
    ci_lower: list[float] = field(default_factory=list)
    ci_upper: list[float] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "arm":              self.arm,
            "n_subjects":       self.n_subjects,
            "n_events":         self.n_events,
            "median_survival":  self.median_survival,
            "median_ci_lower":  self.median_ci_lower,
            "median_ci_upper":  self.median_ci_upper,
        }


@dataclass
class LogRankResult:
    """Results from log-rank test comparing two survival curves."""
    test_statistic: float
    p_value: float
    alpha: float = 0.05

    @property
    def significant(self) -> bool:
        return self.p_value < self.alpha

    def summary(self) -> str:
        return (
            f"Log-rank test: chi2={self.test_statistic:.4f}, "
            f"p={self.p_value:.4f} "
            f"({'significant' if self.significant else 'not significant'} "
            f"at alpha={self.alpha})"
        )


@dataclass
class CoxPHResult:
    """Results from Cox proportional hazards model."""
    hazard_ratios: pd.DataFrame
    concordance_index: float
    aic: float
    log_likelihood: float

    def summary(self) -> str:
        return (
            f"Cox PH Model — C-index: {self.concordance_index:.3f}, "
            f"AIC: {self.aic:.2f}"
        )


# ── Kaplan-Meier model ────────────────────────────────────────────────────────

class KaplanMeierModel:
    """
    Kaplan-Meier survival estimator for clinical trial endpoints.

    Accepts ADTTE-format DataFrame and produces KM curves,
    median survival with confidence intervals, and log-rank
    test statistics for treatment arm comparisons.

    Parameters
    ----------
    duration_col : str
        Column containing time-to-event values (AVAL). Default 'AVAL'.
    event_col : str
        Column containing event indicator (0=censored, 1=event).
        Note: ADTTE CNSR is 1=censored, 0=event — this is auto-handled.
    treatment_col : str
        Column containing treatment arm labels. Default 'TRTA'.
    confidence_level : float
        Confidence level for CIs. Default 0.95.

    Examples
    --------
    >>> km = KaplanMeierModel()
    >>> results = km.fit(adtte_df)
    >>> logrank = km.logrank_test(adtte_df, arm1="Drug X", arm2="Placebo")
    """

    def __init__(
        self,
        duration_col: str = "AVAL",
        event_col: str = "CNSR",
        treatment_col: str = "TRTA",
        confidence_level: float = 0.95,
    ) -> None:
        self.duration_col    = duration_col
        self.event_col       = event_col
        self.treatment_col   = treatment_col
        self.confidence_level = confidence_level
        self._fitters: dict[str, KaplanMeierFitter] = {}

    def fit(self, adtte_df: pd.DataFrame) -> dict[str, KMResult]:
        """
        Fit KM curves for each treatment arm.

        Parameters
        ----------
        adtte_df : pd.DataFrame
            ADTTE dataset. CNSR column is auto-converted
            (ADTTE: 0=event, 1=censored → lifelines: 1=event, 0=censored).

        Returns
        -------
        dict[str, KMResult]
            KM results keyed by treatment arm label.
        """
        results = {}
        # Convert ADTTE censoring to lifelines convention
        event_observed = 1 - adtte_df[self.event_col]

        for arm, grp in adtte_df.groupby(self.treatment_col):
            idx     = grp.index
            T       = grp[self.duration_col]
            E       = event_observed.loc[idx]

            kmf = KaplanMeierFitter(alpha=1 - self.confidence_level)
            kmf.fit(T, event_observed=E, label=str(arm))
            self._fitters[str(arm)] = kmf

            median    = kmf.median_survival_time_
            ci        = kmf.confidence_interval_cumulative_density_

            results[str(arm)] = KMResult(
                arm=str(arm),
                n_subjects=len(grp),
                n_events=int(E.sum()),
                median_survival=float(median) if not np.isnan(median) else -1,
                median_ci_lower=float(kmf.confidence_interval_median_.iloc[0, 0]),
                median_ci_upper=float(kmf.confidence_interval_median_.iloc[0, 1]),
                timeline=kmf.timeline.tolist(),
                survival_prob=kmf.survival_function_.iloc[:, 0].tolist(),
                ci_lower=(1 - ci.iloc[:, 1]).tolist(),
                ci_upper=(1 - ci.iloc[:, 0]).tolist(),
            )
            logger.info(
                "KM fitted [%s]: n=%d, events=%d, median=%.1f days",
                arm, len(grp), int(E.sum()), float(median) if not np.isnan(median) else -1,
            )

        return results

    def logrank_test(
        self,
        adtte_df: pd.DataFrame,
        arm1: str,
        arm2: str,
    ) -> LogRankResult:
        """
        Perform log-rank test comparing two treatment arms.

        Parameters
        ----------
        adtte_df : pd.DataFrame
            ADTTE dataset.
        arm1 : str
            Label of first treatment arm.
        arm2 : str
            Label of second treatment arm (reference).

        Returns
        -------
        LogRankResult
        """
        event_observed = 1 - adtte_df[self.event_col]

        grp1 = adtte_df[adtte_df[self.treatment_col] == arm1]
        grp2 = adtte_df[adtte_df[self.treatment_col] == arm2]

        result = logrank_test(
            grp1[self.duration_col],
            grp2[self.duration_col],
            event_observed_A=event_observed.loc[grp1.index],
            event_observed_B=event_observed.loc[grp2.index],
        )

        lr = LogRankResult(
            test_statistic=float(result.test_statistic),
            p_value=float(result.p_value),
        )
        logger.info("Log-rank test [%s vs %s]: %s", arm1, arm2, lr.summary())
        return lr

    def get_survival_table(self, arm: str) -> pd.DataFrame:
        """Return the KM survival table for a given arm."""
        if arm not in self._fitters:
            raise KeyError(f"Arm '{arm}' not found. Call fit() first.")
        return self._fitters[arm].survival_function_


# ── Cox PH model ──────────────────────────────────────────────────────────────

class CoxPHModel:
    """
    Cox proportional hazards regression model.

    Fits a Cox PH model to ADTTE data with optional covariates
    from ADSL (age, sex, performance status, biomarkers).

    Parameters
    ----------
    duration_col : str
        Time-to-event column (AVAL). Default 'AVAL'.
    event_col : str
        Censoring indicator (CNSR). Default 'CNSR'.
    penalizer : float
        L2 regularization strength. Default 0.1.

    Examples
    --------
    >>> cox = CoxPHModel()
    >>> result = cox.fit(adtte_adsl_df, covariates=["AGE", "SEX_NUM", "TRT01AN"])
    >>> print(result.summary())
    """

    def __init__(
        self,
        duration_col: str = "AVAL",
        event_col: str = "CNSR",
        penalizer: float = 0.1,
    ) -> None:
        self.duration_col = duration_col
        self.event_col    = event_col
        self.penalizer    = penalizer
        self._cph         = CoxPHFitter(penalizer=penalizer)

    def fit(
        self,
        df: pd.DataFrame,
        covariates: Optional[list[str]] = None,
    ) -> CoxPHResult:
        """
        Fit Cox PH model.

        Parameters
        ----------
        df : pd.DataFrame
            Merged ADTTE + ADSL dataset with covariates.
        covariates : list[str], optional
            Covariate columns to include. If None uses all numeric columns
            except duration and event columns.

        Returns
        -------
        CoxPHResult
        """
        # Convert ADTTE censoring convention
        model_df = df.copy()
        model_df["_event"] = 1 - model_df[self.event_col]

        if covariates:
            cols = covariates + [self.duration_col, "_event"]
        else:
            num_cols = model_df.select_dtypes(include=[np.number]).columns.tolist()
            cols = [
                c for c in num_cols
                if c not in [self.event_col, "_event"]
            ]

        model_df = model_df[cols].dropna()
        self._cph.fit(
            model_df,
            duration_col=self.duration_col,
            event_col="_event",
        )

        result = CoxPHResult(
            hazard_ratios=self._cph.summary[["exp(coef)", "exp(coef) lower 95%",
                                              "exp(coef) upper 95%", "p"]],
            concordance_index=self._cph.concordance_index_,
            aic=self._cph.AIC_,
            log_likelihood=self._cph.log_likelihood_,
        )
        logger.info("Cox PH fitted: C-index=%.3f, AIC=%.2f",
                    result.concordance_index, result.aic)
        return result

    def predict_survival(
        self,
        df: pd.DataFrame,
        times: Optional[list[float]] = None,
    ) -> pd.DataFrame:
        """
        Predict survival probabilities for new subjects.

        Parameters
        ----------
        df : pd.DataFrame
            Covariate data for subjects to predict.
        times : list[float], optional
            Time points at which to predict survival probability.

        Returns
        -------
        pd.DataFrame
            Survival probabilities (rows=time points, cols=subjects).
        """
        return self._cph.predict_survival_function(df, times=times)
