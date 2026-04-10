Now the very last test file!

Click the tests folder
Click "Add file" → "Create new file"
Type in the filename box:

test_models.py

Paste this code:

python"""
tests/test_models.py
─────────────────────
Unit tests for survival analysis and AE signal detection models.

Uses synthetic ADTTE and ADAE datasets — no real patient data required.
TensorFlow models use minimal architecture for fast CI execution.

Author : Girish Rajeev
         Clinical Data Scientist | Regulatory Standards Leader | AI/ML Solution Engineer
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.models.survival import (
    KaplanMeierModel,
    CoxPHModel,
    KMResult,
    LogRankResult,
    CoxPHResult,
)
from src.models.ae_signal import AESignalDetector, AESignalResult


STUDY_ID = "DRUGX-2024-003"


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture
def adtte_df() -> pd.DataFrame:
    """Synthetic ADTTE dataset with OS endpoint."""
    np.random.seed(42)
    n = 40
    arms = ["Drug X"] * 20 + ["Placebo"] * 20
    # Drug X arm: longer survival times
    aval = np.concatenate([
        np.random.exponential(300, 20),
        np.random.exponential(180, 20),
    ])
    cnsr = np.random.binomial(1, 0.3, n)
    return pd.DataFrame({
        "USUBJID":  [f"{STUDY_ID}-01-{i:03d}" for i in range(n)],
        "TRTA":     arms,
        "TRT01AN":  [1] * 20 + [2] * 20,
        "PARAMCD":  ["OS"] * n,
        "AVAL":     aval,
        "AVALU":    ["DAYS"] * n,
        "CNSR":     cnsr,
    })


@pytest.fixture
def adsl_df() -> pd.DataFrame:
    """Synthetic ADSL dataset."""
    np.random.seed(42)
    n = 40
    return pd.DataFrame({
        "USUBJID":  [f"{STUDY_ID}-01-{i:03d}" for i in range(n)],
        "TRTA":     ["Drug X"] * 20 + ["Placebo"] * 20,
        "TRT01A":   ["Drug X"] * 20 + ["Placebo"] * 20,
        "TRT01AN":  [1] * 20 + [2] * 20,
        "AGE":      np.random.randint(35, 75, n),
        "SEX":      np.random.choice(["M", "F"], n),
        "SAFFL":    ["Y"] * n,
        "ITTFL":    ["Y"] * n,
        "TRTSDT":   pd.to_datetime(["2024-01-17"] * n),
    })


@pytest.fixture
def adae_df() -> pd.DataFrame:
    """Synthetic ADAE dataset with body system and grade info."""
    np.random.seed(42)
    records = []
    body_systems = [
        "GASTROINTESTINAL DISORDERS",
        "GENERAL DISORDERS",
        "NERVOUS SYSTEM DISORDERS",
        "SKIN AND SUBCUTANEOUS TISSUE DISORDERS",
    ]
    ae_terms = ["NAUSEA", "FATIGUE", "HEADACHE", "RASH"]
    for i in range(40):
        usubjid = f"{STUDY_ID}-01-{i:03d}"
        trta = "Drug X" if i < 20 else "Placebo"
        n_aes = np.random.randint(0, 5)
        for j in range(n_aes):
            idx = np.random.randint(0, 4)
            records.append({
                "USUBJID":   usubjid,
                "TRTA":      trta,
                "TRTEMFL":   "Y",
                "AEDECOD":   ae_terms[idx],
                "AEBODSYS":  body_systems[idx],
                "AESEV":     np.random.choice(["MILD", "MODERATE", "SEVERE"]),
                "ATOXGR":    np.random.randint(1, 4),
                "ASER":      np.random.choice(["Y", "N"], p=[0.1, 0.9]),
                "AEREL":     np.random.choice(["Y", "N"], p=[0.6, 0.4]),
            })
    return pd.DataFrame(records)


# ── KaplanMeierModel tests ────────────────────────────────────────────────────

class TestKaplanMeierModel:

    def test_fit_returns_dict(self, adtte_df):
        km = KaplanMeierModel()
        results = km.fit(adtte_df)
        assert isinstance(results, dict)

    def test_both_arms_present(self, adtte_df):
        km = KaplanMeierModel()
        results = km.fit(adtte_df)
        assert "Drug X" in results
        assert "Placebo" in results

    def test_km_result_type(self, adtte_df):
        km = KaplanMeierModel()
        results = km.fit(adtte_df)
        assert isinstance(results["Drug X"], KMResult)

    def test_subject_counts(self, adtte_df):
        km = KaplanMeierModel()
        results = km.fit(adtte_df)
        assert results["Drug X"].n_subjects == 20
        assert results["Placebo"].n_subjects == 20

    def test_events_within_range(self, adtte_df):
        km = KaplanMeierModel()
        results = km.fit(adtte_df)
        for arm_result in results.values():
            assert 0 <= arm_result.n_events <= arm_result.n_subjects

    def test_survival_prob_between_0_and_1(self, adtte_df):
        km = KaplanMeierModel()
        results = km.fit(adtte_df)
        for arm_result in results.values():
            probs = arm_result.survival_prob
            assert all(0 <= p <= 1 for p in probs)

    def test_timeline_non_negative(self, adtte_df):
        km = KaplanMeierModel()
        results = km.fit(adtte_df)
        for arm_result in results.values():
            assert all(t >= 0 for t in arm_result.timeline)

    def test_to_dict_keys(self, adtte_df):
        km = KaplanMeierModel()
        results = km.fit(adtte_df)
        d = results["Drug X"].to_dict()
        assert "arm" in d
        assert "median_survival" in d
        assert "n_events" in d

    def test_logrank_returns_result(self, adtte_df):
        km = KaplanMeierModel()
        km.fit(adtte_df)
        lr = km.logrank_test(adtte_df, arm1="Drug X", arm2="Placebo")
        assert isinstance(lr, LogRankResult)

    def test_logrank_p_value_range(self, adtte_df):
        km = KaplanMeierModel()
        km.fit(adtte_df)
        lr = km.logrank_test(adtte_df, arm1="Drug X", arm2="Placebo")
        assert 0 <= lr.p_value <= 1

    def test_logrank_summary_string(self, adtte_df):
        km = KaplanMeierModel()
        km.fit(adtte_df)
        lr = km.logrank_test(adtte_df, arm1="Drug X", arm2="Placebo")
        summary = lr.summary()
        assert "Log-rank" in summary
        assert "p=" in summary

    def test_get_survival_table(self, adtte_df):
        km = KaplanMeierModel()
        km.fit(adtte_df)
        table = km.get_survival_table("Drug X")
        assert isinstance(table, pd.DataFrame)
        assert len(table) > 0

    def test_get_survival_table_unknown_arm_raises(self, adtte_df):
        km = KaplanMeierModel()
        km.fit(adtte_df)
        with pytest.raises(KeyError):
            km.get_survival_table("Unknown Arm")


# ── CoxPHModel tests ──────────────────────────────────────────────────────────

class TestCoxPHModel:

    def test_fit_returns_result(self, adtte_df, adsl_df):
        merged = adtte_df.merge(
            adsl_df[["USUBJID", "AGE", "TRT01AN"]],
            on="USUBJID", how="left",
        )
        cox = CoxPHModel()
        result = cox.fit(merged, covariates=["TRT01AN", "AGE", "AVAL"])
        assert isinstance(result, CoxPHResult)

    def test_concordance_index_range(self, adtte_df, adsl_df):
        merged = adtte_df.merge(
            adsl_df[["USUBJID", "AGE", "TRT01AN"]],
            on="USUBJID", how="left",
        )
        cox = CoxPHModel()
        result = cox.fit(merged, covariates=["TRT01AN", "AGE", "AVAL"])
        assert 0 <= result.concordance_index <= 1

    def test_hazard_ratios_dataframe(self, adtte_df, adsl_df):
        merged = adtte_df.merge(
            adsl_df[["USUBJID", "AGE", "TRT01AN"]],
            on="USUBJID", how="left",
        )
        cox = CoxPHModel()
        result = cox.fit(merged, covariates=["TRT01AN", "AGE", "AVAL"])
        assert isinstance(result.hazard_ratios, pd.DataFrame)

    def test_summary_string(self, adtte_df, adsl_df):
        merged = adtte_df.merge(
            adsl_df[["USUBJID", "AGE", "TRT01AN"]],
            on="USUBJID", how="left",
        )
        cox = CoxPHModel()
        result = cox.fit(merged, covariates=["TRT01AN", "AGE", "AVAL"])
        assert "C-index" in result.summary()


# ── AESignalDetector tests ────────────────────────────────────────────────────

class TestAESignalDetector:

    def test_fit_detect_returns_result(self, adae_df, adsl_df):
        detector = AESignalDetector(
            encoding_dim=4,
            epochs=2,
            batch_size=8,
            random_state=42,
        )
        result = detector.fit_detect(adae_df, adsl_df)
        assert isinstance(result, AESignalResult)

    def test_all_subjects_scored(self, adae_df, adsl_df):
        detector = AESignalDetector(encoding_dim=4, epochs=2, batch_size=8)
        result = detector.fit_detect(adae_df, adsl_df)
        assert len(result.usubjid) == len(adsl_df)

    def test_reconstruction_errors_non_negative(self, adae_df, adsl_df):
        detector = AESignalDetector(encoding_dim=4, epochs=2, batch_size=8)
        result = detector.fit_detect(adae_df, adsl_df)
        assert all(e >= 0 for e in result.reconstruction_error)

    def test_anomaly_count_matches_contamination(self, adae_df, adsl_df):
        contamination = 0.10
        detector = AESignalDetector(
            encoding_dim=4,
            epochs=2,
            batch_size=8,
            contamination=contamination,
        )
        result = detector.fit_detect(adae_df, adsl_df)
        expected = int(np.ceil(len(adsl_df) * contamination))
        assert result.n_anomalies <= expected + 2

    def test_to_dataframe(self, adae_df, adsl_df):
        detector = AESignalDetector(encoding_dim=4, epochs=2, batch_size=8)
        result = detector.fit_detect(adae_df, adsl_df)
        df = result.to_dataframe()
        assert isinstance(df, pd.DataFrame)
        assert "USUBJID" in df.columns
        assert "RECONSTRUCTION_ERROR" in df.columns
        assert "ANOMALY_FLAG" in df.columns

    def test_summary_keys(self, adae_df, adsl_df):
        detector = AESignalDetector(encoding_dim=4, epochs=2, batch_size=8)
        result = detector.fit_detect(adae_df, adsl_df)
        summary = result.summary()
        assert "n_subjects" in summary
        assert "n_anomalies" in summary
        assert "anomaly_rate" in summary
        assert "threshold" in summary

    def test_reconstruct_before_fit_raises(self, adae_df, adsl_df):
        detector = AESignalDetector()
        with pytest.raises(RuntimeError, match="must be trained"):
            detector.reconstruct(adae_df, adsl_df)

    def test_feature_matrix_shape(self, adae_df, adsl_df):
        feature_df = AESignalDetector._build_feature_matrix(adae_df, adsl_df)
        assert isinstance(feature_df, pd.DataFrame)
        assert len(feature_df) == len(adsl_df["USUBJID"].unique())
        assert "TOTAL_TEAE" in feature_df.columns

    def test_sorted_by_error_descending(self, adae_df, adsl_df):
        detector = AESignalDetector(encoding_dim=4, epochs=2, batch_size=8)
        result = detector.fit_detect(adae_df, adsl_df)
        df = result.to_dataframe()
        errors = df["RECONSTRUCTION_ERROR"].tolist()
        assert errors == sorted(errors, reverse=True)
