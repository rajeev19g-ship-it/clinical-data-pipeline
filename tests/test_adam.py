Now let's add the ADaM tests:

Click the tests folder from the main repo page
Click "Add file" → "Create new file"
Type in the filename box:

test_adam.py

Paste this code:

python"""
tests/test_adam.py
───────────────────
Unit tests for ADaM derivations and ML imputation modules.

Uses synthetic SDTM-style data — no real patient data required.
Neural imputer tests use a minimal architecture for fast CI execution.

Author : Girish Rajeev
         Clinical Data Scientist | Regulatory Standards Leader | AI/ML Solution Engineer
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.adam.derivations import ADSL, ADAE, ADTTE, ADLB
from src.adam.imputation import (
    ClinicalSimpleImputer,
    ClinicalMICEImputer,
    NeuralImputer,
    get_imputer,
)

STUDY_ID = "DRUGX-2024-003"


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture
def sdtm_dm() -> pd.DataFrame:
    """Synthetic SDTM DM dataset."""
    return pd.DataFrame({
        "STUDYID":  [STUDY_ID] * 4,
        "DOMAIN":   ["DM"] * 4,
        "USUBJID":  [f"{STUDY_ID}-01-00{i}" for i in range(1, 5)],
        "SUBJID":   ["001", "002", "003", "004"],
        "SITEID":   ["01", "01", "02", "02"],
        "AGE":      [45, 67, 38, 72],
        "AGEU":     ["YEARS"] * 4,
        "SEX":      ["M", "F", "M", "F"],
        "RACE":     ["WHITE", "ASIAN", "WHITE", "BLACK OR AFRICAN AMERICAN"],
        "ETHNIC":   ["NOT HISPANIC OR LATINO"] * 4,
        "ARM":      ["Drug X", "Placebo", "Drug X", "Placebo"],
        "ARMCD":    ["DRUGX", "PBO", "DRUGX", "PBO"],
        "RFSTDTC":  ["2024-01-17", "2024-01-18", "2024-01-22", "2024-01-23"],
        "RFICDTC":  ["2024-01-10", "2024-01-11", "2024-01-15", "2024-01-16"],
    })


@pytest.fixture
def adsl_df(sdtm_dm) -> pd.DataFrame:
    """Derived ADSL from synthetic DM."""
    return ADSL(study_id=STUDY_ID).derive(sdtm_dm)


@pytest.fixture
def sdtm_ae() -> pd.DataFrame:
    """Synthetic SDTM AE dataset."""
    return pd.DataFrame({
        "STUDYID":  [STUDY_ID] * 4,
        "DOMAIN":   ["AE"] * 4,
        "USUBJID":  [
            f"{STUDY_ID}-01-001",
            f"{STUDY_ID}-01-001",
            f"{STUDY_ID}-01-002",
            f"{STUDY_ID}-02-003",
        ],
        "AESEQ":    [1, 2, 1, 1],
        "AETERM":   ["NAUSEA", "FATIGUE", "HEADACHE", "RASH"],
        "AEDECOD":  ["NAUSEA", "FATIGUE", "HEADACHE", "RASH"],
        "AESTDTC":  ["2024-02-01", "2024-02-10", "2024-02-05", "2024-02-08"],
        "AEENDTC":  ["2024-02-03", "2024-02-15", "2024-02-06", "2024-02-12"],
        "AESEV":    ["MILD", "MODERATE", "MILD", "SEVERE"],
        "AESER":    ["N", "N", "N", "Y"],
        "AEREL":    ["Y", "N", "Y", "Y"],
        "AEOUT":    ["RECOVERED/RESOLVED"] * 4,
    })


@pytest.fixture
def sdtm_lb() -> pd.DataFrame:
    """Synthetic SDTM LB dataset with some missing values."""
    return pd.DataFrame({
        "STUDYID":   [STUDY_ID] * 6,
        "DOMAIN":    ["LB"] * 6,
        "USUBJID":   [
            f"{STUDY_ID}-01-001",
            f"{STUDY_ID}-01-001",
            f"{STUDY_ID}-01-002",
            f"{STUDY_ID}-01-002",
            f"{STUDY_ID}-02-003",
            f"{STUDY_ID}-02-003",
        ],
        "LBSEQ":     [1, 2, 1, 2, 1, 2],
        "LBTESTCD":  ["HGB", "CREAT", "HGB", "CREAT", "HGB", "CREAT"],
        "LBTEST":    ["Hemoglobin", "Creatinine"] * 3,
        "LBCAT":     ["HEMATOLOGY", "CHEMISTRY"] * 3,
        "LBORRES":   ["13.5", "0.9", None, "1.1", "11.2", "0.8"],
        "LBORRESU":  ["g/dL", "mg/dL"] * 3,
        "LBSTRESN":  [13.5, 0.9, None, 1.1, 11.2, 0.8],
        "LBSTRESU":  ["g/dL", "mg/dL"] * 3,
        "LBSTNRLO":  [12.0, 0.6] * 3,
        "LBSTNRHI":  [17.0, 1.2] * 3,
        "LBNRIND":   ["NORMAL", "NORMAL", None, "NORMAL", "LOW", "NORMAL"],
        "LBDTC":     ["2024-01-17"] * 6,
        "VISIT":     ["SCREENING"] * 6,
    })


@pytest.fixture
def events_df() -> pd.DataFrame:
    """Synthetic OS events dataset."""
    return pd.DataFrame({
        "USUBJID": [
            f"{STUDY_ID}-01-001",
            f"{STUDY_ID}-01-002",
            f"{STUDY_ID}-02-003",
            f"{STUDY_ID}-02-004",
        ],
        "death_date":        ["2024-08-15", None, "2024-10-01", None],
        "last_contact_date": ["2024-08-15", "2024-12-01", "2024-10-01", "2024-11-15"],
    })


@pytest.fixture
def lab_data_with_missing() -> pd.DataFrame:
    """Synthetic numeric lab panel with structured missing values."""
    np.random.seed(42)
    n = 100
    df = pd.DataFrame({
        "HGB":    np.random.normal(13.5, 1.5, n),
        "CREAT":  np.random.normal(0.9,  0.2, n),
        "ALT":    np.random.normal(25.0, 8.0, n),
        "AST":    np.random.normal(22.0, 7.0, n),
        "BILI":   np.random.normal(0.8,  0.3, n),
        "WBC":    np.random.normal(6.5,  1.5, n),
        "PLT":    np.random.normal(220,  40,  n),
        "SODIUM": np.random.normal(140,  3.0, n),
    })
    # Introduce ~15% missingness
    mask = np.random.binomial(1, 0.15, df.shape).astype(bool)
    df[mask] = np.nan
    return df


# ── ADSL tests ────────────────────────────────────────────────────────────────

class TestADSL:

    def test_returns_dataframe(self, sdtm_dm):
        result = ADSL(study_id=STUDY_ID).derive(sdtm_dm)
        assert isinstance(result, pd.DataFrame)

    def test_record_count(self, sdtm_dm):
        result = ADSL(study_id=STUDY_ID).derive(sdtm_dm)
        assert len(result) == 4

    def test_population_flags_present(self, adsl_df):
        for flag in ["ITTFL", "SAFFL", "PPROTFL"]:
            assert flag in adsl_df.columns

    def test_all_itt_flagged(self, adsl_df):
        assert all(adsl_df["ITTFL"] == "Y")

    def test_age_grouping(self, adsl_df):
        assert "AGEGR1" in adsl_df.columns
        assert set(adsl_df["AGEGR1"]).issubset({"<65", "65-74", ">=75"})

    def test_treatment_variables(self, adsl_df):
        assert "TRT01P" in adsl_df.columns
        assert "TRT01A" in adsl_df.columns
        assert "TRT01PN" in adsl_df.columns

    def test_missing_cols_raises(self, sdtm_dm):
        with pytest.raises(ValueError, match="missing required columns"):
            ADSL(study_id=STUDY_ID).derive(sdtm_dm.drop(columns=["AGE"]))


# ── ADAE tests ────────────────────────────────────────────────────────────────

class TestADAE:

    def test_returns_dataframe(self, sdtm_ae, adsl_df):
        result = ADAE(study_id=STUDY_ID).derive(sdtm_ae, adsl_df)
        assert isinstance(result, pd.DataFrame)

    def test_trtemfl_populated(self, sdtm_ae, adsl_df):
        result = ADAE(study_id=STUDY_ID).derive(sdtm_ae, adsl_df)
        assert "TRTEMFL" in result.columns
        assert set(result["TRTEMFL"].unique()).issubset({"Y", ""})

    def test_aestdy_calculated(self, sdtm_ae, adsl_df):
        result = ADAE(study_id=STUDY_ID).derive(sdtm_ae, adsl_df)
        assert "AESTDY" in result.columns

    def test_atoxgr_numeric(self, sdtm_ae, adsl_df):
        result = ADAE(study_id=STUDY_ID).derive(sdtm_ae, adsl_df)
        assert pd.api.types.is_integer_dtype(result["ATOXGR"])
        assert result["ATOXGR"].max() <= 5

    def test_severe_ae_grade(self, sdtm_ae, adsl_df):
        result = ADAE(study_id=STUDY_ID).derive(sdtm_ae, adsl_df)
        severe = result[result["AESEV"] == "SEVERE"]
        assert all(severe["ATOXGR"] == 3)

    def test_serious_flag(self, sdtm_ae, adsl_df):
        result = ADAE(study_id=STUDY_ID).derive(sdtm_ae, adsl_df)
        assert "ASER" in result.columns
        assert set(result["ASER"].unique()).issubset({"Y", "N"})


# ── ADTTE tests ───────────────────────────────────────────────────────────────

class TestADTTE:

    def test_os_returns_dataframe(self, adsl_df, events_df):
        result = ADTTE(study_id=STUDY_ID).derive_os(adsl_df, events_df)
        assert isinstance(result, pd.DataFrame)

    def test_os_paramcd(self, adsl_df, events_df):
        result = ADTTE(study_id=STUDY_ID).derive_os(adsl_df, events_df)
        assert all(result["PARAMCD"] == "OS")

    def test_cnsr_binary(self, adsl_df, events_df):
        result = ADTTE(study_id=STUDY_ID).derive_os(adsl_df, events_df)
        assert set(result["CNSR"].unique()).issubset({0, 1})

    def test_aval_positive(self, adsl_df, events_df):
        result = ADTTE(study_id=STUDY_ID).derive_os(adsl_df, events_df)
        assert all(result["AVAL"].dropna() >= 0)

    def test_avalu_days(self, adsl_df, events_df):
        result = ADTTE(study_id=STUDY_ID).derive_os(adsl_df, events_df)
        assert all(result["AVALU"] == "DAYS")

    def test_death_not_censored(self, adsl_df, events_df):
        result = ADTTE(study_id=STUDY_ID).derive_os(adsl_df, events_df)
        deaths = result[result["EVNTDESC"] == "DEATH"]
        assert all(deaths["CNSR"] == 0)


# ── ADLB tests ────────────────────────────────────────────────────────────────

class TestADLB:

    def test_returns_dataframe(self, sdtm_lb, adsl_df):
        result = ADLB(study_id=STUDY_ID).derive(sdtm_lb, adsl_df)
        assert isinstance(result, pd.DataFrame)

    def test_baseline_flag(self, sdtm_lb, adsl_df):
        result = ADLB(study_id=STUDY_ID).derive(sdtm_lb, adsl_df)
        assert "ABLFL" in result.columns
        assert "Y" in result["ABLFL"].values

    def test_base_populated(self, sdtm_lb, adsl_df):
        result = ADLB(study_id=STUDY_ID).derive(sdtm_lb, adsl_df)
        assert "BASE" in result.columns

    def test_chg_derived(self, sdtm_lb, adsl_df):
        result = ADLB(study_id=STUDY_ID).derive(sdtm_lb, adsl_df)
        assert "CHG" in result.columns
        baseline_rows = result[result["ABLFL"] == "Y"]
        assert all(baseline_rows["CHG"].fillna(0) == 0)

    def test_pchg_derived(self, sdtm_lb, adsl_df):
        result = ADLB(study_id=STUDY_ID).derive(sdtm_lb, adsl_df)
        assert "PCHG" in result.columns


# ── Imputation tests ──────────────────────────────────────────────────────────

class TestClinicalSimpleImputer:

    def test_no_missing_after_imputation(self, lab_data_with_missing):
        imputer = ClinicalSimpleImputer(strategy="median")
        result = imputer.fit_transform(lab_data_with_missing)
        assert result.isna().sum().sum() == 0

    def test_shape_preserved(self, lab_data_with_missing):
        imputer = ClinicalSimpleImputer(strategy="mean")
        result = imputer.fit_transform(lab_data_with_missing)
        assert result.shape == lab_data_with_missing.shape

    def test_specific_cols(self, lab_data_with_missing):
        imputer = ClinicalSimpleImputer(
            strategy="median",
            numeric_cols=["HGB", "CREAT"],
        )
        result = imputer.fit_transform(lab_data_with_missing)
        assert result[["HGB", "CREAT"]].isna().sum().sum() == 0


class TestClinicalMICEImputer:

    def test_no_missing_after_imputation(self, lab_data_with_missing):
        imputer = ClinicalMICEImputer(max_iter=3, random_state=42)
        result = imputer.fit_transform(lab_data_with_missing)
        assert result.isna().sum().sum() == 0

    def test_shape_preserved(self, lab_data_with_missing):
        imputer = ClinicalMICEImputer(max_iter=3)
        result = imputer.fit_transform(lab_data_with_missing)
        assert result.shape == lab_data_with_missing.shape


class TestNeuralImputer:

    def test_no_missing_after_imputation(self, lab_data_with_missing):
        imputer = NeuralImputer(
            encoding_dim=8,
            epochs=3,
            batch_size=16,
            random_state=42,
        )
        result = imputer.fit_transform(lab_data_with_missing)
        assert result.isna().sum().sum() == 0

    def test_shape_preserved(self, lab_data_with_missing):
        imputer = NeuralImputer(encoding_dim=8, epochs=3)
        result = imputer.fit_transform(lab_data_with_missing)
        assert result.shape == lab_data_with_missing.shape

    def test_transform_before_fit_raises(self, lab_data_with_missing):
        imputer = NeuralImputer()
        with pytest.raises(RuntimeError, match="must be fitted"):
            imputer.transform(lab_data_with_missing)

    def test_insufficient_data_raises(self):
        imputer = NeuralImputer(epochs=2)
        small_df = pd.DataFrame({"A": [1.0, 2.0], "B": [3.0, 4.0]})
        with pytest.raises(ValueError, match="at least 10 complete-case rows"):
            imputer.fit(small_df)


class TestGetImputer:

    def test_simple_factory(self):
        imputer = get_imputer("simple", strategy="mean")
        assert isinstance(imputer, ClinicalSimpleImputer)

    def test_mice_factory(self):
        imputer = get_imputer("mice", max_iter=5)
        assert isinstance(imputer, ClinicalMICEImputer)

    def test_neural_factory(self):
        imputer = get_imputer("neural", epochs=2)
        assert isinstance(imputer, NeuralImputer)

    def test_invalid_method_raises(self):
        with pytest.raises(ValueError, match="Unknown method"):
            get_imputer("invalid_method")
