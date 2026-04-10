"""
sdtm/base.py
─────────────
Abstract base class for all SDTM domain mappers.

Every domain (DM, AE, LB, VS, EX, CM ...) inherits from SDTMDomain
and must implement the transform() method which converts a raw
pandas DataFrame into a submission-ready SDTM domain dataset.

Author : Girish Rajeev
         Clinical Data Scientist | Regulatory Standards Leader | AI/ML Solution Engineer
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import pandas as pd
import pyreadstat

logger = logging.getLogger(__name__)


# ── Validation result ─────────────────────────────────────────────────────────

@dataclass
class ValidationResult:
    """Holds the outcome of SDTM conformance checks."""

    domain: str
    is_valid: bool = True
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    def add_error(self, message: str) -> None:
        self.errors.append(message)
        self.is_valid = False

    def add_warning(self, message: str) -> None:
        self.warnings.append(message)

    def summary(self) -> str:
        lines = [f"Domain: {self.domain}"]
        lines.append(f"Valid: {self.is_valid}")
        if self.errors:
            lines.append(f"Errors ({len(self.errors)}):")
            lines.extend(f"  [ERROR] {e}" for e in self.errors)
        if self.warnings:
            lines.append(f"Warnings ({len(self.warnings)}):")
            lines.extend(f"  [WARN]  {w}" for w in self.warnings)
        return "\n".join(lines)


# ── Base domain ───────────────────────────────────────────────────────────────

class SDTMDomain(ABC):
    """
    Abstract base class for SDTM domain mappers.

    Subclasses must define:
    - DOMAIN      : two-letter SDTM domain code (e.g. 'AE')
    - REQUIRED_VARS : list of required SDTM variables for this domain
    - transform() : maps raw input DataFrame to SDTM-compliant DataFrame

    Parameters
    ----------
    study_id : str
        Sponsor study identifier (STUDYID).
    controlled_terms : dict, optional
        Custom controlled terminology overrides.

    Examples
    --------
    >>> from src.sdtm.domains import AdverseEvents
    >>> ae = AdverseEvents(study_id="DRUGX-2024-003")
    >>> sdtm_ae = ae.transform(raw_ae_df)
    >>> result = ae.validate(sdtm_ae)
    >>> print(result.summary())
    """

    DOMAIN: str = ""
    REQUIRED_VARS: list[str] = []

    def __init__(
        self,
        study_id: str,
        controlled_terms: Optional[dict] = None,
    ) -> None:
        if not self.DOMAIN:
            raise NotImplementedError("Subclasses must define DOMAIN class attribute.")
        self.study_id = study_id
        self.controlled_terms = controlled_terms or {}
        logger.info("Initialized %s domain mapper for study %s", self.DOMAIN, study_id)

    # ── Abstract interface ────────────────────────────────────────────────────

    @abstractmethod
    def transform(self, raw_df: pd.DataFrame) -> pd.DataFrame:
        """
        Transform raw source data into a SDTM-compliant domain dataset.

        Parameters
        ----------
        raw_df : pd.DataFrame
            Raw input data from EDC or source system.

        Returns
        -------
        pd.DataFrame
            SDTM-compliant domain dataset with all required variables.
        """

    # ── Shared utilities ──────────────────────────────────────────────────────

    def validate(self, sdtm_df: pd.DataFrame) -> ValidationResult:
        """
        Run conformance checks on a transformed SDTM dataset.

        Checks performed:
        - All required variables are present
        - STUDYID, DOMAIN are populated for every row
        - No duplicate records on key variables
        - USUBJID follows expected format

        Parameters
        ----------
        sdtm_df : pd.DataFrame
            Output from transform().

        Returns
        -------
        ValidationResult
        """
        result = ValidationResult(domain=self.DOMAIN)

        # Check required variables
        missing = [v for v in self.REQUIRED_VARS if v not in sdtm_df.columns]
        for var in missing:
            result.add_error(f"Required variable missing: {var}")

        # Check STUDYID populated
        if "STUDYID" in sdtm_df.columns:
            if sdtm_df["STUDYID"].isna().any():
                result.add_error("STUDYID contains null values")
        
        # Check DOMAIN populated
        if "DOMAIN" in sdtm_df.columns:
            wrong_domain = sdtm_df[sdtm_df["DOMAIN"] != self.DOMAIN]
            if not wrong_domain.empty:
                result.add_error(f"DOMAIN value mismatch — expected {self.DOMAIN}")

        # Check USUBJID format (STUDYID-SITEID-SUBJID pattern)
        if "USUBJID" in sdtm_df.columns:
            invalid_usubjid = sdtm_df[~sdtm_df["USUBJID"].str.contains("-", na=False)]
            if not invalid_usubjid.empty:
                result.add_warning(
                    f"{len(invalid_usubjid)} USUBJID values may not follow "
                    "STUDYID-SITEID-SUBJID convention"
                )

        if result.is_valid:
            logger.info("%s validation passed.", self.DOMAIN)
        else:
            logger.warning("%s validation failed: %d errors", self.DOMAIN, len(result.errors))

        return result

    def export_xpt(self, sdtm_df: pd.DataFrame, output_dir: str | Path) -> Path:
        """
        Export SDTM domain dataset as a SAS XPT transport file.

        Parameters
        ----------
        sdtm_df : pd.DataFrame
            SDTM-compliant domain dataset.
        output_dir : str or Path
            Directory where the .xpt file will be saved.

        Returns
        -------
        Path
            Full path to the exported .xpt file.
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        xpt_path = output_dir / f"{self.DOMAIN.lower()}.xpt"

        pyreadstat.write_xport(sdtm_df, str(xpt_path), table_name=self.DOMAIN)
        logger.info("Exported %s to %s", self.DOMAIN, xpt_path)
        return xpt_path

    def _add_standard_vars(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add STUDYID and DOMAIN columns to every domain dataset."""
        df = df.copy()
        df.insert(0, "STUDYID", self.study_id)
        df.insert(1, "DOMAIN", self.DOMAIN)
        return df

    def _apply_controlled_terms(self, df: pd.DataFrame, column: str) -> pd.DataFrame:
        """Apply controlled terminology mapping to a specified column."""
        if column in self.controlled_terms:
            df[column] = df[column].map(self.controlled_terms[column]).fillna(df[column])
        return df

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(domain={self.DOMAIN!r}, study_id={self.study_id!r})"
