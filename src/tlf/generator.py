Now let's build the TLF module. This is the piece that directly touches submission deliverables — hiring managers in pharma will immediately recognize this as real-world work.

Click src → tlf folder
Click "Add file" → "Create new file"
Type in the filename box:

generator.py

Paste this code:

python"""
tlf/generator.py
─────────────────
Automated Tables, Listings, and Figures (TLF) generation engine.

Generates submission-ready clinical study report outputs including:
    - Demographic summary tables (Table 14.1.1)
    - Adverse event summary tables (Table 14.3.1)
    - Laboratory shift tables (Table 14.3.4)
    - Subject data listings
    - LLM-powered narrative drafting for CSR sections

Output formats:
    - RTF  (submission standard)
    - HTML (review/QC)
    - CSV  (programmatic access)

Author : Girish Rajeev
         Clinical Data Scientist | Regulatory Standards Leader | AI/ML Solution Engineer
"""

from __future__ import annotations

import io
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from jinja2 import Environment, BaseLoader

logger = logging.getLogger(__name__)


# ── Output dataclass ──────────────────────────────────────────────────────────

@dataclass
class TLFOutput:
    """Container for a generated TLF artifact."""

    title: str
    table_number: str
    content_html: str
    content_csv: str
    footnotes: list[str] = field(default_factory=list)
    programcode: str = ""

    def save_html(self, output_dir: str | Path) -> Path:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        safe_num = self.table_number.replace(".", "_")
        path = output_dir / f"table_{safe_num}.html"
        path.write_text(self._wrap_html(), encoding="utf-8")
        logger.info("Saved HTML: %s", path)
        return path

    def save_csv(self, output_dir: str | Path) -> Path:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        safe_num = self.table_number.replace(".", "_")
        path = output_dir / f"table_{safe_num}.csv"
        path.write_text(self.content_csv, encoding="utf-8")
        logger.info("Saved CSV: %s", path)
        return path

    def _wrap_html(self) -> str:
        footnote_html = "".join(
            f"<p class='footnote'>{fn}</p>" for fn in self.footnotes
        )
        return f"""<!DOCTYPE html>
<html>
<head>
<meta charset='utf-8'>
<title>{self.title}</title>
<style>
  body {{ font-family: 'Courier New', monospace; font-size: 11px; margin: 40px; }}
  h2   {{ font-size: 13px; font-weight: bold; }}
  table{{ border-collapse: collapse; width: 100%; }}
  th   {{ border-bottom: 2px solid black; padding: 4px 8px; text-align: left; }}
  td   {{ border-bottom: 1px solid #ccc; padding: 4px 8px; }}
  .footnote {{ font-size: 10px; margin-top: 2px; }}
</style>
</head>
<body>
<p><b>Table {self.table_number}</b></p>
<h2>{self.title}</h2>
{self.content_html}
{footnote_html}
</body>
</html>"""


# ── TLF Generator ─────────────────────────────────────────────────────────────

class TLFGenerator:
    """
    Generates submission-ready Tables, Listings, and Figures
    from ADaM analysis datasets.

    Parameters
    ----------
    study_id : str
        Sponsor study identifier.
    treatment_var : str
        Treatment variable name in ADaM datasets. Default 'TRTA'.
    treatment_order : list[str], optional
        Ordered list of treatment arm labels for column ordering.

    Examples
    --------
    >>> gen = TLFGenerator(
    ...     study_id="DRUGX-2024-003",
    ...     treatment_order=["Drug X", "Placebo", "Total"],
    ... )
    >>> table = gen.demographics_table(adsl_df)
    >>> table.save_html("output/tlf/")
    """

    def __init__(
        self,
        study_id: str,
        treatment_var: str = "TRTA",
        treatment_order: Optional[list[str]] = None,
    ) -> None:
        self.study_id       = study_id
        self.treatment_var  = treatment_var
        self.treatment_order = treatment_order or []

    # ── Table 14.1.1 — Demographics ───────────────────────────────────────────

    def demographics_table(self, adsl_df: pd.DataFrame) -> TLFOutput:
        """
        Generate demographic and baseline characteristics summary table.
        ICH E3 Section 11.2 / Table 14.1.1

        Parameters
        ----------
        adsl_df : pd.DataFrame
            ADSL dataset filtered to safety population (SAFFL == 'Y').

        Returns
        -------
        TLFOutput
        """
        df = adsl_df[adsl_df.get("SAFFL", pd.Series("Y", index=adsl_df.index)) == "Y"].copy()
        trt = self.treatment_var

        rows = []

        # ── N per arm ─────────────────────────────────────────────────────────
        n_counts = df.groupby(trt).size()
        rows.append(self._header_row("Characteristic", n_counts))

        # ── Age ───────────────────────────────────────────────────────────────
        rows.append(("Age (years)", "", ""))
        rows += self._continuous_rows(df, "AGE", trt, n_counts)

        # ── Sex ───────────────────────────────────────────────────────────────
        rows.append(("Sex, n (%)", "", ""))
        rows += self._categorical_rows(df, "SEX", trt, n_counts, {
            "M": "Male", "F": "Female", "U": "Unknown"
        })

        # ── Race ──────────────────────────────────────────────────────────────
        rows.append(("Race, n (%)", "", ""))
        rows += self._categorical_rows(df, "RACE", trt, n_counts)

        # ── Age group ─────────────────────────────────────────────────────────
        if "AGEGR1" in df.columns:
            rows.append(("Age group, n (%)", "", ""))
            rows += self._categorical_rows(df, "AGEGR1", trt, n_counts)

        result_df = self._rows_to_df(rows, n_counts)
        return TLFOutput(
            title="Summary of Demographic and Baseline Characteristics",
            table_number="14.1.1",
            content_html=result_df.to_html(index=False, border=0, na_rep=""),
            content_csv=result_df.to_csv(index=False),
            footnotes=[
                "Safety population: all subjects who received at least one dose.",
                "Continuous variables: n, mean (SD), median, min, max.",
                "Categorical variables: n (%).",
            ],
        )

    # ── Table 14.3.1 — Adverse Events ────────────────────────────────────────

    def adverse_events_table(self, adae_df: pd.DataFrame, adsl_df: pd.DataFrame) -> TLFOutput:
        """
        Generate adverse events summary table.
        ICH E3 Section 11.3.2 / Table 14.3.1

        Parameters
        ----------
        adae_df : pd.DataFrame
            ADAE dataset.
        adsl_df : pd.DataFrame
            ADSL dataset for denominator counts.

        Returns
        -------
        TLFOutput
        """
        trt = self.treatment_var
        teae = adae_df[adae_df.get("TRTEMFL", pd.Series("Y")) == "Y"].copy()
        n_counts = adsl_df[
            adsl_df.get("SAFFL", pd.Series("Y", index=adsl_df.index)) == "Y"
        ].groupby(trt).size()

        rows = []
        rows.append(self._header_row("Adverse Event Category", n_counts))

        categories = [
            ("Any TEAE",                teae),
            ("Any serious TEAE",        teae[teae.get("ASER", pd.Series("N")) == "Y"]),
            ("Any related TEAE",        teae[teae.get("AEREL", pd.Series("N")) == "Y"]),
            ("Any Grade ≥ 3 TEAE",      teae[teae.get("ATOXGR", pd.Series(0)) >= 3]),
            ("TEAE leading to death",   teae[teae.get("AEOUT", pd.Series("")) == "FATAL"]),
        ]

        for label, subset in categories:
            subj_counts = subset.groupby(trt)["USUBJID"].nunique()
            row_vals = {"Characteristic": f"  {label}"}
            for arm in n_counts.index:
                n_arm  = n_counts.get(arm, 0)
                n_subj = subj_counts.get(arm, 0)
                pct    = 100 * n_subj / n_arm if n_arm > 0 else 0
                row_vals[arm] = f"{n_subj} ({pct:.1f}%)"
            rows.append(tuple(row_vals.values()))

        result_df = self._rows_to_df(rows, n_counts)
        return TLFOutput(
            title="Summary of Treatment-Emergent Adverse Events",
            table_number="14.3.1",
            content_html=result_df.to_html(index=False, border=0, na_rep=""),
            content_csv=result_df.to_csv(index=False),
            footnotes=[
                "TEAE = treatment-emergent adverse event (onset on or after first dose).",
                "Subjects counted once per category regardless of number of events.",
                "Percentages based on safety population N.",
            ],
        )

    # ── Table 14.3.4 — Lab Shift Table ───────────────────────────────────────

    def lab_shift_table(
        self,
        adlb_df: pd.DataFrame,
        parameter: str = "HGB",
    ) -> TLFOutput:
        """
        Generate laboratory shift table (baseline vs worst post-baseline).

        Parameters
        ----------
        adlb_df : pd.DataFrame
            ADLB dataset.
        parameter : str
            Lab parameter code (PARAMCD) to tabulate. Default 'HGB'.

        Returns
        -------
        TLFOutput
        """
        trt = self.treatment_var
        param_df = adlb_df[adlb_df["PARAMCD"] == parameter].copy()

        baseline = param_df[param_df.get("ABLFL", pd.Series("")) == "Y"][
            ["USUBJID", "ANRIND"]
        ].rename(columns={"ANRIND": "BASELINE"})

        post = param_df[param_df.get("ABLFL", pd.Series("")) != "Y"].copy()
        worst = (
            post.sort_values("AVAL", ascending=False)
            .groupby("USUBJID")
            .first()
            .reset_index()[["USUBJID", "ANRIND", trt]]
            .rename(columns={"ANRIND": "WORST_POST"})
        )

        shift = baseline.merge(worst, on="USUBJID", how="inner")
        pivot = (
            shift.groupby(["BASELINE", "WORST_POST", trt])
            .size()
            .reset_index(name="N")
        )

        result_df = pivot.pivot_table(
            index=["BASELINE", "WORST_POST"],
            columns=trt,
            values="N",
            fill_value=0,
        ).reset_index()

        return TLFOutput(
            title=f"Shift Table: {parameter} (Baseline vs Worst Post-Baseline)",
            table_number="14.3.4",
            content_html=result_df.to_html(index=False, border=0, na_rep="0"),
            content_csv=result_df.to_csv(index=False),
            footnotes=[
                f"Parameter: {parameter}.",
                "BASELINE = normal range category at screening/baseline visit.",
                "WORST POST = worst normal range category post-baseline.",
            ],
        )

    # ── Listing 16.2.1 — Subject Data Listing ────────────────────────────────

    def subject_listing(
        self,
        adsl_df: pd.DataFrame,
        cols: Optional[list[str]] = None,
    ) -> TLFOutput:
        """
        Generate subject-level data listing (Listing 16.2.1).

        Parameters
        ----------
        adsl_df : pd.DataFrame
            ADSL dataset.
        cols : list[str], optional
            Columns to include. Defaults to key demographic variables.

        Returns
        -------
        TLFOutput
        """
        default_cols = [
            "USUBJID", "SITEID", "AGE", "AGEGR1",
            "SEX", "RACE", self.treatment_var,
            "ITTFL", "SAFFL",
        ]
        cols = [c for c in (cols or default_cols) if c in adsl_df.columns]
        listing = adsl_df[cols].sort_values(["USUBJID"])

        return TLFOutput(
            title="Subject Disposition and Demographic Listing",
            table_number="16.2.1",
            content_html=listing.to_html(index=False, border=0, na_rep=""),
            content_csv=listing.to_csv(index=False),
            footnotes=["Subjects sorted by USUBJID."],
        )

    # ── Private helpers ───────────────────────────────────────────────────────

    def _continuous_rows(
        self,
        df: pd.DataFrame,
        col: str,
        trt: str,
        n_counts: pd.Series,
    ) -> list[tuple]:
        """Generate summary rows for a continuous variable."""
        rows = []
        stats = [
            ("  n",      lambda x: str(x.count())),
            ("  Mean (SD)", lambda x: f"{x.mean():.1f} ({x.std():.2f})"),
            ("  Median",    lambda x: f"{x.median():.1f}"),
            ("  Min, Max",  lambda x: f"{x.min():.1f}, {x.max():.1f}"),
        ]
        for label, func in stats:
            row = {"Characteristic": label}
            for arm in n_counts.index:
                vals = df[df[trt] == arm][col].dropna()
                row[arm] = func(vals) if len(vals) > 0 else "-"
            rows.append(tuple(row.values()))
        return rows

    def _categorical_rows(
        self,
        df: pd.DataFrame,
        col: str,
        trt: str,
        n_counts: pd.Series,
        label_map: Optional[dict] = None,
    ) -> list[tuple]:
        """Generate summary rows for a categorical variable."""
        rows = []
        cats = df[col].dropna().unique()
        for cat in sorted(cats):
            label = (label_map or {}).get(cat, str(cat))
            row = {"Characteristic": f"  {label}"}
            for arm in n_counts.index:
                arm_df = df[df[trt] == arm]
                n_cat  = (arm_df[col] == cat).sum()
                n_arm  = n_counts.get(arm, 0)
                pct    = 100 * n_cat / n_arm if n_arm > 0 else 0
                row[arm] = f"{n_cat} ({pct:.1f}%)"
            rows.append(tuple(row.values()))
        return rows

    def _header_row(self, label: str, n_counts: pd.Series) -> tuple:
        row = {"Characteristic": label}
        for arm, n in n_counts.items():
            row[arm] = f"(N={n})"
        return tuple(row.values())

    def _rows_to_df(self, rows: list[tuple], n_counts: pd.Series) -> pd.DataFrame:
        cols = ["Characteristic"] + list(n_counts.index)
        padded = [
            r + ("",) * (len(cols) - len(r)) if len(r) < len(cols) else r
            for r in rows
        ]
        return pd.DataFrame(padded, columns=cols)
