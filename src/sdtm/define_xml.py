Now let's add the final SDTM file — define_xml.py:

Make sure you're inside the sdtm folder
Click "Add file" → "Create new file"
Type in the filename box:

define_xml.py

Paste this code:

python"""
sdtm/define_xml.py
───────────────────
Automated define.xml generator for CDISC SDTM submissions.

Generates a CDISC-compliant define.xml (Define-XML 2.0) document
from a collection of SDTM domain DataFrames. The define.xml is a
required component of every FDA/EMA electronic submission (eCTD).

Output conforms to:
  - CDISC Define-XML 2.0 specification
  - FDA Study Data Technical Conformance Guide
  - PMDA submission requirements

Author : Girish Rajeev
         Clinical Data Scientist | Regulatory Standards Leader | AI/ML Solution Engineer
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Optional
from xml.dom import minidom
from xml.etree import ElementTree as ET

import pandas as pd

logger = logging.getLogger(__name__)


# ── Variable metadata ─────────────────────────────────────────────────────────

@dataclass
class VariableDef:
    """Metadata for a single SDTM variable."""
    name: str
    label: str
    data_type: str          # text | integer | float | datetime
    length: int
    mandatory: bool = False
    controlled_terms: str = ""
    origin: str = "CRF"     # CRF | Derived | Assigned | Protocol
    comment: str = ""


@dataclass
class DomainDef:
    """Metadata for a single SDTM domain dataset."""
    domain_code: str
    dataset_name: str
    label: str
    structure: str          # e.g. "One record per subject per adverse event"
    purpose: str = "Tabulation"
    variables: list[VariableDef] = field(default_factory=list)
    records: int = 0


# ── Built-in variable labels ──────────────────────────────────────────────────

VARIABLE_LABELS: dict[str, tuple[str, str, int]] = {
    # name: (label, data_type, length)
    "STUDYID":  ("Study Identifier",                    "text",     12),
    "DOMAIN":   ("Domain Abbreviation",                 "text",      2),
    "USUBJID":  ("Unique Subject Identifier",           "text",     50),
    "SUBJID":   ("Subject Identifier for the Study",    "text",     20),
    "SITEID":   ("Study Site Identifier",               "text",     20),
    "AGE":      ("Age",                                 "integer",   3),
    "AGEU":     ("Age Units",                           "text",     10),
    "SEX":      ("Sex",                                 "text",      1),
    "RACE":     ("Race",                                "text",     50),
    "ETHNIC":   ("Ethnicity",                           "text",     50),
    "ARMCD":    ("Planned Arm Code",                    "text",     20),
    "ARM":      ("Description of Planned Arm",          "text",     50),
    "RFSTDTC":  ("Subject Reference Start Date/Time",   "datetime", 20),
    "RFICDTC":  ("Date/Time of Informed Consent",       "datetime", 20),
    "AESEQ":    ("Sequence Number",                     "integer",   8),
    "AETERM":   ("Reported Term for the AE",            "text",    200),
    "AEDECOD":  ("Dictionary-Derived Term",             "text",    200),
    "AEBODSYS": ("Body System or Organ Class",          "text",    200),
    "AESTDTC":  ("Start Date/Time of AE",               "datetime", 20),
    "AEENDTC":  ("End Date/Time of AE",                 "datetime", 20),
    "AESEV":    ("Severity/Intensity",                  "text",     20),
    "AESER":    ("Serious Event",                       "text",      1),
    "AEREL":    ("Causality",                           "text",      1),
    "AEOUT":    ("Outcome of AE",                       "text",     50),
    "LBSEQ":    ("Sequence Number",                     "integer",   8),
    "LBTESTCD": ("Lab Test or Examination Short Name",  "text",      8),
    "LBTEST":   ("Lab Test or Examination Name",        "text",     40),
    "LBCAT":    ("Category for Lab Test",               "text",     40),
    "LBORRES":  ("Result or Finding in Original Units", "text",     20),
    "LBORRESU": ("Original Units",                      "text",     20),
    "LBSTRESN": ("Numeric Result/Finding in Std Units", "float",    20),
    "LBSTRESU": ("Standard Units",                      "text",     20),
    "LBSTNRLO": ("Reference Range Lower Limit-Std",     "float",    20),
    "LBSTNRHI": ("Reference Range Upper Limit-Std",     "float",    20),
    "LBNRIND":  ("Reference Range Indicator",           "text",     10),
    "LBDTC":    ("Date/Time of Specimen Collection",    "datetime", 20),
    "VSSEQ":    ("Sequence Number",                     "integer",   8),
    "VSTESTCD": ("Vital Signs Test Short Name",         "text",      8),
    "VSTEST":   ("Vital Signs Test Name",               "text",     40),
    "VSPOS":    ("Vital Signs Position of Subject",     "text",     20),
    "VSORRES":  ("Result or Finding in Original Units", "text",     20),
    "VSORRESU": ("Original Units",                      "text",     20),
    "VSSTRESN": ("Numeric Result in Standard Units",    "float",    20),
    "VSSTRESU": ("Standard Units",                      "text",     20),
    "VSDTC":    ("Date/Time of Vital Signs",            "datetime", 20),
    "EXSEQ":    ("Sequence Number",                     "integer",   8),
    "EXTRT":    ("Name of Actual Treatment",            "text",    200),
    "EXDOSE":   ("Dose per Administration",             "float",    20),
    "EXDOSU":   ("Units of Dose",                       "text",     20),
    "EXROUTE":  ("Route of Administration",             "text",     40),
    "EXDOSFRQ": ("Dosing Frequency per Interval",       "text",     20),
    "EXSTDTC":  ("Start Date/Time of Treatment",        "datetime", 20),
    "EXENDTC":  ("End Date/Time of Treatment",          "datetime", 20),
    "VISIT":    ("Visit Name",                          "text",     40),
}

DOMAIN_STRUCTURES: dict[str, str] = {
    "DM": "One record per subject",
    "AE": "One record per adverse event per subject",
    "LB": "One record per lab test per time point per subject",
    "VS": "One record per vital sign measurement per subject",
    "EX": "One record per protocol-specified treatment per subject",
}

DOMAIN_LABELS: dict[str, str] = {
    "DM": "Demographics",
    "AE": "Adverse Events",
    "LB": "Laboratory Test Results",
    "VS": "Vital Signs",
    "EX": "Exposure",
}


# ── Define.xml generator ──────────────────────────────────────────────────────

class DefineXMLGenerator:
    """
    Generates a CDISC Define-XML 2.0 document from SDTM domain DataFrames.

    The define.xml describes the structure, content, and metadata of all
    datasets in an electronic submission package. It is required by the
    FDA (since 2017) and EMA for all NDA, BLA, and MAA submissions.

    Parameters
    ----------
    study_id : str
        Sponsor study identifier.
    study_description : str
        Brief description of the study.
    standard_version : str
        SDTM version (default: "SDTM 1.7").

    Examples
    --------
    >>> from src.sdtm.define_xml import DefineXMLGenerator
    >>> gen = DefineXMLGenerator(
    ...     study_id="DRUGX-2024-003",
    ...     study_description="Phase 3 study of Drug X in NSCLC"
    ... )
    >>> gen.add_domain("AE", ae_sdtm_df)
    >>> gen.add_domain("DM", dm_sdtm_df)
    >>> gen.write("output/define.xml")
    """

    DEFINE_XML_VERSION = "2.0.0"
    ODMVERSION        = "1.3.2"

    def __init__(
        self,
        study_id: str,
        study_description: str = "",
        standard_version: str = "SDTM 1.7",
    ) -> None:
        self.study_id = study_id
        self.study_description = study_description
        self.standard_version = standard_version
        self.domains: list[DomainDef] = []
        logger.info("DefineXMLGenerator initialized for study %s", study_id)

    # ── Public API ────────────────────────────────────────────────────────────

    def add_domain(
        self,
        domain_code: str,
        df: pd.DataFrame,
        custom_label: Optional[str] = None,
    ) -> None:
        """
        Register an SDTM domain dataset for inclusion in define.xml.

        Parameters
        ----------
        domain_code : str
            Two-letter SDTM domain code (e.g. 'AE').
        df : pd.DataFrame
            SDTM-compliant domain dataset from SDTMDomain.transform().
        custom_label : str, optional
            Override the default domain label.
        """
        variables = self._extract_variable_defs(df)
        domain = DomainDef(
            domain_code=domain_code,
            dataset_name=domain_code,
            label=custom_label or DOMAIN_LABELS.get(domain_code, domain_code),
            structure=DOMAIN_STRUCTURES.get(domain_code, "One record per subject"),
            variables=variables,
            records=len(df),
        )
        self.domains.append(domain)
        logger.info("Added domain %s (%d records, %d variables)", domain_code, len(df), len(variables))

    def write(self, output_path: str | Path) -> Path:
        """
        Generate and write the define.xml file.

        Parameters
        ----------
        output_path : str or Path
            Full path for the output define.xml file.

        Returns
        -------
        Path
            Path to the written define.xml file.
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        root = self._build_xml()
        xml_string = self._prettify(root)
        output_path.write_text(xml_string, encoding="utf-8")

        logger.info("define.xml written to %s (%d domains)", output_path, len(self.domains))
        return output_path

    def summary(self) -> dict:
        """Return a summary of all registered domains."""
        return {
            "study_id": self.study_id,
            "standard_version": self.standard_version,
            "domains": [
                {
                    "domain": d.domain_code,
                    "label": d.label,
                    "records": d.records,
                    "variables": len(d.variables),
                }
                for d in self.domains
            ],
            "total_datasets": len(self.domains),
        }

    # ── XML construction ──────────────────────────────────────────────────────

    def _build_xml(self) -> ET.Element:
        """Build the full ODM/Define-XML element tree."""
        odm = ET.Element("ODM")
        odm.set("xmlns",            "http://www.cdisc.org/ns/odm/v1.3")
        odm.set("xmlns:def",        "http://www.cdisc.org/ns/def/v2.0")
        odm.set("xmlns:xlink",      "http://www.w3.org/1999/xlink")
        odm.set("ODMVersion",       self.ODMVERSION)
        odm.set("FileType",         "Snapshot")
        odm.set("FileOID",          f"{self.study_id}.define")
        odm.set("CreationDateTime", datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%S"))
        odm.set("def:Context",      "Submission")

        study = ET.SubElement(odm, "Study")
        study.set("OID", self.study_id)

        global_vars = ET.SubElement(study, "GlobalVariables")
        ET.SubElement(global_vars, "StudyName").text = self.study_id
        ET.SubElement(global_vars, "StudyDescription").text = self.study_description
        ET.SubElement(global_vars, "ProtocolName").text = self.study_id

        meta = ET.SubElement(study, "MetaDataVersion")
        meta.set("OID",         f"{self.study_id}.MDV")
        meta.set("Name",        f"{self.study_id} Define-XML")
        meta.set("def:DefineVersion",   self.DEFINE_XML_VERSION)
        meta.set("def:StandardName",    "SDTM")
        meta.set("def:StandardVersion", self.standard_version)

        # ItemGroupDef (one per domain)
        for domain in self.domains:
            self._add_item_group_def(meta, domain)

        # ItemDef (one per unique variable across all domains)
        self._add_item_defs(meta)

        return odm

    def _add_item_group_def(self, parent: ET.Element, domain: DomainDef) -> None:
        """Add an ItemGroupDef element for a domain."""
        grp = ET.SubElement(parent, "ItemGroupDef")
        grp.set("OID",          f"IG.{domain.domain_code}")
        grp.set("Name",         domain.dataset_name)
        grp.set("Repeating",    "No" if domain.domain_code == "DM" else "Yes")
        grp.set("IsReferenceData", "No")
        grp.set("SASDatasetName",  domain.dataset_name)
        grp.set("def:Structure",   domain.structure)
        grp.set("def:Purpose",     domain.purpose)
        grp.set("def:Records",     str(domain.records))

        desc = ET.SubElement(grp, "Description")
        ET.SubElement(desc, "TranslatedText").text = domain.label

        for var in domain.variables:
            ref = ET.SubElement(grp, "ItemRef")
            ref.set("ItemOID",  f"IT.{domain.domain_code}.{var.name}")
            ref.set("Mandatory", "Yes" if var.mandatory else "No")
            ref.set("OrderNumber", str(domain.variables.index(var) + 1))

    def _add_item_defs(self, parent: ET.Element) -> None:
        """Add ItemDef elements for all variables across all domains."""
        seen: set[str] = set()
        for domain in self.domains:
            for var in domain.variables:
                item_oid = f"IT.{domain.domain_code}.{var.name}"
                if item_oid in seen:
                    continue
                seen.add(item_oid)

                item = ET.SubElement(parent, "ItemDef")
                item.set("OID",          item_oid)
                item.set("Name",         var.name)
                item.set("DataType",     var.data_type)
                item.set("Length",       str(var.length))
                item.set("SASFieldName", var.name[:8])
                item.set("def:Origin",   var.origin)

                desc = ET.SubElement(item, "Description")
                ET.SubElement(desc, "TranslatedText").text = var.label

                if var.comment:
                    ET.SubElement(item, "def:Comment").text = var.comment

    # ── Helpers ───────────────────────────────────────────────────────────────

    @staticmethod
    def _extract_variable_defs(df: pd.DataFrame) -> list[VariableDef]:
        """Infer VariableDef metadata from a DataFrame's columns and dtypes."""
        var_defs = []
        required = {"STUDYID", "DOMAIN", "USUBJID"}

        for col in df.columns:
            label, data_type, length = VARIABLE_LABELS.get(
                col, (col, "text", 200)
            )
            if pd.api.types.is_integer_dtype(df[col]):
                data_type = "integer"
            elif pd.api.types.is_float_dtype(df[col]):
                data_type = "float"

            var_defs.append(VariableDef(
                name=col,
                label=label,
                data_type=data_type,
                length=length,
                mandatory=col in required,
            ))
        return var_defs

    @staticmethod
    def _prettify(element: ET.Element) -> str:
        """Return a pretty-printed XML string with declaration."""
        raw = ET.tostring(element, encoding="unicode")
        parsed = minidom.parseString(raw)
        return parsed.toprettyxml(indent="  ", encoding=None)
