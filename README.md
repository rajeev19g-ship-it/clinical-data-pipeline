# Clinical Data Pipeline

A production-grade clinical data pipeline covering the full study lifecycle — from protocol ingestion to submission-ready TLFs — powered by Python, LLMs, and machine learning.

[![CI](https://github.com/rajeev19g-ship-it/clinical-data-pipeline/actions/workflows/ci.yml/badge.svg)](https://github.com/rajeev19g-ship-it/clinical-data-pipeline/actions)
[![Python](https://img.shields.io/badge/Python-3.10%2B-blue)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![CDISC](https://img.shields.io/badge/Standard-CDISC%20SDTM%203.3%20%7C%20ADaM%201.3-orange)](https://www.cdisc.org/)

---
A production-grade CDISC-compliant clinical data pipeline with LLM-powered protocol automation, SDTM/ADaM derivations, automated TLF generation, and ML/DL models for survival analysis and adverse event signal detection.

![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![License](https://img.shields.io/badge/License-MIT-green)
![CDISC](https://img.shields.io/badge/Standard-CDISC%20SDTM%20%7C%20ADaM-teal)
![FDA](https://img.shields.io/badge/Submission-FDA%20eCTD-red)
![TensorFlow](https://img.shields.io/badge/ML-TensorFlow%20%7C%20scikit--learn-orange)

---

### Architecture

```
Protocol PDF / Raw EDC Data
         ↓
Protocol Automation    —  LLM extraction · variable mapping · CRF scaffolding
         ↓
SDTM Pipeline          —  DM · AE · LB · VS · EX · define.xml · terminology
         ↓
ADaM Engine            —  ADSL · ADAE · ADTTE · ADLB · ML imputation
         ↓
TLF Generation         —  Tables · listings · figures · CSR narratives
         ↓
ML / DL Models         —  Kaplan-Meier · Cox PH · AE signal detection
         ↓
FDA eCTD Submission Package
```

---

## Platform Overview

| Module | Files | Description |
|--------|-------|-------------|
| `src/protocol_automation/` | `parser.py`, `variable_mapper.py` | LLM-powered protocol PDF extraction and SDTM/ADaM variable mapping |
| `src/sdtm/` | `base.py`, `domains.py`, `define_xml.py` | SDTM domain mappers (DM/AE/LB/VS/EX), terminology validation, Define-XML 2.0 |
| `src/adam/` | `derivations.py`, `imputation.py` | ADSL/ADAE/ADTTE/ADLB derivations, ML-based missing value imputation |
| `src/tlf/` | `generator.py`, `narrative.py` | Automated TLF shells and LLM-powered CSR Section 11 narrative drafting |
| `src/models/` | `survival.py`, `ae_signal.py` | Kaplan-Meier, Cox PH survival models, denoising autoencoder AE signal detection |

---


---

## Modules

| Module | Files | Description |
|--------|-------|-------------|
| `src/protocol_automation/` | `parser.py`, `variable_mapper.py` | LLM-powered protocol PDF extraction, SDTM/ADaM variable mapping |
| `src/sdtm/` | `base.py`, `domains.py`, `define_xml.py` | SDTM domain mappers (DM/AE/LB/VS/EX), define.xml generator |
| `src/adam/` | `derivations.py`, `imputation.py` | ADSL/ADAE/ADTTE/ADLB derivations, ML-based missing value imputation |
| `src/tlf/` | `generator.py`, `narrative.py` | Automated TLF shells, LLM-powered CSR narrative drafter |
| `src/models/` | `survival.py`, `ae_signal.py` | KM/Cox PH survival models, autoencoder AE signal detection |

---

---

## Tech Stack

| Category | Libraries |
|----------|-----------|
| LLM / NLP | openai, langchain, langchain-openai, transformers |
| ML / DL | TensorFlow, Keras, scikit-learn, lifelines |
| Clinical / CDISC | pyreadstat, xport, Jinja2 |
| Document generation | python-docx, reportlab |
| MLOps | mlflow |
| Serving | FastAPI, uvicorn, pydantic |
| Testing | pytest, pytest-cov |

---

## Key Features

- **LLM Protocol Parser** — Extracts endpoints, arms, inclusion/exclusion criteria, and visit schedules from protocol PDFs using GPT-4
- **SDTM Domain Mappers** — Production-ready DM, AE, LB, VS, EX mappers with CDISC controlled terminology validation and XPT export
- **define.xml Generator** — Automated CDISC Define-XML 2.0 for FDA/EMA eCTD submissions
- **ADaM Derivations** — ADSL population flags, ADAE treatment-emergent flags, ADTTE survival endpoints (OS/PFS), ADLB change from baseline
- **ML Imputation** — Three-tier imputation: SimpleImputer → MICE → Neural autoencoder (TensorFlow)
- **TLF Engine** — Demographic tables (14.1.1), AE summaries (14.3.1), lab shift tables (14.3.4), subject listings (16.2.1)
- **CSR Narrative Drafter** — ICH E3-compliant Section 11.2/11.4/11.5 draft narratives via LLM
- **Survival Models** — Kaplan-Meier with CI, log-rank test, Cox PH regression (lifelines)
- **AE Signal Detection** — Denoising autoencoder for unexpected AE pattern detection

---

## Tech Stack

| Category | Libraries |
|----------|-----------|
| Clinical/CDISC | pyreadstat, xport, pandas |
| LLM/NLP | openai, langchain, transformers |
| ML/DL | scikit-learn, TensorFlow, lifelines |
| TLF Output | Jinja2, python-docx, reportlab |
| Testing | pytest, pytest-cov |
| CI/CD | GitHub Actions |

---

### Repository Structure

Same issue — the tree characters (├──, └──, │) aren't rendering on GitHub. Let me give you a clean replacement for all three READMEs.
For clinical-data-pipeline README, find the Repository Structure section and replace with:
markdown## Repository Structure

```
clinical-data-pipeline/
├── .github/workflows/ci.yml        # GitHub Actions CI pipeline
├── src/
│   ├── protocol_automation/
│   │   ├── parser.py               # LLM protocol parser
│   │   └── variable_mapper.py      # SDTM/ADaM variable mapper
│   ├── sdtm/
│   │   ├── base.py                 # Abstract SDTMDomain base class
│   │   ├── domains.py              # DM, AE, LB, VS, EX mappers
│   │   └── define_xml.py           # Define-XML 2.0 generator
│   ├── adam/
│   │   ├── derivations.py          # ADSL, ADAE, ADTTE, ADLB
│   │   └── imputation.py           # Simple, MICE, Neural imputers
│   ├── tlf/
│   │   ├── generator.py            # TLF generation engine
│   │   └── narrative.py            # LLM CSR narrative drafter
│   └── models/
│       ├── survival.py             # KM, log-rank, Cox PH
│       └── ae_signal.py            # AE anomaly detection
├── tests/
│   ├── test_parser.py
│   ├── test_sdtm.py
│   ├── test_adam.py
│   ├── test_tlf.py
│   └── test_models.py
├── notebooks/                      # Exploratory analysis + MLflow
├── data/                           # Synthetic CDISC pilot data
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
└── README.md
```

# Repository Structure



## Getting Started

```bash
git clone https://github.com/rajeev19g-ship-it/clinical-data-pipeline.git
cd clinical-data-pipeline
pip install -r requirements.txt
```

Set your OpenAI API key:
```bash
export OPENAI_API_KEY="your-key-here"
```

Run the test suite:
```bash
pytest tests/ -v --cov=src
```

---

## Regulatory Standards

- CDISC SDTM Implementation Guide v3.3
- CDISC ADaM Implementation Guide v1.3
- CDISC Define-XML 2.0
- ICH E3 — Structure and Content of Clinical Study Reports
- ICH E9 — Statistical Principles for Clinical Trials
- FDA Study Data Technical Conformance Guide

---

## Author

**Girish Rajeev**
Clinical Data Scientist | Data Analyst | Regulatory Standards Leader | AI/ML Solution Engineer

[![LinkedIn](https://img.shields.io/badge/LinkedIn-Connect-blue)](https://www.linkedin.com/in/girish-rajeev-756808138/)
