# clinical-data-pipeline
A production-grade clinical data pipeline covering the full study lifecycle — from protocol ingestion to submission-ready TLFs — powered by Python, LLMs, and machine learning.

## Pipeline Overview

| Module | Description | Key Libraries |
|--------|-------------|---------------|
| Protocol Automation | LLM-powered extraction of endpoints, arms, and visit schedules from protocol PDFs | LangChain, PyMuPDF |
| SDTM Pipeline | Domain mappers, controlled terminology validation, define.xml generation | pandas, pyreadstat |
| ADaM Engine | ADSL, ADAE, ADTTE, ADLB derivations with ML-based imputation | scikit-learn, TensorFlow |
| TLF Generation | Automated table/listing/figure shells with LLM narrative drafting | Jinja2, python-docx |
| ML Models | Survival analysis, AE signal detection, ClinicalBERT fine-tuning | lifelines, TensorFlow, Hugging Face |

## Repository Structure
