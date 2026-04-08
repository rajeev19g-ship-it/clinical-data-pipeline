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
clinical-data-pipeline/
├── src/
│   ├── protocol_automation/   # LLM protocol parser
│   ├── sdtm/                  # SDTM domain mappers
│   ├── adam/                  # ADaM derivation engine
│   ├── tlf/                   # TLF generation
│   └── models/                # ML/DL models
├── tests/                     # pytest test suite
├── notebooks/                 # Exploratory notebooks
└── data/                      # Synthetic CDISC pilot data and Clinical Annonymized data

## Tech Stack

- **Languages:** Python 3.10+
- **ML/DL:** TensorFlow, scikit-learn, Hugging Face Transformers
- **Clinical:** pyreadstat, pandas, Jinja2
- **LLM:** LangChain, OpenAI API
- **Testing:** pytest, GitHub Actions CI/CD
- **Standards:** CDISC SDTM 3.3, ADaM 1.3, SEND 3.1

## Getting Started

```bash
git clone https://github.com/rajeev19g-ship-it/clinical-data-pipeline.git
cd clinical-data-pipeline
pip install -r requirements.txt
```

## Author:  Girish Rajeev

Clinical Data Scientist | Data Analyst | Regulatory Standards Leader | AI/ML Solution Engineer 
