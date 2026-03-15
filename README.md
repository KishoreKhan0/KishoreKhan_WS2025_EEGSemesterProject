# EEG Walking Synchronization Project

Reproduction and extension of the EEG walking synchronization study using the **ds004033 dataset**.  
This project implements both the **original authors' pipeline** and a **custom improved pipeline**, and compares their results across ERP, time–frequency, and decoding analyses.

---

## Table of Contents

1. [Project Overview](#project-overview)
2. [Dataset](#dataset)
3. [Quick Start](#quick-start)
4. [Project Structure](#project-structure)
5. [Analysis Pipeline](#analysis-pipeline)
6. [Notebooks](#notebooks)
7. [Results Preview](#results-preview)
8. [Reproducibility](#reproducibility)
9. [Author](#author)

# Project Overview

This repository reproduces key analyses from the walking synchronization EEG study and evaluates how different preprocessing pipelines influence the results.

Two pipelines are implemented:

**Authors pipeline**
- Replicates the preprocessing and analysis described in the original publication.

**Seals pipeline (ours)**
- Implements an alternative preprocessing strategy with improved noise handling and artifact correction.

Both pipelines are applied to the same dataset and compared across several analysis stages.

---

# Dataset

This project uses the **OpenNeuro ds004033 dataset**.

https://openneuro.org/datasets/ds004033

The dataset contains EEG recordings collected during:

- **Auditory oddball task**
  - standing
  - walking alone
  - walking together

- **Walking synchronization task**
  - gait events detected using heel strike and toe off markers

See `data/README.md` for download instructions.

---

# Project Structure

```
EEG_PROJECT_FINAL/
│
├── authors_pipeline/ # Replication of authors pipeline
│ |── notebooks/
│ 
├── ours_pipeline/ # Our preprocessing + analysis pipeline
│ ├── notebooks/
│
├── shared/ # Shared analysis code
│ ├── src/
│ ├── scripts/
│ └── notebooks/
│
├── configs/ # YAML configuration files
│
├── data/ # Dataset location (data not included)
│
├── results/ # Project results for every step
│
├── environment.yml # Conda environment
├── requirements.txt # Python dependencies
└── README.md
```



---

# Analysis Pipeline

Dataset (ds004033)
         │
        ▼
Event Mapping
         │
        ▼
EEG Preprocessing
         │
        ▼
ERP Analysis (Oddball Task)
         │
        ▼
Time–Frequency Analysis (Walking Synchronization)
         │
        ▼
Decoding Analysis
         │
        ▼
Pipeline Comparison (Authors vs Ours)


1. **Event mapping**  
   Mapping raw dataset events to experimental conditions.

2. **EEG preprocessing**
   - filtering
   - bad channel handling
   - artifact removal
   - referencing

3. **Oddball ERP replication**
   - deviant vs standard responses
   - group-level ERP comparison

4. **Walking synchronization analysis**
   - stride-based epoching
   - time–frequency decomposition
   - ERSP computation

5. **Pipeline comparison**
   - authors vs ours

6. **Decoding analysis**
   - condition classification using EEG features

---

# Running the Analysis

Activate the environment:

```bash
conda env create -f environment.yml
conda activate eeg_project
```

# Run the main analysis scripts
```bash
python shared/scripts/run_preprocessing.py --config configs/preprocessing_ours.yaml
python shared/scripts/run_oddball_replication.py --config configs/oddball_replication_ours.yaml
python shared/scripts/run_sync_tfr_analysis.py --config configs/sync_tfr_ours.yaml
python shared/scripts/run_decoding_analysis.py --config configs/decoding_ours.yaml
```
# Notebooks

The notebooks demonstrate the full workflow:

| Notebook                       | Description                                        |
| ------------------------------ | -------------------------------------------------- |
| `02_preprocessing.ipynb`       | EEG preprocessing pipeline                         |
| `03_oddball_replication.ipynb` | ERP analysis for oddball task                      |
| `04_pipeline_comparison.ipynb` | Comparison of preprocessing pipelines              |
| `05_sync_tfr_analysis.ipynb`   | Time–frequency analysis of walking synchronization |
| `06_sync_comparison.ipynb`     | Comparison of synchronization results              |
| `07_decoding_analysis.ipynb`   | EEG decoding analysis                              |
| `08_decoding_comparison.ipynb` | Comparison of decoding performance                 |


# Key Results

Both pipelines successfully reproduced the main experimental analyses.

## Comparison highlights:

- ERP responses were replicated across all movement conditions
- Time–frequency patterns were consistent between pipelines
- Decoding accuracy was similar across pipelines

# Reproducibility

To reproduce the project:
- Download dataset *ds004033*
- Place it in **data/ds004033/**
- Install the environment
- Run the analysis scripts or notebooks

All results can then be reproduced locally.

# Author
- Kishore Khan
- Eashan Sai
- Georgina Shirazi
