# Dataset

This project uses the **ds004033 EEG dataset** from OpenNeuro.

https://openneuro.org/datasets/ds004033

The dataset contains EEG recordings from participants performing two experimental paradigms:

- **Auditory oddball task** under three movement conditions  
  - standing  
  - walking alone  
  - walking together  

- **Walking synchronization task**, where participants synchronize their walking with a partner.

The recordings include gait event markers such as:

- RHS — Right Heel Strike  
- RTO — Right Toe Off  

These events are used for stride-based time–frequency EEG analysis.

---

# Downloading the dataset

The dataset is **not included in this repository** due to its large size (~20GB).

Please download it from OpenNeuro.

```bash
openneuro download ds004033
```

After downloading, place the dataset inside the data/ directory so the structure looks like this:
```
data/
└── ds004033/
    ├── sub-001/
    ├── sub-002/
    ├── sub-003/
    ├── ...
    ├── sub-018/
    ├── dataset_description.json
    ├── participants.tsv