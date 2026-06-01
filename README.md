# CT Pulmonary Nodule Classification and Grad-CAM Robustness Analysis

Research-oriented deep learning pipeline for pulmonary nodule classification, volumetric Grad-CAM explainability, and perturbation robustness analysis using thoracic CT imaging derived from the LIDC-IDRI dataset.

## Features

* 3D pulmonary nodule classification
* Volumetric CT preprocessing
* Grad-CAM explainability
* Gaussian perturbation robustness analysis
* Activation drift quantification
* Confidence degradation analysis
* Failure-case interpretability analysis

## Repository Structure

```text id="j3m2ql"
.
├── build_dataset.py
├── build_localization_dataset.py
├── train.py
├── analysis.py
├── graph.py
├── requirements.txt
└── README.md
```

## Installation

```bash id="x1nv8h"
pip install -r requirements.txt
```

## Training

```bash id="b2nv7k"
python train.py
```

## Grad-CAM Analysis

```bash id="m7x0d1"
python analysis.py
```

## Notes

Generated outputs, trained checkpoints, processed datasets, and Grad-CAM figures are intentionally not publicly included.
