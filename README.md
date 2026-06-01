# Volumetric Grad-CAM Robustness Analysis for Thoracic CT Nodules

Research-oriented deep learning pipeline for analysing robustness and explainability behaviour in volumetric thoracic CT imaging using 3D convolutional neural networks and Grad-CAM-based activation analysis.

This project investigates how volumetric CNN attention patterns and prediction confidence change under degraded imaging conditions, with a focus on interpretability stability rather than benchmark optimisation alone.

---

# Overview

The pipeline uses volumetric thoracic CT patches derived from the publicly available LIDC-IDRI dataset to study:

- Volumetric nodule/background discrimination
- 3D Grad-CAM explainability
- Perturbation robustness under Gaussian degradation
- Confidence degradation behaviour
- Activation drift under distribution shift
- Failure-case interpretability analysis

The project was developed as an exploratory robustness and explainability study for medical imaging systems operating under noisy or degraded imaging conditions.

---

# Features

- 3D CNN volumetric inference
- Volumetric CT preprocessing
- Patch-based 3D analysis
- Manual 3D Grad-CAM implementation
- Gaussian perturbation robustness testing
- Activation drift quantification
- Confidence degradation analysis
- Confusion matrix and metric generation
- Explainability visualisation across axial, sagittal, and coronal planes

---

# Repository Structure

```text
.
├── build_dataset.py
├── build_localization_dataset.py
├── train.py
├── analysis.py
├── graph.py
├── requirements.txt
└── README.md
```

---

# Dataset

This project uses Thoracic CT data derived from publicly available LIDC-IDRI imaging collections


The preprocessing pipeline reconstructs fixed-size volumetric patches from CT slices for proof-of-concept volumetric analysis.

> **Note**
>
> The current implementation is intended for exploratory robustness and explainability analysis and should not be interpreted as a clinically validated malignancy prediction system.

---

# Methodology

## 1. Volumetric Preprocessing

The dataset pipeline:

- loads CT slice stacks
- reconstructs volumetric patches
- resizes slices to fixed spatial dimensions
- normalises intensity values
- pads/truncates depth to fixed volumetric size

---

## 2. 3D CNN Architecture

The model consists of:

- stacked 3D convolutional layers
- volumetric max pooling
- adaptive global pooling
- fully connected classification head

The architecture is intentionally lightweight to facilitate rapid experimentation and interpretability analysis.

---

## 3. Grad-CAM Explainability

A manual 3D Grad-CAM implementation is used to:

- capture intermediate activations
- compute gradient-weighted feature importance
- generate volumetric attention maps
- visualise activation behaviour across anatomical planes

Generated attention maps are analysed under both clean and degraded imaging conditions.

---

## 4. Robustness Analysis

To investigate model stability under distribution shift, Gaussian perturbations are applied to volumetric inputs.

The pipeline evaluates:

- prediction confidence degradation
- Grad-CAM activation drift
- changes in spatial attention behaviour
- robustness of volumetric representations under noisy conditions

Activation drift is quantified using voxel-wise differences between clean and perturbed Grad-CAM maps.

---

# Installation

```bash
pip install -r requirements.txt
```

---

# Training

```bash
python train.py
```

---

# Grad-CAM and Robustness Analysis

```bash
python analysis.py
```

---

# Output

The pipeline generates:

- Grad-CAM visualisations
- perturbed-image explainability maps
- confusion matrices
- activation drift statistics
- robustness evaluation metrics
- CSV result summaries

---



Future work may include:

- motion artefact simulation
- scanner-domain variability
- reconstruction-noise analysis
- multimodal integration
- transformer-based volumetric representations

---

# Research Focus

The primary goal of this project is to explore:

> How stable are deep-learning-based volumetric attention mechanisms under degraded imaging conditions?

rather than simply maximising classification performance.

The broader motivation is understanding reliability and interpretability behaviour in medical imaging systems operating under realistic distribution shift and noisy acquisition conditions.

---

# Notes

Generated datasets, trained checkpoints, processed CT volumes, and output visualisations are intentionally not publicly included.
