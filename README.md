# WDCS

Wise Diagnosis Colposcopy System: a structured dual-branch feature-fusion framework for AI-assisted colposcopic diagnosis.

---

## Overview

This repository contains the key Python implementation of **WDCS**, developed for colposcopy-assisted cervical lesion diagnosis under a clinically interpretable feature-fusion pipeline.

The codebase focuses on:

* Four-class colposcopic diagnosis: **Normal, LSIL, HSIL, and Cancer**
* Recognition of six clinically meaningful colposcopic semantic signs
* Lesion region segmentation and lesion-extent delineation
* Decoder-derived lesion spatial feature extraction
* Structured feature-level fusion with a downstream **LightGBM classifier**

⚠️ Note: This repository is a simplified research codebase. It keeps the core model architecture and training pipeline only. Deployment inference scripts, clinical UI modules, ROC plotting, confusion matrix plotting, and complete metric-reporting scripts are intentionally not included.

---

## Methodological Framework

The Python implementation of WDCS includes the following core components:

### 1. Semantic-Sign Recognition Branch

* A shared ImageNet-pretrained backbone is used to extract global image representations.
* Six task-specific classification heads are constructed for colposcopic semantic-sign recognition:
  * Acetowhite epithelium
  * Lesion margin
  * Punctation
  * Mosaic
  * Atypical vessels
  * Inner border sign
* The semantic-sign branch supports backbone replacement, including:
  * EfficientNet / EfficientNetV2
  * ResNet-50
  * ViT-B/16
  * Swin-T

All backbones are evaluated within the same WDCS feature-fusion pipeline rather than as independent vanilla image classifiers.

---

### 2. Segmentation Branch

* A SegFormer-style encoder-decoder branch is implemented for lesion segmentation.
* The segmentation branch produces two outputs:
  * Predicted lesion mask for lesion visualization and lesion-extent delineation
  * Decoder-derived lesion spatial feature vector for downstream structured classification
* The lesion spatial feature is extracted from the decoder fused feature map using global average pooling.

This implementation explicitly treats the spatial feature as a decoder-derived representation, not as hand-crafted mask morphology.

---

### 3. Structured Feature-Level Fusion

For each image, WDCS constructs a structured feature representation using:

* Predicted semantic-sign categories converted into one-hot vectors
* Decoder-derived lesion spatial feature vector from the segmentation branch
* Concatenation of both feature types to form the fused representation

The final diagnosis is obtained by training a downstream structured classifier, with **LightGBM** used as the default model.

The repository also supports a semantic-sign-only ablation by using only the one-hot semantic-sign representation.

---

### 4. Staged Training Pipeline

WDCS is trained in a staged manner:

1. Train the semantic-sign recognition branch using expert-labeled semantic-sign annotations.
2. Train the segmentation branch using lesion mask annotations.
3. Extract semantic-sign predictions and decoder-derived lesion spatial features.
4. Train the downstream LightGBM classifier using the structured feature matrix.

External cohorts should only be used for final testing and should not be used for model selection or hyperparameter tuning.

---

## Data Format

The metadata file can be provided as `.csv`, `.xlsx`, or `.xls`.

Required columns:

```text
image_path
split
diagnosis_label
mask_path
acetowhite_epithelium
lesion_margin
punctation
mosaic
atypical_vessels
inner_border_sign
```

Recommended split values:

```text
train
val
test
external1
external2
```

For semantic-sign recognition, each semantic-sign label should be encoded as an integer category ID.  
For segmentation training, masks should be integer label maps. Samples without valid masks are skipped in the segmentation dataset.

---

## Repository Structure

```text
WDCS_minimal_github/
├── configs/
│   └── default.yaml
├── data/
├── outputs/
├── wdcs/
│   ├── data/
│   ├── models/
│   ├── losses/
│   ├── training/
│   ├── feature_extraction/
│   ├── classical_ml/
│   └── utils/
├── requirements.txt
└── README.md
```

---
## Training Configuration

The main training hyperparameters used in WDCS are summarized below.

### Training Hyperparameters

| Module                           | Model and input                                          | Optimizer and training schedule                                                                                      | Loss / objective                   |
| -------------------------------- | -------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------- | ---------------------------------- |
| Semantic-sign recognition branch | EfficientNetV2-S<br>Input size: 384 × 384                | Optimizer: AdamW<br>Learning rate: 3e-4<br>Batch size: 32<br>Epochs: 100<br>Learning-rate schedule: cosine annealing | Multi-task cross-entropy           |
| Segmentation branch              | SegFormer-B0<br>Input size: 384 × 384                    | Optimizer: AdamW<br>Learning rate: 1e-4<br>Batch size: 32<br>Epochs: 100<br>Learning-rate schedule: cosine annealing | Dice loss + weighted cross-entropy |
| Downstream classifier            | LightGBM<br>Input: fused structured features             | Tree-based structured classifier                                                                                     | Multiclass log-loss objective      |
| Backbone comparison              | ResNet-50<br>ViT-Tiny<br>Swin-T<br>Input size: 384 × 384 | Optimizer: AdamW<br>Learning rate: 3e-4<br>Batch size: 32<br>Epochs: 100<br>Learning-rate schedule: cosine annealing | Multi-task cross-entropy           |

All models were initialized with ImageNet-pretrained weights when available and trained using the same augmentation strategy, including random resized crop, random horizontal flip, mild rotation, and mild color jitter.

### Semantic-sign Recognition Settings

| Semantic sign         | Abbreviation | Number of classes | Loss weight |
| --------------------- | -----------: | ----------------: | ----------- |
| Acetowhite epithelium |           AE |                 3 | Weighted CE |
| Lesion margin         |       Margin |                 3 | Weighted CE |
| Punctation            |           PN |                 3 | Weighted CE |
| Mosaic                |           MC |                 3 | Weighted CE |
| Atypical vessels      |           AV |                 2 | Weighted CE |
| Inner border sign     |          IBS |                 2 | Weighted CE |

In the semantic-sign recognition branch, six task-specific classification heads share the same visual backbone. The total semantic-sign loss is computed as the sum of the six weighted cross-entropy losses. Class weights are used to reduce the effect of class imbalance within each semantic-sign task.

All outputs are saved under the `outputs/` directory using reproducible file names.

---

## Disclaimer

This repository is intended for research purposes only. The model outputs should not be used as standalone clinical decisions. Lesion delineation and semantic-sign recognition results provide auxiliary spatial and structured information and should be interpreted by qualified clinicians in appropriate clinical contexts.
