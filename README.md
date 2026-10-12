# 🩺 Automated Multi-Class Skin Lesion Classification using DenseNet121 & Class-Weighted Cross-Entropy Loss

[![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)](https://pytorch.org/)
[![ISIC 2019](https://img.shields.io/badge/Dataset-ISIC%202019-blue?style=for-the-badge)](https://challenge.isic-archive.com/landing/2019/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg?style=for-the-badge)](https://opensource.org/licenses/MIT)

## 📌 Executive Summary
Dermatological diagnostics face critical challenges due to extreme class imbalance between benign conditions and rare, high-fatality malignancies like Melanoma and Squamous Cell Carcinoma (SCC). This project presents an end-to-end deep learning framework utilizing a fine-tuned **DenseNet121** architecture combined with **Class-Weighted Cross-Entropy Loss** to classify skin lesions across 4 primary categories from the ISIC 2019 dataset.

By penalizing misclassifications on minority diagnostic classes, the model achieved an overall **80.47% Test Accuracy** while drastically improving clinical sensitivity on **Melanoma (83% Recall)** and **SCC (76% Recall)**.

---

## 📊 Dataset & Experimental Split Strategy
The model was trained and evaluated on the **ISIC 2019 Dataset**, comprising 25,331 labeled dermoscopic images. 

To ensure realistic clinical generalization and address class imbalance, a strict **80 / 10 / 10 Stratified Split** was applied:

| Dataset Split | Percentage | Number of Images | Purpose |
| :--- | :---: | :---: | :--- |
| **Training Set** | **80%** | **20,264** | Model parameter updates & feature extraction |
| **Validation Set** | **10%** | **2,533** | Dynamic hyperparameter tuning & early stopping check |
| **Held-out Test Set** | **10%** | **2,534** | Final unbiased performance evaluation |

### Diagnostic Class Distribution
* **Benign:** ~66.7% (13,486 Train / 1,686 Test)
* **Melanoma:** ~17.8% (3,618 Train / 452 Test)
* **Basal Cell Carcinoma (BCC):** ~13.1% (2,658 Train / 333 Test)
* **Squamous Cell Carcinoma (SCC):** ~2.5% (502 Train / 63 Test)

---

## 🛠️ Methodology & Technical Architecture
1. **Backbone Architecture:** DenseNet121 pretrained on ImageNet, utilizing dense feature reuse to preserve fine dermatological patterns.
2. **Loss Function Optimization:** Dynamic inverse class weighting added to standard Cross-Entropy Loss to counter majority-class dominance:
   $$\text{Weight}_c = \frac{N_{\text{total}}}{C \times N_c}$$
3. **Data Augmentation:** Random Horizontal Flips, Random Vertical Flips, Tensor Normalization ($\mu=[0.485, 0.456, 0.406]$, $\sigma=[0.229, 0.224, 0.225]$).
4. **Optimization:** Adam Optimizer ($\text{lr}=10^{-4}$), Batch Size of 32, evaluated across 10 complete epochs on NVIDIA T4 GPU.

---

## 📈 Quantitative Results

### Overall Held-out Test Metrics (2,534 Images)
| Metric | Value |
| :--- | :---: |
| **Test Accuracy** | **80.47%** |
| **Weighted Precision** | **0.8447** |
| **Weighted Recall (Sensitivity)** | **0.8047** |
| **Weighted F1-Score** | **0.8139** |

### Per-Class Performance Breakdown
| Diagnostic Class | Precision | Recall (Sensitivity) | F1-Score | Support (Test Images) |
| :--- | :---: | :---: | :---: | :---: |
| **Melanoma** | 0.59 | **0.83** | 0.69 | 452 |
| **BCC** | 0.75 | **0.89** | 0.82 | 333 |
| **SCC** | 0.44 | **0.76** | 0.56 | 63 |
| **Benign** | **0.95** | 0.78 | 0.86 | 1,686 |

---

## 🖼️ Visual Evaluation Artifacts

<p align="center">
  <img src="assets/Figure_1_Confusion_Matrix.png" width="45%" alt="Confusion Matrix"/>
  <img src="assets/ROC_Curves.png>
</p>

* **Confusion Matrix (Left):** Demonstrates strong diagonal classification across all classes, explicitly controlling false negative rates for Melanoma and SCC.
* **Multi-Class ROC Curves (Right):** Evaluates area under the curve (AUC) performance across all diagnostic thresholds.

---

