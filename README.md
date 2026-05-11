# 🧠 Autism Screening Explainability
### Interpretable Machine Learning for Pediatric ASD Screening

**Suhani Sahai** | Behavioral Interventionist, ICAN | De Anza College  
📧 ssahai0307@gmail.com

---

## Overview

This project applies interpretable machine learning to pediatric autism spectrum disorder (ASD) screening using the Q-CHAT behavioral dataset. Unlike typical ML demos that stop at accuracy numbers, this pipeline connects model predictions back to **real ABA therapy goal areas** using SHAP explainability — bridging the gap between data science and clinical practice.

Built by a behavioral interventionist with **300+ hours of direct ABA therapy experience**, this project reflects a unique perspective: understanding not just *what* the model predicts, but *why* — and what that means for a child sitting across from you in a one-on-one session.

---

## Key Findings

### 🔑 Feature Scaling Changes Everything
SVM accuracy jumped from **50.8% → 98.3%** after applying `StandardScaler`. This is a critical preprocessing step frequently overlooked in clinical ML pipelines — and a concrete, reproducible finding.

### 🔑 SHAP Reveals Clinical Meaning
SHAP values show not just *which* features predict ASD, but *how much* each feature pushes the prediction toward YES or NO for each individual child — directly aligned with ABA therapy goal tracking.

### 🔑 Communication & Social Responsiveness Dominate
Questions A1, A2, A4, and A9 (name response, eye contact, pointing to share interest, gestures) are the strongest predictors — consistent with the focus areas in real one-on-one ABA sessions at ICAN.

---

## Dataset

- **Source**: [Autism Screening — Child Data (Kaggle)](https://www.kaggle.com/datasets/fabdelja/autism-screening-for-toddlers)
- **Size**: 292 children × 21 features
- **Target**: ASD diagnosis (YES / NO) — balanced: 151 NO, 141 YES
- **Leakage removed**: `result` and `age_desc` dropped before modeling

### Behavioral Screening Questions (A1–A10)

| Feature | Question |
|---------|----------|
| A1 | Does your child look at you when you call his/her name? |
| A2 | How easy is it to get eye contact with your child? |
| A3 | Does your child point to indicate that s/he wants something? |
| A4 | Does your child point to share interest with you? |
| A5 | Does your child pretend? (e.g. care for dolls, toy phone) |
| A6 | Does your child follow where you're looking? |
| A7 | If visibly upset, does your child show signs of comfort? |
| A8 | Would you describe your child's first words as typical? |
| A9 | Does your child use simple gestures? |
| A10 | Does your child stare at nothing with no apparent purpose? |

---

## Pipeline

```
Data Loading → Leakage Check → Correlation Analysis
     → StandardScaler → 6 Model Training → SHAP Explainability
```

1. **Preprocessing** — dropped leakage columns, encoded categoricals, imputed missing values
2. **Correlation analysis** — verified no high-correlation leakage (A4 highest at r=0.569)
3. **Feature scaling** — `StandardScaler` applied (critical for SVM and KNN)
4. **Model comparison** — 6 classifiers evaluated on Accuracy, AUC-ROC, and StratifiedKFold CV
5. **SHAP explainability** — feature importance mapped to clinical meaning

---

## Results

| Model | Accuracy | AUC-ROC | CV-AUC |
|-------|----------|---------|--------|
| **Logistic Regression** | **0.983** | **1.000** | **1.000** |
| Support Vector Machine | 0.983 | 1.000 | 0.994 |
| Random Forest | 0.966 | 0.997 | 0.981 |
| Gradient Boosting | 0.915 | 0.975 | 0.981 |
| Decision Tree | 0.898 | 0.898 | 0.859 |
| K-Nearest Neighbor | 0.831 | 0.943 | 0.948 |

**Best Model: Logistic Regression** (AUC: 1.000, CV-AUC: 1.000)

```
Classification Report — Logistic Regression:
              precision    recall  f1-score
          NO       1.00      0.97      0.98
         YES       0.97      1.00      0.98
    accuracy                           0.98
```

---

## Outputs

| File | Description |
|------|-------------|
| `outputs/suhani_asd_poc_v4_results.png` | Model comparison chart, confusion matrix, correlation heatmap, v3 vs v4 scaling impact |
| `outputs/suhani_asd_poc_v4_shap.png` | SHAP summary plot — feature importance with clinical annotations |

---

## Why This Matters Clinically

> *"In 300+ hours of ABA therapy sessions at ICAN, I've watched children respond — or not respond — to their name, make eye contact, reach out and point. These aren't just features in a dataset. They're the exact behaviors we track, celebrate, and build goals around every single session."*
> — Suhani Sahai

SHAP explainability is used in state-of-the-art clinical AI research (including TRIAD's behavioral escalation prediction work) because it makes model decisions **auditable and interpretable** — a non-negotiable requirement for any tool used near clinical decision-making.

---

## Setup & Usage

```bash
# Clone the repo
git clone https://github.com/ssahai03/autism-screening-explainability.git
cd autism-screening-explainability

# Install dependencies
pip install pandas numpy scikit-learn matplotlib seaborn shap scipy

# Run the pipeline
python poc_v4.py
```

**Requirements**: Python 3.8+, see above for packages.

---

## What's New in v4

- ✅ `StandardScaler` applied — SVM accuracy fixed (50.8% → 98.3%)
- ✅ Correlation analysis — leakage verified and cleared
- ✅ SHAP explainability — *why* the model predicts ASD
- ✅ `StratifiedKFold` CV — more robust evaluation
- ✅ Correlation heatmap — feature relationships visualized
- ✅ v3 vs v4 scaling impact chart

---

## Next Steps

- [ ] Medium article — SHAP clinical insights + the SVM scaling story
- [ ] Kaggle public notebook
- [ ] Outreach to Dr. Warren & Dr. Weitlauf (TRIAD / Vanderbilt)

---

## Author

**Suhani Sahai**  
Behavioral Interventionist, ICAN | De Anza College Student  
📧 ssahai0307@gmail.com | 🐙 [@ssahai03](https://github.com/ssahai03)

*This project was developed as part of an independent ML learning curriculum supervised by Nitin Sahai.*

---

*Dataset used for educational and research purposes only. Not intended for clinical diagnosis.*
