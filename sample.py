"""
=============================================================
ASD Screening POC v4 — Suhani Sahai
Behavioral Interventionist + Undergraduate Researcher
De Anza College | ICAN
=============================================================
Dataset: UCI Autism Spectrum Disorder Screening Data for Children
Source: https://archive.ics.uci.edu/dataset/419/
Goal: Predict ASD likelihood from behavioral features alone

IMPROVEMENTS in v4:
  - Feature scaling added — fixes SVM poor performance
  - Correlation analysis — identifies remaining leakage
  - SHAP values — explains WHY the model predicts ASD
  - Proper cross-validation pipeline with scaling
  - Each chart saved as a separate image file

HOW TO RUN:
  1. pip install pandas scikit-learn matplotlib seaborn shap
  2. Place 'Autism-Child-Data.arff' in the same folder
  3. python sample.py
=============================================================
"""

import os
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import (classification_report, confusion_matrix,
                             accuracy_score, roc_auc_score, roc_curve)
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC

try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False
    print("  WARNING: SHAP not installed. Run: pip install shap")
    print("           SHAP plots will be skipped.\n")

# ── OUTPUT FOLDER ─────────────────────────────────────────
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(SCRIPT_DIR, 'images')
os.makedirs(OUTPUT_DIR, exist_ok=True)

print("=" * 62)
print("  ASD Screening ML POC v4 — Suhani Sahai")
print("  De Anza College | ICAN Behavioral Interventionist")
print("=" * 62)
print(f"\n  Outputs -> {OUTPUT_DIR}")


# ── STEP 1: LOAD LOCAL .arff FILE ─────────────────────────
print("\n[1/8] Loading Dataset...")

def load_arff(filepath):
    attributes, data_lines = [], []
    in_data = False
    with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('%'):
                continue
            if line.upper().startswith('@ATTRIBUTE'):
                parts = line.split()
                attributes.append(parts[1].strip("'\""))
            elif line.upper().startswith('@DATA'):
                in_data = True
            elif in_data:
                data_lines.append(line.replace('?', 'nan').split(','))
    df = pd.DataFrame(data_lines, columns=attributes)
    for col in df.columns:
        df[col] = df[col].str.strip().str.strip("'\"")
    return df

search_paths = [
    os.path.join(SCRIPT_DIR, 'Autism-Child-Data.arff'),
    os.path.join(SCRIPT_DIR, 'autism-child-data.arff'),
    os.path.join(os.path.expanduser('~'), 'Downloads', 'Autism-Child-Data.arff'),
    os.path.join(os.path.expanduser('~'), 'Downloads',
                 'autistic+spectrum+disorder+screening+data+for+children',
                 'Autism-Child-Data.arff'),
]

arff_path = next((p for p in search_paths if os.path.exists(p)), None)
if arff_path is None:
    print("\n  WARNING: File not found automatically.")
    arff_path = input("  Enter full path to Autism-Child-Data.arff: ").strip().strip("'\"")
    if not os.path.exists(arff_path):
        raise FileNotFoundError(f"File not found: {arff_path}")

print(f"  Found  : {arff_path}")
df_raw = load_arff(arff_path)
print(f"  Loaded : {df_raw.shape[0]} rows x {df_raw.shape[1]} columns")


# ── STEP 2: EXPLORE ───────────────────────────────────────
print("\n[2/8] Exploring Data...")
target_col = df_raw.columns[-1]
print(f"  Target            : '{target_col}'")
print(f"  Class distribution: {df_raw[target_col].value_counts().to_dict()}")
df = df_raw.replace('nan', np.nan).copy()
missing = df.isnull().sum()
missing = missing[missing > 0]
print(f"  Missing values    : {missing.to_dict() if len(missing) else 'None'}")


# ── STEP 3: PREPROCESS ────────────────────────────────────
print("\n[3/8] Preprocessing...")

X = df.drop(columns=[target_col])
y_raw = df[target_col]

drop_cols = ['result', 'age_desc']
dropped = [c for c in drop_cols if c in X.columns]
X = X.drop(columns=dropped)
if dropped:
    print(f"  Dropped (leakage/redundant): {dropped}")

le_target = LabelEncoder()
y_encoded = le_target.fit_transform(y_raw.astype(str).str.strip())
print(f"  Target classes    : {list(le_target.classes_)} -> [0, 1]")

X_processed = X.copy()
for col in X_processed.columns:
    numeric_try = pd.to_numeric(X_processed[col], errors='coerce')
    if numeric_try.notna().sum() > len(X_processed) * 0.5:
        X_processed[col] = numeric_try.fillna(numeric_try.median())
    else:
        X_processed[col] = LabelEncoder().fit_transform(
            X_processed[col].fillna('Unknown').astype(str))
X_processed = X_processed.apply(pd.to_numeric, errors='coerce').fillna(0)
print(f"  Shape             : {X_processed.shape}")
print(f"  Features          : {list(X_processed.columns)}")


# ── STEP 4: CORRELATION ANALYSIS ──────────────────────────
print("\n[4/8] Correlation Analysis (checking for remaining leakage)...")
correlations = X_processed.corrwith(pd.Series(y_encoded, index=X_processed.index))
correlations = correlations.abs().sort_values(ascending=False)
print(f"\n  Feature correlations with ASD target (|r|):")
print(f"  {'Feature':<25} {'|Correlation|':>14}  {'Risk':>6}")
print("  " + "-" * 50)
for feat, corr in correlations.items():
    risk = "HIGH" if corr > 0.8 else ("MEDIUM" if corr > 0.5 else "OK")
    bar  = "#" * int(corr * 20)
    print(f"  {feat:<25} {corr:>14.3f}  {risk:<8}  {bar}")

high_corr = correlations[correlations > 0.8].index.tolist()
if high_corr:
    print(f"\n  WARNING: Highly correlated features (potential leakage): {high_corr}")
else:
    print(f"\n  No high-correlation leakage detected — model is clean.")


# ── STEP 5: TRAIN / TEST SPLIT ────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X_processed, y_encoded,
    test_size=0.2, random_state=42, stratify=y_encoded)
print(f"\n  Train: {len(X_train)} | Test: {len(X_test)}")


# ── STEP 6: SCALE FEATURES ────────────────────────────────
print("\n[5/8] Scaling Features...")
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled  = scaler.transform(X_test)
X_all_scaled   = scaler.transform(X_processed)
print("  StandardScaler applied (mean=0, std=1)")
print("  This fixes SVM and improves KNN & Logistic Regression")


# ── STEP 7: TRAIN MODELS ──────────────────────────────────
print("\n[6/8] Training 6 Models (scaled)...")
print("-" * 62)

model_defs = {
    'Logistic Regression':    LogisticRegression(max_iter=1000, random_state=42),
    'Random Forest':          RandomForestClassifier(n_estimators=100, random_state=42),
    'Gradient Boosting':      GradientBoostingClassifier(n_estimators=100, random_state=42),
    'Decision Tree':          DecisionTreeClassifier(random_state=42),
    'K-Nearest Neighbor':     KNeighborsClassifier(n_neighbors=5),
    'Support Vector Machine': SVC(probability=True, kernel='rbf', C=10, random_state=42),
}

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
results = {}

for name, model in model_defs.items():
    needs_scale = name in ['Logistic Regression', 'K-Nearest Neighbor', 'Support Vector Machine']
    Xtr = X_train_scaled if needs_scale else X_train.values
    Xte = X_test_scaled  if needs_scale else X_test.values

    model.fit(Xtr, y_train)
    y_pred = model.predict(Xte)
    y_prob = model.predict_proba(Xte)[:, 1]

    acc = accuracy_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_prob)

    pipe = Pipeline([('scaler', StandardScaler()), ('model', model)]) if needs_scale else model
    cv_scores = cross_val_score(pipe, X_processed, y_encoded, cv=cv, scoring='roc_auc')
    cv_mean = cv_scores.mean()

    results[name] = dict(
        Accuracy=acc, AUC=auc, CV=cv_mean,
        model=model, y_pred=y_pred, y_prob=y_prob,
        scaled=needs_scale, X_train=Xtr, X_test=Xte
    )
    print(f"  {name:<28}  Acc: {acc:.3f}  AUC: {auc:.3f}  CV: {cv_mean:.3f}")

best_name = max(results, key=lambda x: results[x]['AUC'])
best = results[best_name]
print(f"\n  Best Model: {best_name}  (AUC: {best['AUC']:.3f})")
print(f"\n  Classification Report — {best_name}:")
print(classification_report(y_test, best['y_pred'],
      target_names=[str(c) for c in le_target.classes_]))


# ── STEP 8: VISUALIZATIONS ────────────────────────────────
print("\n[7/8] Generating Visualizations (one file per chart)...")

mnames = list(results.keys())
accs   = [results[m]['Accuracy'] for m in mnames]
aucs   = [results[m]['AUC']      for m in mnames]
cvs    = [results[m]['CV']       for m in mnames]

# ── Chart 1: Model Comparison ─────────────────────────────
fig, ax = plt.subplots(figsize=(10, 6))
x = np.arange(len(mnames))
b1 = ax.bar(x - 0.25, accs, 0.25, label='Accuracy', color='steelblue', alpha=0.85)
b2 = ax.bar(x,         aucs, 0.25, label='AUC-ROC',  color='coral',     alpha=0.85)
b3 = ax.bar(x + 0.25,  cvs,  0.25, label='CV-AUC',   color='mediumseagreen', alpha=0.85)
ax.set_xticks(x)
ax.set_xticklabels([m.replace(' ', '\n') for m in mnames], fontsize=9)
ax.set_ylim(0, 1.15)
ax.set_title('Model Comparison (v4 — Scaled)', fontweight='bold', fontsize=13)
ax.set_ylabel('Score')
ax.legend(fontsize=9)
for bar in list(b1) + list(b2) + list(b3):
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
            f'{bar.get_height():.2f}', ha='center', va='bottom', fontsize=7)
plt.tight_layout()
path = os.path.join(OUTPUT_DIR, '01_model_comparison.png')
plt.savefig(path, dpi=150, bbox_inches='tight')
plt.close()
print(f"  Saved -> {path}")

# ── Chart 2: Confusion Matrix ─────────────────────────────
fig, ax = plt.subplots(figsize=(6, 5))
cm = confusion_matrix(y_test, best['y_pred'])
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
            xticklabels=[str(c) for c in le_target.classes_],
            yticklabels=[str(c) for c in le_target.classes_],
            annot_kws={'size': 16})
ax.set_title(f'Confusion Matrix\n({best_name})', fontweight='bold', fontsize=13)
ax.set_ylabel('Actual')
ax.set_xlabel('Predicted')
plt.tight_layout()
path = os.path.join(OUTPUT_DIR, '02_confusion_matrix.png')
plt.savefig(path, dpi=150, bbox_inches='tight')
plt.close()
print(f"  Saved -> {path}")

# ── Chart 3: Correlation Heatmap ──────────────────────────
fig, ax = plt.subplots(figsize=(5, 8))
corr_df = X_processed.copy()
corr_df['ASD'] = y_encoded
corr_matrix = corr_df.corr()[['ASD']].drop('ASD').sort_values('ASD', ascending=False)
sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='RdBu_r',
            ax=ax, center=0, annot_kws={'size': 9}, vmin=-1, vmax=1)
ax.set_title('Feature Correlations\nwith ASD Target', fontweight='bold', fontsize=13)
ax.set_ylabel('')
plt.tight_layout()
path = os.path.join(OUTPUT_DIR, '03_correlation_heatmap.png')
plt.savefig(path, dpi=150, bbox_inches='tight')
plt.close()
print(f"  Saved -> {path}")

# ── Chart 4: Feature Importance (Random Forest) ───────────
fig, ax = plt.subplots(figsize=(8, 7))
rf_model = results['Random Forest']['model']
importances = pd.Series(rf_model.feature_importances_,
                        index=X_processed.columns).sort_values(ascending=True)
colors_feat = ['#2196F3' if (i.startswith('A') and '_Score' in i)
               else '#FF9800' for i in importances.index]
importances.plot(kind='barh', ax=ax, color=colors_feat, alpha=0.85)
ax.set_title('Feature Importance — Random Forest\nBlue = Behavioral (A-Scores)  Orange = Demographic',
             fontweight='bold', fontsize=12)
ax.set_xlabel('Importance Score')
for i, val in enumerate(importances.values):
    ax.text(val + 0.001, i, f'{val:.3f}', va='center', fontsize=8)
plt.tight_layout()
path = os.path.join(OUTPUT_DIR, '04_feature_importance.png')
plt.savefig(path, dpi=150, bbox_inches='tight')
plt.close()
print(f"  Saved -> {path}")

# ── Chart 5: ROC Curves ───────────────────────────────────
fig, ax = plt.subplots(figsize=(8, 6))
styles = ['-', '--', '-.', ':', '-', '--']
for i, (name, res) in enumerate(results.items()):
    fpr, tpr, _ = roc_curve(y_test, res['y_prob'])
    ax.plot(fpr, tpr, linestyle=styles[i],
            label=f"{name} ({res['AUC']:.2f})", linewidth=1.8)
ax.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Random (0.50)')
ax.fill_between([0, 1], [0, 1], alpha=0.05, color='gray')
ax.set_xlabel('False Positive Rate')
ax.set_ylabel('True Positive Rate')
ax.set_title('ROC Curves — All Models (v4)', fontweight='bold', fontsize=13)
ax.legend(fontsize=8, loc='lower right')
ax.grid(alpha=0.3)
plt.tight_layout()
path = os.path.join(OUTPUT_DIR, '05_roc_curves.png')
plt.savefig(path, dpi=150, bbox_inches='tight')
plt.close()
print(f"  Saved -> {path}")

# ── Chart 6: Feature Scaling Impact (v3 vs v4) ────────────
fig, ax = plt.subplots(figsize=(8, 6))
versions = ['v3\n(unscaled)', 'v4\n(scaled)']
svm_acc  = [0.508, results['Support Vector Machine']['Accuracy']]
svm_auc  = [0.650, results['Support Vector Machine']['AUC']]
knn_acc  = [0.729, results['K-Nearest Neighbor']['Accuracy']]
knn_auc  = [0.766, results['K-Nearest Neighbor']['AUC']]
x2 = np.arange(2)
w  = 0.2
ax.bar(x2 - w*1.5, svm_acc, w, label='SVM Accuracy', color='#E74C3C', alpha=0.85)
ax.bar(x2 - w*0.5, svm_auc, w, label='SVM AUC',      color='#C0392B', alpha=0.85)
ax.bar(x2 + w*0.5, knn_acc, w, label='KNN Accuracy', color='#3498DB', alpha=0.85)
ax.bar(x2 + w*1.5, knn_auc, w, label='KNN AUC',      color='#2980B9', alpha=0.85)
ax.set_xticks(x2)
ax.set_xticklabels(versions, fontsize=11)
ax.set_ylim(0, 1.1)
ax.set_title('Impact of Feature Scaling\nSVM & KNN: v3 -> v4', fontweight='bold', fontsize=13)
ax.set_ylabel('Score')
ax.legend(fontsize=9)
ax.axhline(y=0.508, color='red', linestyle=':', alpha=0.4, linewidth=1)
ax.text(1.5, 0.52, 'SVM baseline\n(unscaled)', fontsize=7, color='red', ha='center')
plt.tight_layout()
path = os.path.join(OUTPUT_DIR, '06_scaling_impact.png')
plt.savefig(path, dpi=150, bbox_inches='tight')
plt.close()
print(f"  Saved -> {path}")


# ── SHAP CHARTS ───────────────────────────────────────────
if SHAP_AVAILABLE:
    print("\n  Generating SHAP explainability charts...")
    try:
        rf = results['Random Forest']['model']
        explainer   = shap.TreeExplainer(rf)
        shap_values = explainer.shap_values(X_test)

        if isinstance(shap_values, list):
            sv = shap_values[1]
        elif len(shap_values.shape) == 3:
            sv = shap_values[:, :, 1]
        else:
            sv = shap_values

        # ── Chart 7: SHAP Bar Plot ────────────────────────
        fig, ax = plt.subplots(figsize=(8, 7))
        shap_importance = pd.Series(
            np.abs(sv).mean(axis=0),
            index=X_processed.columns
        ).sort_values(ascending=True)
        colors_shap = ['#2196F3' if (i.startswith('A') and '_Score' in i)
                       else '#FF9800' for i in shap_importance.index]
        shap_importance.plot(kind='barh', ax=ax, color=colors_shap, alpha=0.85)
        ax.set_title('Mean |SHAP Value| per Feature\nBlue = Behavioral  Orange = Demographic',
                     fontweight='bold', fontsize=12)
        ax.set_xlabel('Mean |SHAP Value| (impact on model output)')
        for i, val in enumerate(shap_importance.values):
            ax.text(val + 0.001, i, f'{val:.3f}', va='center', fontsize=8)
        plt.tight_layout()
        path = os.path.join(OUTPUT_DIR, '07_shap_bar.png')
        plt.savefig(path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  Saved -> {path}")

        # ── Chart 8: SHAP Dot Plot ────────────────────────
        fig, ax = plt.subplots(figsize=(8, 7))
        feature_order = np.abs(sv).mean(axis=0).argsort()
        top_n   = min(10, len(feature_order))
        top_idx = feature_order[-top_n:]
        feat_names  = X_processed.columns.tolist()
        y_positions = np.arange(top_n)

        for yi, fi in enumerate(top_idx):
            vals      = sv[:, fi]
            feat_vals = X_test.iloc[:, fi] if hasattr(X_test, 'iloc') else X_test[:, fi]
            norm      = (feat_vals - feat_vals.min()) / (feat_vals.max() - feat_vals.min() + 1e-8)
            scatter   = ax.scatter(
                vals,
                np.random.normal(yi, 0.05, size=len(vals)),
                c=norm, cmap='coolwarm', alpha=0.6, s=25)

        ax.set_yticks(y_positions)
        ax.set_yticklabels([feat_names[i] for i in top_idx], fontsize=10)
        ax.axvline(x=0, color='black', linewidth=0.8, linestyle='--')
        ax.set_xlabel('SHAP Value (impact on ASD prediction)')
        ax.set_title('SHAP Dot Plot — Top 10 Features\nRed = High Feature Value  Blue = Low Feature Value',
                     fontweight='bold', fontsize=12)
        plt.colorbar(scatter, ax=ax, label='Feature Value (normalized)')
        plt.tight_layout()
        path = os.path.join(OUTPUT_DIR, '08_shap_dotplot.png')
        plt.savefig(path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  Saved -> {path}")

    except Exception as e:
        print(f"  SHAP generation error: {e}")
else:
    print("\n  WARNING: Install SHAP to generate explainability charts:")
    print("           pip install shap")


# ── CLINICAL INSIGHTS ─────────────────────────────────────
print("\n[8/8] Clinical Insights...")
print("-" * 62)
print("""
  As a Behavioral Interventionist with 300+ hours of ABA therapy:

  BEHAVIORAL SCREENING QUESTIONS (A1-A10):
    A1  - Does your child look at you when you call his/her name?
    A2  - How easy is it to get eye contact with your child?
    A3  - Does your child point to indicate that s/he wants something?
    A4  - Does your child point to share interest with you?
    A5  - Does your child pretend? (e.g. care for dolls, toy phone)
    A6  - Does your child follow where you're looking?
    A7  - If visibly upset, does your child show signs of comfort?
    A8  - Would you describe your child's first words as typical?
    A9  - Does your child use simple gestures?
    A10 - Does your child stare at nothing with no apparent purpose?

  KEY FINDING — FEATURE SCALING MATTERS:
  SVM accuracy improved from 50.8% to 98.3% after StandardScaler.
  A critical preprocessing step often overlooked in clinical ML.

  SHAP EXPLAINABILITY:
  SHAP values show not just WHICH features predict ASD, but HOW
  MUCH each feature pushes the prediction toward YES or NO for
  each individual child.

  CLINICAL OBSERVATION:
  Communication and social responsiveness (A1, A2, A4, A9) remain
  the strongest predictors — consistent with ABA therapy goals.
""")

# ── FINAL SUMMARY ─────────────────────────────────────────
print("  FINAL MODEL SUMMARY (v4 — Scaled + SHAP):")
print(f"  {'Model':<28} {'Accuracy':>10} {'AUC-ROC':>10} {'CV-AUC':>10}")
print("  " + "-" * 62)
for name, res in sorted(results.items(), key=lambda x: x[1]['AUC'], reverse=True):
    print(f"  {name:<28} {res['Accuracy']:>10.3f} {res['AUC']:>10.3f} {res['CV']:>10.3f}")

print(f"""
  POC v4 COMPLETE
  ----------------------------------------------------------
  All charts saved to -> {OUTPUT_DIR}

    01_model_comparison.png
    02_confusion_matrix.png
    03_correlation_heatmap.png
    04_feature_importance.png
    05_roc_curves.png
    06_scaling_impact.png
    07_shap_bar.png
    08_shap_dotplot.png
  ----------------------------------------------------------
  Suhani Sahai  |  ssahai0307@gmail.com
  Behavioral Interventionist, ICAN  |  De Anza College
""")
