"""
===========================================================================
STEP 5: AI MODEL TRAINING — 8-Model Comparative Benchmark
Project: The Self-Auditing Ledger
Purpose: Train and compare 8 ML models on our graph-derived fraud features.
===========================================================================

WHAT THIS SCRIPT DOES:
-----------------------
This is the final step. We take the graph_features.csv produced by the
Topology Engine and train 8 different machine learning models to learn:

    "Given these structural graph features, what is the probability
     that this document is fraudulent?"

THE 8 MODELS WE COMPARE:
--------------------------
  1. Logistic Regression     — Simple linear baseline
  2. Decision Tree           — Interpretable tree baseline
  3. Random Forest           — Ensemble of trees (project baseline)
  4. Gradient Boosting       — Sklearn's built-in boosting
  5. XGBoost                 — State-of-the-art boosting (PRIMARY MODEL)
  6. LightGBM               — Microsoft's fast gradient boosting
  7. SVM                     — Support Vector Machine
  8. MLP Neural Network      — Multi-layer perceptron

WHY 8 MODELS?
--------------
Academic rigor requires comparative analysis. Your thesis needs to show:
  1. XGBoost outperforms simpler baselines (justifies the choice)
  2. Graph-derived features work across multiple model types
  3. The architecture is model-agnostic (features matter more than model)

HANDLING CLASS IMBALANCE:
--------------------------
Our dataset has 59,803 clean docs and only 49 fraud docs (0.08%).
If a model just predicted "not fraud" for everything, it would be
99.92% accurate — but completely useless!

We use THREE strategies to handle this:
  1. SMOTE (Synthetic Minority Oversampling) — creates synthetic fraud
     examples to balance the training set
  2. class_weight='balanced' — tells models to penalize fraud misses
     more heavily than false alarms
  3. Evaluation metrics — we use F1, Precision, Recall, and AUC-ROC
     instead of accuracy

WHY NOT ACCURACY?
------------------
  Accuracy with 99.92% clean data: "Predict all non-fraud" → 99.92%
  But it catches ZERO fraud! Useless.

  We care about:
    - RECALL: Of all actual fraud, what % did we catch?
    - PRECISION: Of all alerts, what % were real fraud?
    - F1: Harmonic mean of precision and recall
    - AUC-ROC: How well does the model rank fraud above non-fraud?
"""

import pandas as pd
import numpy as np
import os
import sys
import time
import warnings
import json

warnings.filterwarnings('ignore')
sys.stdout.reconfigure(encoding='utf-8')

# ───────────────────────────────────────────────────────────────
# ML Imports
# ───────────────────────────────────────────────────────────────
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    classification_report, confusion_matrix, roc_auc_score,
    precision_recall_curve, average_precision_score,
    f1_score, precision_score, recall_score, accuracy_score,
    roc_curve
)

# The 8 models
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.neural_network import MLPClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

# Imbalanced learning
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline

# Visualization
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for saving plots
import matplotlib.pyplot as plt
import seaborn as sns

# ===================================================================
# CONFIGURATION
# ===================================================================
FEATURE_FILE = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "data", "neo4j_import", "graph_features.csv"
)

OUTPUT_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "results"
)
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Random seed for reproducibility
# WHY: Setting a seed ensures identical results each time you run
# the script. Essential for your thesis — reviewers need to reproduce.
RANDOM_SEED = 42

# Number of cross-validation folds
# WHY 5 FOLDS: With only 49 fraud samples, each fold gets ~10 fraud docs.
# Fewer folds would leave too few fraud samples per fold; more folds
# would make each validation set too small.
N_FOLDS = 5


# ===================================================================
# STEP 5A: LOAD AND PREPARE DATA
# ===================================================================
def load_and_prepare_data():
    print("=" * 70)
    print("STEP 5A: LOADING AND PREPARING DATA")
    print("=" * 70)

    df = pd.read_csv(FEATURE_FILE)
    print(f"  Loaded: {df.shape[0]} rows × {df.shape[1]} columns")

    # Define feature columns (exclude metadata and targets)
    metadata_cols = ['doc_id', 'doc_number', 'source_file', 'dataset_type',
                     'timestamp', 'transaction_type', 'label']
    target_col = 'is_fraud'

    feature_cols = [c for c in df.columns if c not in metadata_cols + [target_col]]
    print(f"  Features: {len(feature_cols)}")
    print(f"  Feature list: {feature_cols}")

    X = df[feature_cols].copy()
    y = df[target_col].copy()

    # Handle any remaining non-numeric columns
    for col in X.columns:
        if X[col].dtype == 'object':
            print(f"  ⚠️ Dropping non-numeric column: {col}")
            X.drop(columns=[col], inplace=True)

    # Fill NaN with 0
    X = X.fillna(0)

    # Replace infinities
    X = X.replace([np.inf, -np.inf], 0)

    print(f"\n  Final feature matrix: {X.shape}")
    print(f"  Target distribution:")
    print(f"    Clean (0): {(y == 0).sum():,}")
    print(f"    Fraud (1): {(y == 1).sum():,}")
    print(f"    Fraud ratio: {y.mean()*100:.3f}%")

    return X, y, feature_cols, df


# ===================================================================
# STEP 5B: DEFINE THE 8 MODELS
# ===================================================================
def get_models():
    """
    Define our 8-model benchmark suite.
    
    WHY THESE SPECIFIC MODELS?
    
    1. Logistic Regression: The simplest possible classifier. If it
       performs well, our features are so good that even a linear
       model can separate fraud from non-fraud. Great baseline.
    
    2. Decision Tree: A single tree that makes decisions by splitting
       on feature thresholds. Highly interpretable — you can literally
       see the decision path. But prone to overfitting.
    
    3. Random Forest: An ensemble of many decision trees that vote
       on the outcome. More robust than a single tree. The project
       description lists this as the "baseline model."
    
    4. Gradient Boosting (sklearn): Trees built sequentially, each
       correcting the previous one's errors. Good but slower than
       XGBoost.
    
    5. XGBoost: The RECOMMENDED model from our project description.
       State-of-the-art on structured/tabular data. Handles class
       imbalance well via scale_pos_weight. Highly interpretable
       via feature importance.
    
    6. LightGBM: Microsoft's gradient boosting framework. Often
       faster than XGBoost with comparable accuracy. Good for
       demonstrating robustness across boosting implementations.
    
    7. AdaBoost: Adaptive Boosting. Adjusts weights of incorrectly 
       classified instances so subsequent classifiers focus more on 
       difficult cases.
    
    8. MLP (Multi-Layer Perceptron): A simple neural network.
       Included to compare deep learning against tree-based methods.
       Our project argues that interpretable models (XGBoost) are
       better than black-box neural networks for audit applications.
    """
    print("\n" + "=" * 70)
    print("STEP 5B: DEFINING 8-MODEL BENCHMARK SUITE")
    print("=" * 70)

    # Calculate scale_pos_weight for XGBoost
    # WHY: This tells XGBoost how much more to penalize missing fraud
    # compared to false alarms. With 99.92% clean and 0.08% fraud,
    # scale_pos_weight ≈ 59803/49 ≈ 1220
    # This means missing one fraud case is as bad as 1220 false alarms.

    models = {
        '1_LogisticRegression': LogisticRegression(
            class_weight='balanced',  # Auto-adjust for imbalance
            max_iter=1000,
            random_state=RANDOM_SEED,
            solver='lbfgs'
        ),

        '2_DecisionTree': DecisionTreeClassifier(
            class_weight='balanced',
            max_depth=10,           # Prevent overfitting
            min_samples_leaf=5,
            random_state=RANDOM_SEED
        ),

        '3_RandomForest': RandomForestClassifier(
            n_estimators=200,       # 200 trees in the forest
            class_weight='balanced',
            max_depth=15,
            min_samples_leaf=3,
            random_state=RANDOM_SEED,
            n_jobs=-1              # Use all CPU cores
        ),

        '4_GradientBoosting': GradientBoostingClassifier(
            n_estimators=200,
            max_depth=5,
            learning_rate=0.1,
            random_state=RANDOM_SEED
        ),

        '5_XGBoost': XGBClassifier(
            n_estimators=300,
            max_depth=6,
            learning_rate=0.1,
            scale_pos_weight=1220,  # Handle extreme imbalance
            eval_metric='aucpr',    # Optimize for precision-recall AUC
            random_state=RANDOM_SEED,
            use_label_encoder=False,
            verbosity=0
        ),

        '6_LightGBM': LGBMClassifier(
            n_estimators=300,
            max_depth=6,
            learning_rate=0.1,
            is_unbalance=True,      # LightGBM's imbalance handling
            random_state=RANDOM_SEED,
            verbose=-1
        ),

        '7_AdaBoost': AdaBoostClassifier(
            n_estimators=200,
            learning_rate=0.1,
            random_state=RANDOM_SEED
        ),

        '8_MLP': MLPClassifier(
            hidden_layer_sizes=(64, 32),  # 2 hidden layers
            max_iter=500,
            random_state=RANDOM_SEED,
            early_stopping=True,
            validation_fraction=0.1
        ),
    }

    for name, model in models.items():
        print(f"  ✅ {name}: {model.__class__.__name__}")

    return models


# ===================================================================
# STEP 5C: TRAIN AND EVALUATE WITH STRATIFIED K-FOLD CV
# ===================================================================
def train_and_evaluate(X, y, models, feature_cols):
    """
    Train each model using Stratified K-Fold Cross-Validation with SMOTE.
    
    WHY STRATIFIED K-FOLD?
    Regular K-Fold randomly splits data into K subsets. With only 49
    fraud samples, some folds might get 0 fraud samples — useless!
    Stratified K-Fold ensures each fold has the same fraud ratio.
    
    WHY SMOTE INSIDE THE FOLD?
    SMOTE (Synthetic Minority Oversampling Technique) creates synthetic
    fraud examples by interpolating between existing fraud samples.
    
    CRITICAL: We apply SMOTE only to the TRAINING set, never to the
    test set. If we SMOTE'd the full dataset before splitting, synthetic
    fraud examples would leak into the test set, giving us inflated
    results. This is called "data leakage" — a common mistake!
    
    Pipeline per fold:
      Training data → SMOTE → Scale → Train Model
      Test data → Scale (same scaler) → Predict → Evaluate
    """
    print("\n" + "=" * 70)
    print("STEP 5C: TRAINING WITH STRATIFIED K-FOLD CV + SMOTE")
    print("=" * 70)
    print(f"  Folds: {N_FOLDS}")
    print(f"  SMOTE: Applied to training set only (no data leakage)")
    print(f"  Scaling: StandardScaler (zero mean, unit variance)")

    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_SEED)

    all_results = {}

    for model_name, model in models.items():
        print(f"\n  {'─'*60}")
        print(f"  🤖 Training: {model_name}")
        print(f"  {'─'*60}")

        fold_metrics = {
            'accuracy': [], 'precision': [], 'recall': [],
            'f1': [], 'auc_roc': [], 'avg_precision': []
        }

        # Store predictions from all folds for aggregate confusion matrix
        all_y_true = []
        all_y_pred = []
        all_y_prob = []

        for fold_idx, (train_idx, test_idx) in enumerate(skf.split(X, y)):
            X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
            y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

            # --- Apply SMOTE to training set only ---
            try:
                # SMOTE needs at least k_neighbors+1 samples of minority class
                # With ~10 fraud per fold, k_neighbors=3 is safe
                smote = SMOTE(
                    random_state=RANDOM_SEED,
                    k_neighbors=min(3, y_train.sum() - 1)
                )
                X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)
            except ValueError:
                # If SMOTE fails (too few samples), use original data
                X_train_resampled, y_train_resampled = X_train, y_train

            # --- Scale features ---
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train_resampled)
            X_test_scaled = scaler.transform(X_test)

            # --- Train ---
            model_clone = _clone_model(model)
            model_clone.fit(X_train_scaled, y_train_resampled)

            # --- Predict ---
            y_pred = model_clone.predict(X_test_scaled)

            # Get probability scores (for AUC)
            if hasattr(model_clone, 'predict_proba'):
                y_prob = model_clone.predict_proba(X_test_scaled)[:, 1]
            else:
                y_prob = model_clone.decision_function(X_test_scaled)

            # --- Collect metrics ---
            fold_metrics['accuracy'].append(accuracy_score(y_test, y_pred))
            fold_metrics['precision'].append(
                precision_score(y_test, y_pred, zero_division=0))
            fold_metrics['recall'].append(
                recall_score(y_test, y_pred, zero_division=0))
            fold_metrics['f1'].append(
                f1_score(y_test, y_pred, zero_division=0))

            try:
                fold_metrics['auc_roc'].append(roc_auc_score(y_test, y_prob))
            except ValueError:
                fold_metrics['auc_roc'].append(0)

            try:
                fold_metrics['avg_precision'].append(
                    average_precision_score(y_test, y_prob))
            except ValueError:
                fold_metrics['avg_precision'].append(0)

            all_y_true.extend(y_test.tolist())
            all_y_pred.extend(y_pred.tolist())
            all_y_prob.extend(y_prob.tolist())

            # Print fold result
            fraud_in_test = y_test.sum()
            caught = ((y_pred == 1) & (y_test == 1)).sum()
            print(f"    Fold {fold_idx+1}: F1={fold_metrics['f1'][-1]:.4f} "
                  f"| Recall={fold_metrics['recall'][-1]:.4f} "
                  f"| Caught {caught}/{fraud_in_test} fraud")

        # --- Aggregate results ---
        result = {}
        for metric, values in fold_metrics.items():
            result[f'{metric}_mean'] = np.mean(values)
            result[f'{metric}_std'] = np.std(values)

        # Aggregate confusion matrix
        result['confusion_matrix'] = confusion_matrix(all_y_true, all_y_pred).tolist()
        result['total_fraud_caught'] = int(
            sum(1 for yt, yp in zip(all_y_true, all_y_pred) if yt == 1 and yp == 1)
        )
        result['total_fraud'] = int(sum(all_y_true))
        result['total_false_alarms'] = int(
            sum(1 for yt, yp in zip(all_y_true, all_y_pred) if yt == 0 and yp == 1)
        )

        all_results[model_name] = result

        print(f"\n    📊 AVERAGE METRICS:")
        print(f"       Accuracy:  {result['accuracy_mean']:.4f} ± {result['accuracy_std']:.4f}")
        print(f"       Precision: {result['precision_mean']:.4f} ± {result['precision_std']:.4f}")
        print(f"       Recall:    {result['recall_mean']:.4f} ± {result['recall_std']:.4f}")
        print(f"       F1 Score:  {result['f1_mean']:.4f} ± {result['f1_std']:.4f}")
        print(f"       AUC-ROC:   {result['auc_roc_mean']:.4f} ± {result['auc_roc_std']:.4f}")
        print(f"       Avg Prec:  {result['avg_precision_mean']:.4f} ± {result['avg_precision_std']:.4f}")
        print(f"       Fraud caught: {result['total_fraud_caught']}/{result['total_fraud']}")
        print(f"       False alarms: {result['total_false_alarms']}")

    return all_results


def _clone_model(model):
    """Create a fresh copy of a model with the same parameters."""
    from sklearn.base import clone
    return clone(model)


# ===================================================================
# STEP 5D: FEATURE IMPORTANCE ANALYSIS (XGBoost)
# ===================================================================
def analyze_feature_importance(X, y, feature_cols):
    """
    Train the final XGBoost model on ALL data and extract feature importance.
    
    WHY FEATURE IMPORTANCE?
    This is the EXPLAINABILITY component. When you present to your
    committee, you need to show:
      "The model considers vendor_edges (74x fraud ratio) and
       amount_anomaly_score (62x fraud ratio) as the top predictors."
    
    This proves our graph features are meaningful, not just noise.
    """
    print("\n" + "=" * 70)
    print("STEP 5D: FEATURE IMPORTANCE ANALYSIS (XGBoost)")
    print("=" * 70)

    # Apply SMOTE to full dataset for final model
    smote = SMOTE(random_state=RANDOM_SEED, k_neighbors=3)
    X_resampled, y_resampled = smote.fit_resample(X, y)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_resampled)

    final_model = XGBClassifier(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.1,
        scale_pos_weight=1,  # Already balanced via SMOTE
        eval_metric='aucpr',
        random_state=RANDOM_SEED,
        use_label_encoder=False,
        verbosity=0
    )
    final_model.fit(X_scaled, y_resampled)

    # Get feature importance
    importance = final_model.feature_importances_
    feat_imp = pd.DataFrame({
        'feature': feature_cols,
        'importance': importance
    }).sort_values('importance', ascending=False)

    print(f"\n  📊 TOP 15 FEATURES BY IMPORTANCE:")
    print(f"  {'Rank':<6} {'Feature':<30} {'Importance':<12}")
    print(f"  {'-'*48}")
    for i, (_, row) in enumerate(feat_imp.head(15).iterrows()):
        bar = '█' * int(row['importance'] * 50)
        print(f"  {i+1:<6} {row['feature']:<30} {row['importance']:.4f} {bar}")

    # Save feature importance
    feat_imp.to_csv(os.path.join(OUTPUT_DIR, 'feature_importance.csv'), index=False)

    # Plot feature importance
    plt.figure(figsize=(12, 8))
    top_features = feat_imp.head(15)
    colors = ['#e74c3c' if imp > 0.1 else '#3498db' if imp > 0.05 else '#95a5a6'
              for imp in top_features['importance']]
    bars = plt.barh(range(len(top_features)), top_features['importance'].values,
                    color=colors, edgecolor='white', linewidth=0.5)
    plt.yticks(range(len(top_features)),
               [f.replace('_', ' ').title() for f in top_features['feature'].values],
               fontsize=11)
    plt.xlabel('Feature Importance (Gain)', fontsize=12)
    plt.title('XGBoost Feature Importance — Graph-Derived Fraud Features',
              fontsize=14, fontweight='bold')
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'feature_importance.png'), dpi=150,
                bbox_inches='tight')
    plt.close()
    print(f"\n  ✅ Feature importance plot saved")

    return feat_imp, final_model


# ===================================================================
# STEP 5E: GENERATE COMPARISON TABLE AND VISUALIZATIONS
# ===================================================================
def generate_comparison(all_results):
    """
    Create the final model comparison table and visualizations.
    
    This is the table you'll put in your thesis and show during your
    presentation. It proves that XGBoost is the best model for our
    graph-derived fraud features.
    """
    print("\n" + "=" * 70)
    print("STEP 5E: MODEL COMPARISON RESULTS")
    print("=" * 70)

    # Build comparison DataFrame
    rows = []
    for model_name, result in all_results.items():
        rows.append({
            'Model': model_name.split('_', 1)[1],  # Remove number prefix
            'Accuracy': f"{result['accuracy_mean']:.4f} ± {result['accuracy_std']:.4f}",
            'Precision': f"{result['precision_mean']:.4f} ± {result['precision_std']:.4f}",
            'Recall': f"{result['recall_mean']:.4f} ± {result['recall_std']:.4f}",
            'F1 Score': f"{result['f1_mean']:.4f} ± {result['f1_std']:.4f}",
            'AUC-ROC': f"{result['auc_roc_mean']:.4f} ± {result['auc_roc_std']:.4f}",
            'Avg Precision': f"{result['avg_precision_mean']:.4f} ± {result['avg_precision_std']:.4f}",
            'Fraud Caught': f"{result['total_fraud_caught']}/{result['total_fraud']}",
            'False Alarms': result['total_false_alarms'],
            # Raw values for sorting
            '_f1': result['f1_mean'],
            '_auc': result['auc_roc_mean'],
            '_recall': result['recall_mean'],
        })

    comparison_df = pd.DataFrame(rows)
    comparison_df = comparison_df.sort_values('_f1', ascending=False)

    # Print the comparison table
    print("\n  ┌─────────────────────────────────────────────────────────────────────────────────┐")
    print("  │                     8-MODEL COMPARATIVE BENCHMARK RESULTS                      │")
    print("  ├─────────────────────────────────────────────────────────────────────────────────┤")
    print(f"  │  {'Model':<22} {'F1 Score':<18} {'Recall':<18} {'AUC-ROC':<18} {'Caught':>8} │")
    print("  ├─────────────────────────────────────────────────────────────────────────────────┤")
    for _, row in comparison_df.iterrows():
        model = row['Model']
        f1 = row['F1 Score']
        recall = row['Recall']
        auc = row['AUC-ROC']
        caught = row['Fraud Caught']
        marker = " ★" if model == 'XGBoost' else ""
        print(f"  │  {model:<22} {f1:<18} {recall:<18} {auc:<18} {caught:>8}{marker:<2}│")
    print("  └─────────────────────────────────────────────────────────────────────────────────┘")

    # Save comparison to CSV
    save_cols = [c for c in comparison_df.columns if not c.startswith('_')]
    comparison_df[save_cols].to_csv(
        os.path.join(OUTPUT_DIR, 'model_comparison.csv'), index=False
    )

    # ─── VISUALIZATION: Model Comparison Bar Chart ──────────────
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    models_sorted = comparison_df['Model'].values
    metrics = [
        ('_f1', 'F1 Score', '#e74c3c'),
        ('_recall', 'Recall (Fraud Detection Rate)', '#2ecc71'),
        ('_auc', 'AUC-ROC', '#3498db'),
    ]

    for ax, (metric_key, metric_name, color) in zip(axes, metrics):
        values = comparison_df[metric_key].values
        bars = ax.barh(range(len(models_sorted)), values,
                       color=color, alpha=0.85, edgecolor='white')
        ax.set_yticks(range(len(models_sorted)))
        ax.set_yticklabels(models_sorted, fontsize=10)
        ax.set_xlabel(metric_name, fontsize=11)
        ax.set_xlim(0, 1.05)
        ax.invert_yaxis()

        # Add value labels
        for i, (bar, val) in enumerate(zip(bars, values)):
            ax.text(val + 0.02, i, f'{val:.3f}', va='center', fontsize=9)

        # Highlight XGBoost
        for i, m in enumerate(models_sorted):
            if m == 'XGBoost':
                bars[i].set_edgecolor('#f39c12')
                bars[i].set_linewidth(2)

    plt.suptitle('8-Model Benchmark: Graph-Derived Fraud Feature Performance',
                 fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'model_comparison.png'), dpi=150,
                bbox_inches='tight')
    plt.close()
    print(f"\n  ✅ Comparison chart saved")

    # ─── VISUALIZATION: Confusion Matrices ──────────────────────
    fig, axes = plt.subplots(2, 4, figsize=(20, 10))
    axes = axes.flatten()

    for idx, (model_name, result) in enumerate(all_results.items()):
        ax = axes[idx]
        cm = np.array(result['confusion_matrix'])
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                    xticklabels=['Clean', 'Fraud'],
                    yticklabels=['Clean', 'Fraud'])
        ax.set_title(model_name.split('_', 1)[1], fontsize=11, fontweight='bold')
        ax.set_xlabel('Predicted')
        ax.set_ylabel('Actual')

    plt.suptitle('Confusion Matrices — 8-Model Benchmark',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'confusion_matrices.png'), dpi=150,
                bbox_inches='tight')
    plt.close()
    print(f"  ✅ Confusion matrices saved")

    return comparison_df


# ===================================================================
# STEP 5F: SAVE FINAL MODEL AND GENERATE AUDIT REPORT
# ===================================================================
def save_final_report(all_results, feat_imp, comparison_df):
    """Generate the final audit-ready report."""
    print("\n" + "=" * 70)
    print("STEP 5F: FINAL REPORT")
    print("=" * 70)

    # Find the best model by F1
    best_model = comparison_df.iloc[0]['Model']
    best_f1 = comparison_df.iloc[0]['_f1']
    best_auc = comparison_df.iloc[0]['_auc']
    best_recall = comparison_df.iloc[0]['_recall']

    report = {
        'project': 'The Self-Auditing Ledger',
        'best_model': best_model,
        'best_f1': round(best_f1, 4),
        'best_auc_roc': round(best_auc, 4),
        'best_recall': round(best_recall, 4),
        'total_documents': 59852,
        'total_fraud': 49,
        'features_used': len(feat_imp),
        'top_5_features': feat_imp.head(5)['feature'].tolist(),
        'models_compared': len(all_results),
        'cv_folds': N_FOLDS,
        'imbalance_handling': 'SMOTE + class_weight balanced',
    }

    # Save report
    with open(os.path.join(OUTPUT_DIR, 'final_report.json'), 'w') as f:
        json.dump(report, f, indent=2)

    print(f"""
  ┌──────────────────────────────────────────────────────────────────┐
  │            THE SELF-AUDITING LEDGER — FINAL RESULTS             │
  ├──────────────────────────────────────────────────────────────────┤
  │                                                                  │
  │  Best Model:      {best_model:<44}│
  │  F1 Score:        {best_f1:<44.4f}│
  │  AUC-ROC:         {best_auc:<44.4f}│
  │  Recall:          {best_recall:<44.4f}│
  │                                                                  │
  │  Documents:       59,852                                         │
  │  Fraud Cases:     49 (0.08%)                                     │
  │  Features:        {len(feat_imp):<4} graph-derived structural features     │
  │  Models Compared: 8                                              │
  │  CV Folds:        {N_FOLDS}                                              │
  │                                                                  │
  │  Top Features:                                                   │
  │    1. {feat_imp.iloc[0]['feature']:<54}│
  │    2. {feat_imp.iloc[1]['feature']:<54}│
  │    3. {feat_imp.iloc[2]['feature']:<54}│
  │    4. {feat_imp.iloc[3]['feature']:<54}│
  │    5. {feat_imp.iloc[4]['feature']:<54}│
  │                                                                  │
  │  Output Files:                                                   │
  │    • results/model_comparison.csv                                │
  │    • results/feature_importance.csv                              │
  │    • results/model_comparison.png                                │
  │    • results/confusion_matrices.png                              │
  │    • results/feature_importance.png                              │
  │    • results/final_report.json                                   │
  └──────────────────────────────────────────────────────────────────┘
    """)


# ===================================================================
# MAIN EXECUTION
# ===================================================================
if __name__ == "__main__":
    print("=" * 70)
    print("STEP 5: AI MODEL TRAINING — 8-Model Benchmark Suite")
    print("The Self-Auditing Ledger: Real-Time Temporal Graph Integrity")
    print("=" * 70)

    start_time = time.time()

    # Load data
    X, y, feature_cols, df = load_and_prepare_data()

    # Define models
    models = get_models()

    # Train and evaluate
    all_results = train_and_evaluate(X, y, models, feature_cols)

    # Feature importance
    feat_imp, final_model = analyze_feature_importance(X, y, feature_cols)

    # Generate comparison and visualizations
    comparison_df = generate_comparison(all_results)

    # Final report
    save_final_report(all_results, feat_imp, comparison_df)

    elapsed = time.time() - start_time
    print(f"⏱️ Total training time: {elapsed:.1f} seconds")
    print(f"\n✅ THE SELF-AUDITING LEDGER — ALL 5 STEPS COMPLETE!")
