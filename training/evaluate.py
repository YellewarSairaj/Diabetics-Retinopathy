"""
Model Evaluation & Performance Metrics Report Module
====================================================
Domain: Healthcare / Computer Vision / Deep Learning
Dataset: APTOS 2019 Blindness Detection
Task: Comprehensive multi-class model evaluation:
      - Accuracy, Precision, Recall, F1-Score
      - Confusion Matrix (normalized & absolute)
      - Per-Class Classification Report
      - ROC-AUC Curves (One-vs-Rest)
      - Save all artifacts to reports/ directory

Author: Senior AI Engineer & ML Research Scientist
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import keras

from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    roc_curve,
    auc,
    precision_score,
    recall_score,
    f1_score,
    accuracy_score
)

from dataset import DatasetManager
from augmentation import FundusDataSequence

# APTOS 2019 class names
CLASS_NAMES = ["No DR", "Mild", "Moderate", "Severe", "Proliferative DR"]
NUM_CLASSES = 5


class DRModelEvaluator:
    """
    Loads a trained Keras model and performs exhaustive evaluation on the
    held-out test set, generating all ML metrics and visual artifacts.
    """

    def __init__(
        self,
        model_path: str,
        test_df: pd.DataFrame,
        batch_size: int = 16,
        target_size: tuple = (224, 224),
        reports_dir: str = "reports"
    ):
        self.model_path = model_path
        self.test_df = test_df.reset_index(drop=True)
        self.batch_size = batch_size
        self.target_size = target_size
        self.reports_dir = reports_dir
        os.makedirs(self.reports_dir, exist_ok=True)

        print(f"\n[+] Loading trained model from: '{model_path}'...")
        self.model = keras.models.load_model(model_path)
        print(f"[OK] Model loaded successfully. Input shape: {self.model.input_shape}")

        # Build test generator (no augmentation, no shuffle)
        self.test_gen = FundusDataSequence(
            dataframe=self.test_df,
            batch_size=self.batch_size,
            target_size=self.target_size,
            is_training=False,
            shuffle=False,
            augment=False
        )

    def get_predictions(self):
        """
        Runs inference on the full test set.
        Returns:
          y_true  : integer ground truth labels (N,)
          y_pred  : integer predicted labels    (N,)
          y_proba : softmax probability matrix  (N, 5)
        """
        print("\n[+] Running inference on test set...")
        y_proba = self.model.predict(self.test_gen, verbose=1)
        y_pred  = np.argmax(y_proba, axis=1)

        # Reconstruct ground truth in the same order as test_gen (no shuffle)
        y_true = self.test_df['diagnosis'].values[:len(y_pred)]

        print(f"[OK] Inference complete. Evaluated {len(y_true)} test samples.")
        return y_true, y_pred, y_proba

    def compute_metrics(self, y_true: np.ndarray, y_pred: np.ndarray, y_proba: np.ndarray) -> dict:
        """
        Computes and prints the full evaluation metrics report.
        Returns a metrics dictionary.
        """
        print("\n" + "="*60)
        print("         MODEL EVALUATION METRICS REPORT           ")
        print("="*60)

        accuracy  = accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred, average='weighted', zero_division=0)
        recall    = recall_score(y_true, y_pred, average='weighted', zero_division=0)
        f1        = f1_score(y_true, y_pred, average='weighted', zero_division=0)

        # One-hot encode for AUC computation
        y_true_onehot = np.eye(NUM_CLASSES)[y_true]
        try:
            roc_auc = roc_auc_score(y_true_onehot, y_proba, multi_class='ovr', average='macro')
        except ValueError:
            roc_auc = float('nan')

        print(f"\n  Overall Test Metrics:")
        print(f"  {'Accuracy':<30}: {accuracy*100:.2f}%")
        print(f"  {'Precision (Weighted)':<30}: {precision:.4f}")
        print(f"  {'Recall (Weighted)':<30}: {recall:.4f}")
        print(f"  {'F1 Score (Weighted)':<30}: {f1:.4f}")
        print(f"  {'ROC-AUC Score (Macro OvR)':<30}: {roc_auc:.4f}")

        print(f"\n  Per-Class Classification Report:")
        print("-" * 60)
        report = classification_report(
            y_true, y_pred,
            target_names=CLASS_NAMES,
            digits=4,
            zero_division=0
        )
        print(report)

        metrics = {
            "accuracy":   round(accuracy, 4),
            "precision":  round(precision, 4),
            "recall":     round(recall, 4),
            "f1_score":   round(f1, 4),
            "roc_auc":    round(roc_auc, 4)
        }

        # Save metrics to CSV
        metrics_path = os.path.join(self.reports_dir, "evaluation_metrics.csv")
        pd.DataFrame([metrics]).to_csv(metrics_path, index=False)
        print(f"\n[OK] Metrics saved to: '{metrics_path}'")

        return metrics

    def plot_confusion_matrix(self, y_true: np.ndarray, y_pred: np.ndarray) -> None:
        """Plots and saves both absolute-count and normalized confusion matrices."""
        cm  = confusion_matrix(y_true, y_pred)
        cm_norm = cm.astype('float') / cm.sum(axis=1, keepdims=True)

        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        fig.suptitle('Confusion Matrix – APTOS 2019 DR Detection', fontsize=14, fontweight='bold')

        for ax, data, title, fmt in zip(
            axes,
            [cm, cm_norm],
            ['Absolute Counts', 'Normalized (Row %)'],
            ['d', '.2f']
        ):
            sns.heatmap(
                data, annot=True, fmt=fmt, cmap='Blues', ax=ax,
                xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES,
                linewidths=0.5, linecolor='gray'
            )
            ax.set_title(title, fontsize=12)
            ax.set_xlabel('Predicted Label', fontsize=11)
            ax.set_ylabel('True Label', fontsize=11)
            plt.setp(ax.get_xticklabels(), rotation=30, ha='right')

        plt.tight_layout()
        save_path = os.path.join(self.reports_dir, "confusion_matrix.png")
        plt.savefig(save_path, dpi=300)
        plt.close()
        print(f"[OK] Confusion matrix saved to: '{save_path}'")

    def plot_roc_curves(self, y_true: np.ndarray, y_proba: np.ndarray) -> None:
        """Plots per-class ROC curves (One-vs-Rest) and the macro-average."""
        y_true_onehot = np.eye(NUM_CLASSES)[y_true]

        plt.figure(figsize=(10, 7))
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']

        for i, (name, color) in enumerate(zip(CLASS_NAMES, colors)):
            fpr, tpr, _ = roc_curve(y_true_onehot[:, i], y_proba[:, i])
            roc_auc_val = auc(fpr, tpr)
            plt.plot(fpr, tpr, color=color, linewidth=2,
                     label=f'{name} (AUC = {roc_auc_val:.3f})')

        plt.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Random Classifier')
        plt.xlabel('False Positive Rate', fontsize=13)
        plt.ylabel('True Positive Rate (Recall)', fontsize=13)
        plt.title('ROC Curves – One-vs-Rest (APTOS 2019 DR Detection)', fontsize=14)
        plt.legend(loc='lower right', fontsize=11)
        plt.grid(alpha=0.3)
        plt.tight_layout()

        save_path = os.path.join(self.reports_dir, "roc_curves.png")
        plt.savefig(save_path, dpi=300)
        plt.close()
        print(f"[OK] ROC curves saved to: '{save_path}'")

    def run_full_evaluation(self) -> dict:
        """Orchestrates the full evaluation pipeline and generates all report artifacts."""
        y_true, y_pred, y_proba = self.get_predictions()
        metrics = self.compute_metrics(y_true, y_pred, y_proba)
        self.plot_confusion_matrix(y_true, y_pred)
        self.plot_roc_curves(y_true, y_proba)
        print(f"\n[OK] Full evaluation complete. All reports saved in '{self.reports_dir}/'")
        return metrics


if __name__ == "__main__":
    print("[+] Initializing Model Evaluation Pipeline...")

    # Load dataset
    manager  = DatasetManager(dataset_dir="aptos2019-blindness-detection")
    df       = manager.load_metadata()
    _, _, test_df = manager.get_stratified_splits(df)

    # Path to trained model (use the final fine-tuned model)
    model_path = os.path.join("models", "final_efficientnet_b0_dr_model.keras")

    if not os.path.exists(model_path):
        print(f"[WARN] Trained model not found at '{model_path}'.")
        print("       Please run training/train.py first to generate a trained model.")
        sys.exit(0)

    evaluator = DRModelEvaluator(
        model_path=model_path,
        test_df=test_df,
        batch_size=16,
        target_size=(224, 224)
    )

    metrics = evaluator.run_full_evaluation()
