"""
Two-Stage Fine-Tuning & Model Training Module
=============================================
Domain: Healthcare / Computer Vision / Deep Learning
Dataset: APTOS 2019 Blindness Detection
Task: Execute end-to-end two-stage transfer learning fine-tuning workflow:
      Stage 1: Train Top Classifier Head (Frozen Backbone, LR = 0.001)
      Stage 2: End-to-End Fine-Tuning (Unfrozen Backbone, LR = 0.00001)

Features:
      - Class Weighting integration
      - Callbacks (ModelCheckpoint, EarlyStopping, ReduceLROnPlateau, TensorBoard)
      - Training Loss & Accuracy Plotting

Author: Senior AI Engineer & MLOps Architect
"""

import os
import sys

# Ensure UTF-8 output encoding for Windows terminals
if hasattr(sys.stdout, 'reconfigure'):
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import keras

from dataset import DatasetManager
from augmentation import FundusDataSequence, compute_class_weights
from model import ModelFactory
from callbacks import get_training_callbacks


class DRTrainer:
    """
    Orchestrates the two-stage training paradigm for Diabetic Retinopathy detection.
    """

    def __init__(
        self,
        architecture: str = "efficientnet_b0",
        batch_size: int = 16,
        target_size: tuple = (224, 224),
        dataset_dir: str = "aptos2019-blindness-detection"
    ):
        self.architecture = architecture
        self.batch_size = batch_size
        self.target_size = target_size
        self.dataset_dir = dataset_dir

        # Initialize dataset manager & stratified splits
        self.manager = DatasetManager(dataset_dir=self.dataset_dir)
        self.metadata_df = self.manager.load_metadata()
        self.train_df, self.val_df, self.test_df = self.manager.get_stratified_splits(self.metadata_df)

        # Compute class weights
        self.class_weights = compute_class_weights(self.train_df)

        # Build Keras Sequence Generators
        self.train_gen = FundusDataSequence(
            dataframe=self.train_df,
            batch_size=self.batch_size,
            target_size=self.target_size,
            is_training=True,
            augment=True
        )
        self.val_gen = FundusDataSequence(
            dataframe=self.val_df,
            batch_size=self.batch_size,
            target_size=self.target_size,
            is_training=False,
            augment=False
        )

    def train_stage1(self, epochs: int = 10, lr: float = 0.001) -> keras.Model:
        """
        Stage 1: Feature Extraction Warmup.
        Trains only the top classification head with frozen backbone.
        """
        print(f"\n" + "="*60)
        print(f"   STAGE 1: FEATURE EXTRACTION ({self.architecture.upper()})   ")
        print("="*60)

        # Build model with frozen backbone
        model = ModelFactory.build_model(
            architecture=self.architecture,
            input_shape=(*self.target_size, 3),
            num_classes=5,
            dropout_rate=0.4,
            freeze_backbone=True,
            learning_rate=lr
        )

        callbacks = get_training_callbacks(
            model_name=self.architecture,
            stage="stage1",
            patience_early=5,
            patience_lr=2
        )

        print(f"\n[+] Starting Stage 1 Training for {epochs} epochs...")
        history_stage1 = model.fit(
            self.train_gen,
            validation_data=self.val_gen,
            epochs=epochs,
            class_weight=self.class_weights,
            callbacks=callbacks,
            verbose=1
        )

        self.plot_history(history_stage1, stage_name="Stage 1 (Frozen Backbone)")
        return model

    def train_stage2(
        self,
        model: keras.Model,
        epochs: int = 15,
        fine_tune_lr: float = 0.00001
    ) -> keras.Model:
        """
        Stage 2: End-to-End Fine-Tuning.
        Unfreezes backbone and re-compiles with a low learning rate (1e-5).
        """
        print(f"\n" + "="*60)
        print(f"   STAGE 2: END-TO-END FINE-TUNING ({self.architecture.upper()})   ")
        print("="*60)

        # Unfreeze all layers in the backbone
        model.trainable = True

        # Re-compile model with small learning rate for fine-tuning
        optimizer = keras.optimizers.Adam(learning_rate=fine_tune_lr)
        model.compile(
            optimizer=optimizer,
            loss="categorical_crossentropy",
            metrics=[
                "accuracy",
                keras.metrics.CategoricalAccuracy(name="categorical_accuracy"),
                keras.metrics.Precision(name="precision"),
                keras.metrics.Recall(name="recall"),
                keras.metrics.AUC(name="auc")
            ]
        )

        trainable_params = sum(w.shape.num_elements() for w in model.trainable_weights)
        print(f"[OK] Unfrozen Backbone for Stage 2 Fine-Tuning:")
        print(f"     - Fine-Tuning LR : {fine_tune_lr}")
        print(f"     - Trainable Params: {trainable_params:,}")

        callbacks = get_training_callbacks(
            model_name=self.architecture,
            stage="stage2",
            patience_early=7,
            patience_lr=3
        )

        print(f"\n[+] Starting Stage 2 Fine-Tuning for {epochs} epochs...")
        history_stage2 = model.fit(
            self.train_gen,
            validation_data=self.val_gen,
            epochs=epochs,
            class_weight=self.class_weights,
            callbacks=callbacks,
            verbose=1
        )

        self.plot_history(history_stage2, stage_name="Stage 2 (Fine-Tuned Backbone)")
        
        # Save final complete production model
        final_model_path = os.path.join("models", f"final_{self.architecture}_dr_model.keras")
        model.save(final_model_path)
        print(f"\n[OK] Final Trained Model Successfully Saved to: '{final_model_path}'")

        return model

    def plot_history(self, history, stage_name: str = "Training") -> None:
        """Plots and saves loss and accuracy curves for training and validation sets."""
        os.makedirs("outputs", exist_ok=True)
        acc = history.history.get('accuracy', [])
        val_acc = history.history.get('val_accuracy', [])
        loss = history.history.get('loss', [])
        val_loss = history.history.get('val_loss', [])

        epochs_range = range(1, len(acc) + 1)

        plt.figure(figsize=(14, 5))

        # Loss Plot
        plt.subplot(1, 2, 1)
        plt.plot(epochs_range, loss, 'b-o', label='Training Loss')
        plt.plot(epochs_range, val_loss, 'r-s', label='Validation Loss')
        plt.title(f'{stage_name} - Cross-Entropy Loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True)

        # Accuracy Plot
        plt.subplot(1, 2, 2)
        plt.plot(epochs_range, acc, 'b-o', label='Training Accuracy')
        plt.plot(epochs_range, val_acc, 'r-s', label='Validation Accuracy')
        plt.title(f'{stage_name} - Classification Accuracy')
        plt.xlabel('Epochs')
        plt.ylabel('Accuracy')
        plt.legend()
        plt.grid(True)

        save_path = os.path.join("outputs", f"{self.architecture}_{stage_name.replace(' ', '_')}_history.png")
        plt.tight_layout()
        plt.savefig(save_path, dpi=300)
        plt.close()
        print(f"[OK] Saved Training Curves Plot to: '{save_path}'")


if __name__ == "__main__":
    print("[+] Starting Two-Stage Fine-Tuning Pipeline Verification...")

    # Fast verification run (1 epoch per stage)
    trainer = DRTrainer(architecture="efficientnet_b0", batch_size=16)
    
    print("\n--- Running Stage 1 Warmup (1 epoch dry run) ---")
    stage1_model = trainer.train_stage1(epochs=1, lr=0.001)

    print("\n--- Running Stage 2 Fine-Tuning (1 epoch dry run) ---")
    final_model = trainer.train_stage2(stage1_model, epochs=1, fine_tune_lr=0.00001)

    print("\n[OK] Two-Stage Training Pipeline Verified Successfully!")
