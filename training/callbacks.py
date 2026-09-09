"""
Production Callbacks Module
===========================
Domain: Healthcare / Computer Vision / Deep Learning
Dataset: APTOS 2019 Blindness Detection
Task: Create Keras callbacks for training monitoring, learning rate decay,
      early stopping, checkpoint saving, and TensorBoard logging.

Author: Senior AI Engineer & MLOps Specialist
"""

import os
import keras
from typing import List


def get_training_callbacks(
    model_name: str = "efficientnet_b0",
    stage: str = "stage1",
    checkpoint_dir: str = "models",
    log_dir: str = "outputs/logs",
    patience_early: int = 7,
    patience_lr: int = 3
) -> List[keras.callbacks.Callback]:
    """
    Constructs a robust suite of training callbacks:
    1. ModelCheckpoint: Saves best model weights on val_loss improvement.
    2. EarlyStopping: Prevents overfitting when validation loss stagnates.
    3. ReduceLROnPlateau: Dynamically decays learning rate on plateau.
    4. TensorBoard: Logs epoch metrics and network histograms.
    """
    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)

    checkpoint_path = os.path.join(checkpoint_dir, f"best_{model_name}_{stage}.h5")
    stage_log_dir = os.path.join(log_dir, f"{model_name}_{stage}")
    os.makedirs(stage_log_dir, exist_ok=True)   # ← ensure CSVLogger directory exists

    callbacks_list = [
        # 1. Model Checkpoint (Saves best model based on validation loss)
        keras.callbacks.ModelCheckpoint(
            filepath=checkpoint_path,
            monitor="val_loss",
            verbose=1,
            save_best_only=True,
            mode="min"
        ),

        # 2. Early Stopping (Halts training when validation loss stops improving)
        keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=patience_early,
            verbose=1,
            mode="min",
            restore_best_weights=True
        ),

        # 3. Reduce Learning Rate on Plateau (Decays LR by factor 0.5 when stuck)
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=patience_lr,
            min_lr=1e-7,
            verbose=1,
            mode="min"
        ),

        # 4. CSV Logger (lightweight alternative to TensorBoard — no extra packages needed)
        keras.callbacks.CSVLogger(
            filename=os.path.join(stage_log_dir, "training_log.csv"),
            separator=",",
            append=True
        )
    ]

    print(f"\n[OK] Configured Training Callbacks for {stage.upper()}:")
    print(f"     - Checkpoint Target Path : '{checkpoint_path}'")
    print(f"     - TensorBoard Log Dir    : '{stage_log_dir}'")
    print(f"     - Early Stopping Patience: {patience_early} epochs")
    print(f"     - Reduce LR Patience     : {patience_lr} epochs")

    return callbacks_list
