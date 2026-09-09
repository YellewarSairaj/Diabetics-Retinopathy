"""
Transfer Learning Model Factory Module
======================================
Domain: Healthcare / Computer Vision / Deep Learning
Dataset: APTOS 2019 Blindness Detection
Task: Flexible Convolutional Neural Network (CNN) architecture factory supporting:
      - EfficientNet-B0 (Default Primary Model)
      - EfficientNet-B3
      - ResNet50
      - MobileNetV2
      
Features:
      - Pre-trained ImageNet backbones
      - Global Average Pooling (GAP)
      - Dropout Regularization
      - Dense Output Layer with Softmax Activation
      - Dynamic backbone freezing / fine-tuning toggle

Author: Senior AI Engineer & Neural Architecture Specialist
"""

import keras
from keras import layers, models
from typing import Tuple, Optional


class ModelFactory:
    """
    Factory class for building and instantiating deep transfer learning architectures.
    Allows seamless switching between EfficientNet-B0, EfficientNet-B3, ResNet50, and MobileNetV2.
    """

    SUPPORTED_ARCHITECTURES = [
        "efficientnet_b0",
        "efficientnet_b3",
        "resnet50",
        "mobilenet_v2"
    ]

    @staticmethod
    def get_base_backbone(
        architecture: str,
        input_shape: Tuple[int, int, int] = (224, 224, 3)
    ) -> keras.Model:
        """Instantiates pre-trained ImageNet feature extraction backbone without top classification head."""
        arch_clean = architecture.lower().strip()

        if arch_clean == "efficientnet_b0":
            backbone = keras.applications.EfficientNetB0(
                include_top=False,
                weights="imagenet",
                input_shape=input_shape
            )
        elif arch_clean == "efficientnet_b3":
            backbone = keras.applications.EfficientNetB3(
                include_top=False,
                weights="imagenet",
                input_shape=input_shape
            )
        elif arch_clean == "resnet50":
            backbone = keras.applications.ResNet50(
                include_top=False,
                weights="imagenet",
                input_shape=input_shape
            )
        elif arch_clean == "mobilenet_v2":
            backbone = keras.applications.MobileNetV2(
                include_top=False,
                weights="imagenet",
                input_shape=input_shape
            )
        else:
            raise ValueError(
                f"Unsupported architecture '{architecture}'. "
                f"Choose from: {ModelFactory.SUPPORTED_ARCHITECTURES}"
            )

        return backbone

    @staticmethod
    def build_model(
        architecture: str = "efficientnet_b0",
        input_shape: Tuple[int, int, int] = (224, 224, 3),
        num_classes: int = 5,
        dropout_rate: float = 0.4,
        freeze_backbone: bool = True,
        learning_rate: float = 0.001
    ) -> keras.Model:
        """
        Builds complete Transfer Learning model architecture:
        Input -> Pretrained Backbone -> Global Average Pooling -> Dropout -> Dense(5, Softmax)
        """
        # Step 1: Instantiate pre-trained backbone
        base_model = ModelFactory.get_base_backbone(architecture, input_shape)

        # Freeze backbone weights for Stage 1 Transfer Learning if specified
        base_model.trainable = not freeze_backbone

        # Step 2: Build Functional Keras Model Pipeline
        inputs = keras.Input(shape=input_shape, name="input_image")

        # Forward pass through pre-trained backbone
        features = base_model(inputs, training=not freeze_backbone)

        # Global Average Pooling to reduce 3D feature map to 1D vector
        x = layers.GlobalAveragePooling2D(name="global_average_pooling")(features)

        # Batch Normalization for feature stabilization
        x = layers.BatchNormalization(name="batch_normalization")(x)

        # Dropout Regularization to prevent co-adaptation/overfitting
        if dropout_rate > 0.0:
            x = layers.Dropout(dropout_rate, name="dropout_regularization")(x)

        # Dense Classification Output Layer with Softmax Activation
        outputs = layers.Dense(num_classes, activation="softmax", name="softmax_predictions")(x)

        model = keras.Model(inputs=inputs, outputs=outputs, name=f"DR_Detection_{architecture}")

        # Step 3: Compile model with Adam optimizer & Categorical Cross-Entropy loss
        optimizer = keras.optimizers.Adam(learning_rate=learning_rate)
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
        print(f"\n[OK] Successfully Built & Compiled Transfer Learning Model:")
        print(f"     - Architecture Base : {architecture.upper()}")
        print(f"     - Input Resolution  : {input_shape}")
        print(f"     - Backbone Frozen   : {freeze_backbone}")
        print(f"     - Initial Learn Rate: {learning_rate}")
        print(f"     - Total Parameters  : {model.count_params():,}")
        print(f"     - Trainable Params  : {trainable_params:,}")

        return model


if __name__ == "__main__":
    print("[+] Testing Transfer Learning Architecture Factory...")

    for arch in ModelFactory.SUPPORTED_ARCHITECTURES:
        print(f"\n--- Testing {arch} ---")
        m = ModelFactory.build_model(architecture=arch, freeze_backbone=True)
        assert m.input_shape == (None, 224, 224, 3)
        assert m.output_shape == (None, 5)

    print("\n[OK] Model Factory Verification Passed for all 4 Architectures!")
