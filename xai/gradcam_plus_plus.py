"""
Explainable AI (XAI) Module - Grad-CAM++ Implementation
========================================================
Domain: Healthcare / Computer Vision / Deep Learning
Dataset: APTOS 2019 Blindness Detection
Task: Grad-CAM++ (Generalized Gradient-weighted Class Activation Mapping)
      - Computes higher-order partial derivatives (2nd and 3rd gradients)
      - Superior localization for multiple scattered retinal micro-lesions
      - Generates high-resolution thermal heatmaps & clinical overlays

Author: Senior AI Engineer & XAI Research Specialist
"""

import os
import cv2
import numpy as np
import tensorflow as tf
import keras
import matplotlib.pyplot as plt
from typing import Tuple, Dict, Union, Optional


class GradCAMPlusPlus:
    """
    Computes Grad-CAM++ (Gradient-weighted Class Activation Mapping++) for a given
    trained Keras model and target convolutional layer.
    """

    def __init__(self, model: keras.Model, layer_name: Optional[str] = None):
        """
        Args:
            model: Trained Keras classification model.
            layer_name: Name of the target convolutional layer. If None, automatically
                        finds the last 4D output convolutional layer in the backbone.
        """
        self.model = model
        self.layer_name = layer_name or self._find_target_layer()
        print(f"[+] Grad-CAM++ initialized with target layer: '{self.layer_name}'")

        # Sub-model returning feature map output & model prediction
        try:
            target_layer = self.model.get_layer(self.layer_name)
            self.grad_model = tf.keras.Model(
                inputs=self.model.inputs,
                outputs=[target_layer.output, self.model.output]
            )
        except (ValueError, KeyError):
            submodel = None
            for layer in self.model.layers:
                if hasattr(layer, 'get_layer'):
                    try:
                        target_layer = layer.get_layer(self.layer_name)
                        submodel = layer
                        break
                    except ValueError:
                        continue
            if submodel is None:
                raise ValueError(f"Target layer '{self.layer_name}' not found.")
            feature_extractor = tf.keras.Model(inputs=submodel.inputs, outputs=target_layer.output)
            inputs = self.model.inputs
            x = inputs[0] if isinstance(inputs, list) else inputs
            conv_out = feature_extractor(x)
            preds = self.model(inputs)
            self.grad_model = tf.keras.Model(inputs=inputs, outputs=[conv_out, preds])

    def _find_target_layer(self) -> str:
        for layer in reversed(self.model.layers):
            if hasattr(layer, 'layers'):
                for sub_layer in reversed(layer.layers):
                    if hasattr(sub_layer, 'output_shape') and isinstance(sub_layer.output_shape, tuple):
                        if len(sub_layer.output_shape) == 4:
                            return sub_layer.name
        for layer in reversed(self.model.layers):
            if hasattr(layer, 'output_shape') and isinstance(layer.output_shape, tuple):
                if len(layer.output_shape) == 4:
                    return layer.name
        return "top_activation"

    def compute_heatmap(
        self,
        img_array: np.ndarray,
        class_index: Optional[int] = None,
        eps: float = 1e-8
    ) -> Tuple[np.ndarray, int, float]:
        """
        Computes Grad-CAM++ heatmap utilizing second and third higher-order gradients.
        """
        if img_array.ndim == 3:
            img_array = np.expand_dims(img_array, axis=0)

        # Extract submodel backbone feature maps and route through head
        submodel = self.model.get_layer("efficientnetb0") if "efficientnetb0" in [l.name for l in self.model.layers] else self.model.layers[1]
        target_layer = submodel.get_layer(self.layer_name)
        extractor = tf.keras.Model(inputs=submodel.inputs, outputs=target_layer.output)

        with tf.GradientTape() as tape1:
            with tf.GradientTape() as tape2:
                with tf.GradientTape() as tape3:
                    conv_outputs = extractor(img_array)
                    tape3.watch(conv_outputs)
                    tape2.watch(conv_outputs)
                    tape1.watch(conv_outputs)

                    x = self.model.get_layer("global_average_pooling")(conv_outputs)
                    x = self.model.get_layer("batch_normalization")(x)
                    x = self.model.get_layer("dropout_regularization")(x)
                    predictions = self.model.get_layer("softmax_predictions")(x)

                    if class_index is None:
                        class_index = int(tf.argmax(predictions[0]))

                    class_score = predictions[0, class_index]

                grads_1st = tape3.gradient(class_score, conv_outputs)
                if grads_1st is None:
                    grads_1st = tf.ones_like(conv_outputs)
            grads_2nd = tape2.gradient(grads_1st, conv_outputs)
            if grads_2nd is None:
                grads_2nd = tf.ones_like(conv_outputs)
        grads_3rd = tape1.gradient(grads_2nd, conv_outputs)
        if grads_3rd is None:
            grads_3rd = tf.ones_like(conv_outputs)

        # Grad-CAM++ Alpha Weights Calculation:
        conv_out = conv_outputs[0]  # (H, W, Channels)
        g1 = grads_1st[0]
        g2 = grads_2nd[0]
        g3 = grads_3rd[0]

        # Elementwise denominator
        denom = 2.0 * g2 + conv_out * g3
        denom = tf.where(denom != 0, denom, tf.ones_like(denom) * eps)

        alpha = g2 / denom

        # Apply ReLU to 1st gradients
        positive_g1 = tf.maximum(g1, 0.0)
        alpha_positive = alpha * positive_g1

        # Channel importance weights w_k^c via spatial sum
        weights = tf.reduce_sum(alpha_positive, axis=(0, 1))

        # Weighted combination of feature maps
        heatmap = conv_out @ weights[..., tf.newaxis]
        heatmap = tf.squeeze(heatmap)

        # Apply ReLU to final combined heatmap
        heatmap = tf.maximum(heatmap, 0.0).numpy()

        # Normalize between [0, 1]
        max_val = np.max(heatmap)
        if max_val != 0:
            heatmap = heatmap / (max_val + eps)

        confidence = float(predictions[0, class_index].numpy())
        return heatmap, class_index, confidence

    @staticmethod
    def overlay_heatmap(
        heatmap: np.ndarray,
        original_img: np.ndarray,
        alpha: float = 0.4,
        colormap: int = cv2.COLORMAP_JET
    ) -> Tuple[np.ndarray, np.ndarray]:
        h, w = original_img.shape[:2]
        resized_heatmap = cv2.resize(heatmap, (w, h))
        heatmap_uint8 = np.uint8(255 * resized_heatmap)

        colored_heatmap_bgr = cv2.applyColorMap(heatmap_uint8, colormap)
        colored_heatmap_rgb = cv2.cvtColor(colored_heatmap_bgr, cv2.COLOR_BGR2RGB)

        superimposed = cv2.addWeighted(
            original_img.astype(np.uint8), 1 - alpha,
            colored_heatmap_rgb, alpha, 0
        )
        return colored_heatmap_rgb, superimposed
