# ─────────────────────────────  Imports  ─────────────────────────────
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import tensorflow as tf
from tensorflow.keras import layers, models

from vision_proyecto.config import CLASES_GESTOS, TAM_IMAGEN_CNN


# ────────────────────────────  Dispositivo  ───────────────────────────
def configurar_dispositivo() -> str:
    gpus = tf.config.list_physical_devices("GPU")

    if gpus:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        tf.keras.mixed_precision.set_global_policy("mixed_float16")
        nombres = ", ".join(gpu.name for gpu in gpus)

        if "directml" in str(gpus[0]).lower():
            print(f"[DirectML] GPU disponible con precisión mixta: {nombres}")
            return "DirectML"
        else:
            print(f"[GPU/CUDA] Entrenando con precisión mixta: {nombres}")
            return "GPU"

    print("[CPU] No se detectó GPU (DirectML o CUDA). Entrenando en CPU.")
    print("[Windows nativo] Para usar RTX 3080 con TF>=2.11:")
    print("  1. DirectML (recomendado): pip install tensorflow-directml")
    print("  2. WSL2: pip install 'tensorflow[and-cuda]'")
    return "CPU"


# ─────────────────────────────  Arquitectura  ─────────────────────────
def _bloque_conv(x, filtros: int):
    x = layers.Conv2D(filtros, 3, padding="same", use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    x = layers.Conv2D(filtros, 3, padding="same", use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    return layers.MaxPooling2D()(x)


def construir_cnn() -> tf.keras.Model:
    entradas = layers.Input(shape=(TAM_IMAGEN_CNN, TAM_IMAGEN_CNN, 3))
    x = entradas
    for filtros in (32, 64, 128, 256):
        x = _bloque_conv(x, filtros)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.35)(x)
    x = layers.Dense(128, activation="relu")(x)
    x = layers.Dropout(0.25)(x)
    salidas = layers.Dense(len(CLASES_GESTOS), activation="softmax",
                           dtype="float32")(x)
    modelo = models.Model(entradas, salidas, name="cnn_gestos_desde_cero")
    modelo.compile(
        optimizer=tf.keras.optimizers.Adam(1e-3),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return modelo
