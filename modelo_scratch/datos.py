# ─────────────────────────────  Imports  ─────────────────────────────
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import tensorflow as tf
import tensorflow_datasets as tfds

from vision_proyecto.config import TAM_IMAGEN_CNN


# ─────────────────────────────  Config  ──────────────────────────────
NOMBRE_DATASET = "rock_paper_scissors"
AUTOTUNE = tf.data.AUTOTUNE


# ──────────────────────────────  Core  ───────────────────────────────
def _preprocesar(imagen, etiqueta):
    imagen = tf.image.resize(imagen, (TAM_IMAGEN_CNN, TAM_IMAGEN_CNN))
    imagen = tf.cast(imagen, tf.float32) / 255.0
    return imagen, etiqueta


def _aumentar(imagen, etiqueta):
    imagen = tf.image.random_flip_left_right(imagen)
    imagen = tf.image.random_brightness(imagen, 0.15)
    imagen = tf.image.random_contrast(imagen, 0.8, 1.2)
    imagen = tf.clip_by_value(imagen, 0.0, 1.0)
    return imagen, etiqueta


def cargar_datasets(tam_lote: int = 32):
    (ds_entrenamiento, ds_validacion, ds_prueba), info = tfds.load(
        NOMBRE_DATASET,
        split=["train[:85%]", "train[85%:]", "test"],
        as_supervised=True,
        with_info=True,
    )

    ds_entrenamiento = (
        ds_entrenamiento
        .map(_preprocesar, num_parallel_calls=AUTOTUNE)
        .cache()
        .shuffle(2048)
        .map(_aumentar, num_parallel_calls=AUTOTUNE)
        .batch(tam_lote)
        .prefetch(AUTOTUNE)
    )
    ds_validacion = (
        ds_validacion
        .map(_preprocesar, num_parallel_calls=AUTOTUNE)
        .cache()
        .batch(tam_lote)
        .prefetch(AUTOTUNE)
    )
    ds_prueba = (
        ds_prueba
        .map(_preprocesar, num_parallel_calls=AUTOTUNE)
        .batch(tam_lote)
        .prefetch(AUTOTUNE)
    )
    return ds_entrenamiento, ds_validacion, ds_prueba, info


def cargar_prueba_cruda():
    return tfds.load(NOMBRE_DATASET, split="test", as_supervised=True)
