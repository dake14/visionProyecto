# ─────────────────────────────  Imports  ─────────────────────────────
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import tensorflow as tf

from vision_proyecto.config import DIR_FREIHAND, TAM_IMAGEN_CNN
from vision_proyecto.core.flexion import calcular_flexiones_np


# ─────────────────────────────  Config  ──────────────────────────────
AUTOTUNE = tf.data.AUTOTUNE
RUTA_XYZ = DIR_FREIHAND / "training_xyz.json"
DIR_RGB = DIR_FREIHAND / "training" / "rgb"
SEMILLA = 42


# ──────────────────────────────  Core  ───────────────────────────────
def _verificar_dataset() -> None:
    if RUTA_XYZ.exists() and DIR_RGB.exists():
        return
    raise FileNotFoundError(
        f"No se encontró FreiHAND en {DIR_FREIHAND}.\n"
        "1. Descarga FreiHAND_pub_v2.zip de\n"
        "   https://lmb.informatik.uni-freiburg.de/resources/datasets/FreihandDataset.en.html\n"
        f"2. Descomprímelo en {DIR_FREIHAND}\n"
        "   (debe quedar training_xyz.json y training/rgb/*.jpg dentro)"
    )


def _rutas_imagenes() -> list[str]:
    rutas = sorted(str(p) for p in DIR_RGB.glob("*.jpg"))
    if not rutas:
        raise FileNotFoundError(f"No hay imágenes .jpg en {DIR_RGB}")
    return rutas


def _flexiones_base() -> np.ndarray:
    puntos = np.asarray(json.loads(RUTA_XYZ.read_text()), dtype=np.float32)
    return np.asarray([calcular_flexiones_np(p) for p in puntos], dtype=np.float32)


def _cargar_imagen(ruta, etiqueta):
    datos = tf.io.read_file(ruta)
    imagen = tf.image.decode_jpeg(datos, channels=3)
    imagen = tf.image.resize(imagen, (TAM_IMAGEN_CNN, TAM_IMAGEN_CNN))
    imagen = tf.cast(imagen, tf.float32) / 255.0
    return imagen, etiqueta


def _aumentar(imagen, etiqueta):
    imagen = tf.image.random_brightness(imagen, 0.15)
    imagen = tf.image.random_contrast(imagen, 0.8, 1.2)
    imagen = tf.clip_by_value(imagen, 0.0, 1.0)
    return imagen, etiqueta


def cargar_datasets_freihand(tam_lote: int = 32, fraccion_val: float = 0.1):
    _verificar_dataset()

    rutas = np.asarray(_rutas_imagenes())
    flex_base = _flexiones_base()
    base = len(flex_base)
    etiquetas = np.stack([flex_base[i % base] for i in range(len(rutas))])

    rng = np.random.default_rng(SEMILLA)
    orden = rng.permutation(len(rutas))
    rutas, etiquetas = rutas[orden], etiquetas[orden]

    n_val = int(len(rutas) * fraccion_val)
    rutas_val, etiquetas_val = rutas[:n_val], etiquetas[:n_val]
    rutas_tr, etiquetas_tr = rutas[n_val:], etiquetas[n_val:]

    ds_entrenamiento = (
        tf.data.Dataset.from_tensor_slices((rutas_tr, etiquetas_tr))
        .shuffle(4096, seed=SEMILLA)
        .map(_cargar_imagen, num_parallel_calls=AUTOTUNE)
        .map(_aumentar, num_parallel_calls=AUTOTUNE)
        .batch(tam_lote)
        .prefetch(AUTOTUNE)
    )
    ds_validacion = (
        tf.data.Dataset.from_tensor_slices((rutas_val, etiquetas_val))
        .map(_cargar_imagen, num_parallel_calls=AUTOTUNE)
        .batch(tam_lote)
        .prefetch(AUTOTUNE)
    )

    info = {
        "muestras_base": int(base),
        "imagenes_totales": int(len(rutas)),
        "entrenamiento": int(len(rutas_tr)),
        "validacion": int(len(rutas_val)),
    }
    return ds_entrenamiento, ds_validacion, info
