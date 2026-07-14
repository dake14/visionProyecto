# ─────────────────────────────  Imports  ─────────────────────────────
from __future__ import annotations

import numpy as np

from vision_proyecto.config import (
    CLASES_GESTOS,
    RUTA_CNN,
    RUTA_CNN_REGRESION,
    TAM_IMAGEN_CNN,
)


# ──────────────────────────────  Recortes  ───────────────────────────
def _recorte_central(frame_rgb: np.ndarray) -> np.ndarray:
    alto, ancho = frame_rgb.shape[:2]
    lado = min(alto, ancho)
    y0 = (alto - lado) // 2
    x0 = (ancho - lado) // 2
    return frame_rgb[y0:y0 + lado, x0:x0 + lado]


def _recorte_bbox(frame_rgb: np.ndarray, bbox: tuple) -> np.ndarray:
    alto, ancho = frame_rgb.shape[:2]
    x0n, y0n, x1n, y1n = bbox
    x0, x1 = int(x0n * ancho), int(x1n * ancho)
    y0, y1 = int(y0n * alto), int(y1n * alto)
    return frame_rgb[y0:y1, x0:x1]


def _preparar_entrada(tf, frame_rgb: np.ndarray, bbox: tuple | None):
    recorte = _recorte_bbox(frame_rgb, bbox) if bbox else _recorte_central(frame_rgb)
    if recorte.size == 0:
        recorte = _recorte_central(frame_rgb)
    imagen = tf.image.resize(recorte, (TAM_IMAGEN_CNN, TAM_IMAGEN_CNN))
    imagen = tf.cast(imagen, tf.float32) / 255.0
    return tf.expand_dims(imagen, 0)


# ──────────────────────────────  Clases  ─────────────────────────────
class ClasificadorGestosCNN:

    def __init__(self):
        import tensorflow as tf

        self._tf = tf
        self._modelo = tf.keras.models.load_model(RUTA_CNN)
        self._infer = tf.function(
            lambda x: self._modelo(x, training=False),
            reduce_retracing=True,
        )

    @staticmethod
    def disponible() -> bool:
        return RUTA_CNN.exists()

    def predecir(self, frame_rgb: np.ndarray,
                 bbox: tuple | None = None) -> tuple[str, float]:
        logits = self._infer(_preparar_entrada(self._tf, frame_rgb, bbox))
        probabilidades = np.asarray(logits)[0]
        indice = int(np.argmax(probabilidades))
        return CLASES_GESTOS[indice], float(probabilidades[indice])


class PredictorAngulosCNN:

    def __init__(self):
        import tensorflow as tf

        self._tf = tf
        self._modelo = tf.keras.models.load_model(RUTA_CNN_REGRESION)
        self._infer = tf.function(
            lambda x: self._modelo(x, training=False),
            reduce_retracing=True,
        )

    @staticmethod
    def disponible() -> bool:
        return RUTA_CNN_REGRESION.exists()

    def predecir(self, frame_rgb: np.ndarray,
                 bbox: tuple | None = None) -> list[float]:
        salida = self._infer(_preparar_entrada(self._tf, frame_rgb, bbox))
        return [float(v) for v in np.asarray(salida)[0]]
