# ─────────────────────────────  Imports  ─────────────────────────────
from __future__ import annotations

import numpy as np

from vision_proyecto.config import CLASES_GESTOS, RUTA_CNN, TAM_IMAGEN_CNN


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

    @staticmethod
    def _recorte_central(frame_rgb: np.ndarray) -> np.ndarray:
        alto, ancho = frame_rgb.shape[:2]
        lado = min(alto, ancho)
        y0 = (alto - lado) // 2
        x0 = (ancho - lado) // 2
        return frame_rgb[y0:y0 + lado, x0:x0 + lado]

    @staticmethod
    def _recorte_bbox(frame_rgb: np.ndarray, bbox: tuple) -> np.ndarray:
        alto, ancho = frame_rgb.shape[:2]
        x0n, y0n, x1n, y1n = bbox
        x0, x1 = int(x0n * ancho), int(x1n * ancho)
        y0, y1 = int(y0n * alto), int(y1n * alto)
        return frame_rgb[y0:y1, x0:x1]

    def _preparar(self, frame_rgb: np.ndarray, bbox: tuple | None) -> np.ndarray:
        recorte = self._recorte_bbox(frame_rgb, bbox) if bbox else self._recorte_central(frame_rgb)
        if recorte.size == 0:
            recorte = self._recorte_central(frame_rgb)
        imagen = self._tf.image.resize(recorte, (TAM_IMAGEN_CNN, TAM_IMAGEN_CNN))
        imagen = self._tf.cast(imagen, self._tf.float32) / 255.0
        return self._tf.expand_dims(imagen, 0)

    def predecir(self, frame_rgb: np.ndarray,
                 bbox: tuple | None = None) -> tuple[str, float]:
        logits = self._infer(self._preparar(frame_rgb, bbox))
        probabilidades = np.asarray(logits)[0]
        indice = int(np.argmax(probabilidades))
        return CLASES_GESTOS[indice], float(probabilidades[indice])
