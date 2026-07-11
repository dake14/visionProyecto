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

    def _preparar(self, frame_rgb: np.ndarray) -> np.ndarray:
        alto, ancho = frame_rgb.shape[:2]
        lado = min(alto, ancho)
        y0 = (alto - lado) // 2
        x0 = (ancho - lado) // 2
        recorte = frame_rgb[y0:y0 + lado, x0:x0 + lado]
        imagen = self._tf.image.resize(recorte, (TAM_IMAGEN_CNN, TAM_IMAGEN_CNN))
        imagen = self._tf.cast(imagen, self._tf.float32) / 255.0
        return self._tf.expand_dims(imagen, 0)

    def predecir(self, frame_rgb: np.ndarray) -> tuple[str, float]:
        logits = self._infer(self._preparar(frame_rgb))
        probabilidades = np.asarray(logits)[0]
        indice = int(np.argmax(probabilidades))
        return CLASES_GESTOS[indice], float(probabilidades[indice])
