# ─────────────────────────────  Imports  ─────────────────────────────
from __future__ import annotations

import threading
import time

import mediapipe as mp
from mediapipe.tasks.python import BaseOptions
from mediapipe.tasks.python.vision import (
    HandLandmarker,
    HandLandmarkerOptions,
    RunningMode,
)

from vision_proyecto.config import (
    CONFIANZA_DETECCION,
    CONFIANZA_PRESENCIA,
    CONFIANZA_TRACKING,
    NUM_MANOS,
)


# ──────────────────────────────  Clases  ─────────────────────────────
class RastreadorManos:

    def __init__(self, ruta_modelo: str):
        self._lock = threading.Lock()
        self._resultado = None
        self._ultimo_timestamp = 0
        opciones = HandLandmarkerOptions(
            base_options=BaseOptions(
                model_asset_path=ruta_modelo,
                delegate=BaseOptions.Delegate.CPU,
            ),
            running_mode=RunningMode.LIVE_STREAM,
            num_hands=NUM_MANOS,
            min_hand_detection_confidence=CONFIANZA_DETECCION,
            min_hand_presence_confidence=CONFIANZA_PRESENCIA,
            min_tracking_confidence=CONFIANZA_TRACKING,
            result_callback=self._al_recibir_resultado,
        )
        self._landmarker = HandLandmarker.create_from_options(opciones)

    def _al_recibir_resultado(self, result, output_image, timestamp_ms: int) -> None:
        with self._lock:
            self._resultado = result

    def procesar_frame(self, frame_rgb) -> None:
        timestamp_ms = int(time.perf_counter() * 1000)
        if timestamp_ms <= self._ultimo_timestamp:
            timestamp_ms = self._ultimo_timestamp + 1
        self._ultimo_timestamp = timestamp_ms
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
        self._landmarker.detect_async(mp_image, timestamp_ms)

    def ultimo_resultado(self):
        with self._lock:
            return self._resultado

    def cerrar(self) -> None:
        self._landmarker.close()


class RastreadorImagen:

    def __init__(self, ruta_modelo: str):
        opciones = HandLandmarkerOptions(
            base_options=BaseOptions(
                model_asset_path=ruta_modelo,
                delegate=BaseOptions.Delegate.CPU,
            ),
            running_mode=RunningMode.IMAGE,
            num_hands=NUM_MANOS,
            min_hand_detection_confidence=CONFIANZA_DETECCION,
        )
        self._landmarker = HandLandmarker.create_from_options(opciones)

    def detectar(self, frame_rgb):
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
        return self._landmarker.detect(mp_image)

    def cerrar(self) -> None:
        self._landmarker.close()
