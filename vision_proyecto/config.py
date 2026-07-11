# ─────────────────────────────  Imports  ─────────────────────────────
from __future__ import annotations

from pathlib import Path


# ──────────────────────────────  Rutas  ──────────────────────────────
RAIZ_PROYECTO = Path(__file__).resolve().parent.parent
DIR_MODELOS = RAIZ_PROYECTO / "modelos"
DIR_RESULTADOS = RAIZ_PROYECTO / "resultados"

RUTA_LANDMARKER = DIR_MODELOS / "hand_landmarker.task"
URL_LANDMARKER = (
    "https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
    "hand_landmarker/float16/latest/hand_landmarker.task"
)

RUTA_CNN = DIR_MODELOS / "cnn_gestos.keras"
RUTA_HISTORIAL_CNN = DIR_MODELOS / "cnn_historial.json"


# ─────────────────────  Servos de la mano (ESP32)  ────────────────────
ORDEN_DEDOS = ["pulgar", "indice", "corazon", "anular", "menique"]

LIMITES_SERVOS = {
    "pulgar":  {"canal": 8, "abierto": 1840, "cerrado": 2420},
    "indice":  {"canal": 6, "abierto": 2140, "cerrado": 540},
    "corazon": {"canal": 4, "abierto": 2125, "cerrado": 510},
    "anular":  {"canal": 2, "abierto": 1820, "cerrado": 520},
    "menique": {"canal": 0, "abierto": 1540, "cerrado": 520},
}


# ────────────────────────  Protocolo serial  ──────────────────────────
BAUDIOS = 115200
HZ_ENVIO_SERIAL = 20
PREFIJO_TRAMA = "S:"


# ──────────────────────────  Visión / cámara  ─────────────────────────
ANCHO_CAMARA = 1280
ALTO_CAMARA = 720
NUM_MANOS = 1
CONFIANZA_DETECCION = 0.5
CONFIANZA_PRESENCIA = 0.5
CONFIANZA_TRACKING = 0.5


# ─────────────────────────  Modelo desde cero  ────────────────────────
TAM_IMAGEN_CNN = 128
CLASES_GESTOS = ["piedra", "papel", "tijera"]

POSTURAS_GESTO = {
    "piedra": [1.0, 1.0, 1.0, 1.0, 1.0],
    "papel":  [0.0, 0.0, 0.0, 0.0, 0.0],
    "tijera": [1.0, 0.0, 0.0, 1.0, 1.0],
}
