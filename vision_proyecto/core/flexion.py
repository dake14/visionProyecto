# ─────────────────────────────  Imports  ─────────────────────────────
from __future__ import annotations

import math

from vision_proyecto.config import ORDEN_DEDOS


# ─────────────────────────────  Config  ──────────────────────────────
CADENAS_DEDOS = {
    "pulgar":  (1, 2, 3, 4),
    "indice":  (5, 6, 7, 8),
    "corazon": (9, 10, 11, 12),
    "anular":  (13, 14, 15, 16),
    "menique": (17, 18, 19, 20),
}

ANGULO_EXTENDIDO = 175.0
RANGO_FLEXION = 80.0
ANGULO_EXTENDIDO_PULGAR = 172.0
RANGO_FLEXION_PULGAR = 40.0


# ──────────────────────────────  Core  ───────────────────────────────
def _angulo(a, b, c) -> float:
    v1 = (a.x - b.x, a.y - b.y, a.z - b.z)
    v2 = (c.x - b.x, c.y - b.y, c.z - b.z)
    punto = sum(p * q for p, q in zip(v1, v2))
    norma = math.sqrt(sum(p * p for p in v1)) * math.sqrt(sum(q * q for q in v2))
    if norma == 0:
        return ANGULO_EXTENDIDO
    coseno = max(-1.0, min(1.0, punto / norma))
    return math.degrees(math.acos(coseno))


def _flexion_dedo(landmarks, dedo: str) -> float:
    mcp, pip, dip, tip = CADENAS_DEDOS[dedo]
    if dedo == "pulgar":
        angulo = (_angulo(landmarks[mcp], landmarks[pip], landmarks[dip])
                  + _angulo(landmarks[pip], landmarks[dip], landmarks[tip])) / 2
        cruda = (ANGULO_EXTENDIDO_PULGAR - angulo) / RANGO_FLEXION_PULGAR
    else:
        angulo = (_angulo(landmarks[mcp], landmarks[pip], landmarks[dip])
                  + _angulo(landmarks[pip], landmarks[dip], landmarks[tip])) / 2
        cruda = (ANGULO_EXTENDIDO - angulo) / RANGO_FLEXION
    return max(0.0, min(1.0, cruda))


def calcular_flexiones(landmarks) -> list[float]:
    return [_flexion_dedo(landmarks, dedo) for dedo in ORDEN_DEDOS]


def clasificar_gesto(flexiones: list[float]) -> str:
    pulgar, indice, corazon, anular, menique = flexiones
    promedio = sum(flexiones) / len(flexiones)
    if promedio < 0.30:
        return "papel"
    if indice < 0.35 and corazon < 0.35 and anular > 0.55 and menique > 0.55:
        return "tijera"
    if promedio > 0.60:
        return "piedra"
    return "indefinido"


class SuavizadorFlexiones:

    def __init__(self, alfa: float = 0.5):
        self._alfa = alfa
        self._valores: list[float] | None = None

    def actualizar(self, flexiones: list[float]) -> list[float]:
        if self._valores is None:
            self._valores = list(flexiones)
        else:
            self._valores = [
                self._alfa * nuevo + (1 - self._alfa) * previo
                for nuevo, previo in zip(flexiones, self._valores)
            ]
        return list(self._valores)

    def reiniciar(self) -> None:
        self._valores = None
