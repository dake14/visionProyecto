# ─────────────────────────────  Imports  ─────────────────────────────
from __future__ import annotations

import cv2


# ─────────────────────────────  Config  ──────────────────────────────
CONEXIONES_MANO = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12),
    (9, 13), (13, 14), (14, 15), (15, 16),
    (13, 17), (0, 17), (17, 18), (18, 19), (19, 20),
]

COLOR_POR_DEDO = [
    (147, 20, 255),
    (0, 165, 255),
    (0, 255, 0),
    (255, 0, 0),
    (0, 255, 255),
]

RANGOS_DEDOS = [(1, 4), (5, 8), (9, 12), (13, 16), (17, 20)]


# ──────────────────────────────  Core  ───────────────────────────────
def _color_conexion(idx1: int, idx2: int) -> tuple:
    for i, (inicio, fin) in enumerate(RANGOS_DEDOS):
        if inicio <= idx1 <= fin or inicio <= idx2 <= fin:
            return COLOR_POR_DEDO[i]
    return (255, 255, 255)


def dibujar_esqueleto(frame, landmarks, lateralidad: str) -> None:
    alto, ancho = frame.shape[:2]
    puntos = [(int(lm.x * ancho), int(lm.y * alto)) for lm in landmarks]

    for a, b in CONEXIONES_MANO:
        cv2.line(frame, puntos[a], puntos[b], _color_conexion(a, b), 2, cv2.LINE_AA)

    for i, punto in enumerate(puntos):
        radio = 7 if i == 0 else 4
        cv2.circle(frame, punto, radio, (0, 0, 200), -1, cv2.LINE_AA)
        cv2.circle(frame, punto, radio, (255, 255, 255), 1, cv2.LINE_AA)

    x0, y0 = puntos[0]
    cv2.putText(frame, lateralidad, (x0 - 30, y0 + 28),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 255), 2, cv2.LINE_AA)


def dibujar_hud(frame, fps: float, gesto: str, modo: str) -> None:
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (300, 96), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.45, frame, 0.55, 0, frame)

    color_fps = (0, 255, 80) if fps >= 25 else (0, 165, 255)
    cv2.putText(frame, f"FPS:   {fps:.1f}", (10, 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, color_fps, 2, cv2.LINE_AA)
    cv2.putText(frame, f"Modo:  {modo}", (10, 52),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1, cv2.LINE_AA)
    cv2.putText(frame, f"Gesto: {gesto}", (10, 80),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (80, 200, 255), 1, cv2.LINE_AA)
