# ─────────────────────────────  Imports  ─────────────────────────────
from __future__ import annotations

import urllib.request

from vision_proyecto.config import DIR_MODELOS, DIR_RESULTADOS, RUTA_LANDMARKER, URL_LANDMARKER


# ──────────────────────────────  Core  ───────────────────────────────
def asegurar_directorios() -> None:
    DIR_MODELOS.mkdir(parents=True, exist_ok=True)
    DIR_RESULTADOS.mkdir(parents=True, exist_ok=True)


def asegurar_modelo_landmarker() -> str:
    asegurar_directorios()
    if not RUTA_LANDMARKER.exists():
        print(f"[SETUP] Descargando hand_landmarker.task desde {URL_LANDMARKER}")
        urllib.request.urlretrieve(URL_LANDMARKER, RUTA_LANDMARKER)
        print(f"[SETUP] Modelo guardado en {RUTA_LANDMARKER}")
    return str(RUTA_LANDMARKER)
