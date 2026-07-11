# ─────────────────────────────  Imports  ─────────────────────────────
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from datos import cargar_prueba_cruda
from vision_proyecto.config import CLASES_GESTOS, DIR_RESULTADOS, RUTA_CNN
from vision_proyecto.core.flexion import calcular_flexiones, clasificar_gesto
from vision_proyecto.core.tracking import RastreadorImagen
from vision_proyecto.utils import asegurar_directorios, asegurar_modelo_landmarker


# ─────────────────────────────  Config  ──────────────────────────────
CALENTAMIENTO = 10
GESTO_A_INDICE = {gesto: indice for indice, gesto in enumerate(CLASES_GESTOS)}


# ───────────────────────────  Benchmarks  ─────────────────────────────
def evaluar_mediapipe(imagenes: list[np.ndarray], etiquetas: list[int]) -> dict:
    rastreador = RastreadorImagen(asegurar_modelo_landmarker())
    for imagen in imagenes[:CALENTAMIENTO]:
        rastreador.detectar(imagen)

    latencias = []
    predicciones = []
    detectadas = 0
    for imagen in imagenes:
        inicio = time.perf_counter()
        resultado = rastreador.detectar(imagen)
        latencias.append((time.perf_counter() - inicio) * 1000)

        if resultado.hand_landmarks:
            detectadas += 1
            flexiones = calcular_flexiones(resultado.hand_landmarks[0])
            predicciones.append(GESTO_A_INDICE.get(clasificar_gesto(flexiones), -1))
        else:
            predicciones.append(-1)

    rastreador.cerrar()
    aciertos = sum(1 for p, e in zip(predicciones, etiquetas) if p == e)
    return {
        "nombre": "MediaPipe + reglas",
        "latencia_media_ms": float(np.mean(latencias)),
        "latencia_std_ms": float(np.std(latencias)),
        "fps_equivalente": 1000.0 / float(np.mean(latencias)),
        "exactitud": aciertos / len(etiquetas),
        "tasa_deteccion_mano": detectadas / len(imagenes),
    }


def evaluar_cnn(imagenes: list[np.ndarray], etiquetas: list[int]) -> dict:
    from vision_proyecto.core.inferencia_cnn import ClasificadorGestosCNN

    clasificador = ClasificadorGestosCNN()
    for imagen in imagenes[:CALENTAMIENTO]:
        clasificador.predecir(imagen)

    latencias = []
    predicciones = []
    for imagen in imagenes:
        inicio = time.perf_counter()
        gesto, _ = clasificador.predecir(imagen)
        latencias.append((time.perf_counter() - inicio) * 1000)
        predicciones.append(GESTO_A_INDICE[gesto])

    aciertos = sum(1 for p, e in zip(predicciones, etiquetas) if p == e)
    return {
        "nombre": "CNN desde cero",
        "latencia_media_ms": float(np.mean(latencias)),
        "latencia_std_ms": float(np.std(latencias)),
        "fps_equivalente": 1000.0 / float(np.mean(latencias)),
        "exactitud": aciertos / len(etiquetas),
        "tasa_deteccion_mano": 1.0,
    }


# ─────────────────────────────  Salidas  ──────────────────────────────
def guardar_grafica(resultados: list[dict]) -> Path:
    plt.style.use("dark_background")
    figura, (eje_lat, eje_acc) = plt.subplots(1, 2, figsize=(12, 4.5))
    figura.patch.set_facecolor("#1e1e1e")

    nombres = [r["nombre"] for r in resultados]
    colores = ["#2a82da", "#3ddc84"]

    latencias = [r["latencia_media_ms"] for r in resultados]
    errores = [r["latencia_std_ms"] for r in resultados]
    eje_lat.set_facecolor("#2d2d2d")
    eje_lat.bar(nombres, latencias, yerr=errores, color=colores, capsize=6)
    eje_lat.set_title("Latencia por imagen (ms)", color="#dcdcdc")
    eje_lat.grid(axis="y", alpha=0.2)

    exactitudes = [r["exactitud"] * 100 for r in resultados]
    eje_acc.set_facecolor("#2d2d2d")
    eje_acc.bar(nombres, exactitudes, color=colores)
    eje_acc.set_title("Exactitud en test (%)", color="#dcdcdc")
    eje_acc.set_ylim(0, 100)
    eje_acc.grid(axis="y", alpha=0.2)

    ruta = DIR_RESULTADOS / "comparacion_mediapipe_vs_cnn.png"
    figura.tight_layout()
    figura.savefig(ruta, dpi=130)
    plt.close(figura)
    return ruta


def imprimir_tabla(resultados: list[dict]) -> None:
    encabezado = (f"{'Sistema':<22}{'Latencia (ms)':>15}{'FPS':>8}"
                  f"{'Exactitud':>12}{'Detección':>12}")
    print("\n" + encabezado)
    print("-" * len(encabezado))
    for r in resultados:
        print(f"{r['nombre']:<22}"
              f"{r['latencia_media_ms']:>11.2f} ±{r['latencia_std_ms']:>2.1f}"
              f"{r['fps_equivalente']:>8.1f}"
              f"{r['exactitud']:>11.2%}"
              f"{r['tasa_deteccion_mano']:>11.2%}")


# ──────────────────────────────  Main  ───────────────────────────────
def comparar(muestras: int) -> None:
    asegurar_directorios()

    if not RUTA_CNN.exists():
        print(f"[ERROR] No existe {RUTA_CNN}. Entrena primero: python entrenar.py")
        sys.exit(1)

    print(f"[DATOS] Cargando {muestras} imágenes del split de prueba...")
    imagenes = []
    etiquetas = []
    for imagen, etiqueta in cargar_prueba_cruda().take(muestras):
        imagenes.append(imagen.numpy().astype(np.uint8))
        etiquetas.append(int(etiqueta.numpy()))

    print("[BENCH] MediaPipe + reglas de flexión...")
    resultado_mp = evaluar_mediapipe(imagenes, etiquetas)
    print("[BENCH] CNN desde cero...")
    resultado_cnn = evaluar_cnn(imagenes, etiquetas)

    resultados = [resultado_mp, resultado_cnn]
    imprimir_tabla(resultados)

    ruta_json = DIR_RESULTADOS / "comparacion_rendimiento.json"
    ruta_json.write_text(json.dumps(resultados, indent=2, ensure_ascii=False),
                         encoding="utf-8")
    ruta_grafica = guardar_grafica(resultados)
    print(f"\n[OK] Resultados en {ruta_json}")
    print(f"[OK] Gráfica en {ruta_grafica}")


# ──────────────────────────────  Entry  ──────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Compara rendimiento: MediaPipe vs CNN desde cero")
    parser.add_argument("--muestras", type=int, default=200)
    args = parser.parse_args()
    comparar(args.muestras)
