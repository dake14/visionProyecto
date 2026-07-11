# ─────────────────────────────  Imports  ─────────────────────────────
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix

from datos import cargar_datasets
from modelo import configurar_dispositivo
from vision_proyecto.config import CLASES_GESTOS, DIR_RESULTADOS, RUTA_CNN
from vision_proyecto.utils import asegurar_directorios


# ─────────────────────────────  Gráficas  ─────────────────────────────
def guardar_matriz_confusion(matriz: np.ndarray) -> Path:
    plt.style.use("dark_background")
    figura, eje = plt.subplots(figsize=(6, 5))
    figura.patch.set_facecolor("#1e1e1e")
    imagen = eje.imshow(matriz, cmap="Blues")
    figura.colorbar(imagen)

    eje.set_xticks(range(len(CLASES_GESTOS)), CLASES_GESTOS)
    eje.set_yticks(range(len(CLASES_GESTOS)), CLASES_GESTOS)
    eje.set_xlabel("Predicción")
    eje.set_ylabel("Real")
    eje.set_title("Matriz de confusión — CNN desde cero", color="#dcdcdc")

    umbral = matriz.max() / 2
    for fila in range(matriz.shape[0]):
        for columna in range(matriz.shape[1]):
            color = "black" if matriz[fila, columna] > umbral else "#dcdcdc"
            eje.text(columna, fila, str(matriz[fila, columna]),
                     ha="center", va="center", color=color)

    ruta = DIR_RESULTADOS / "cnn_matriz_confusion.png"
    figura.tight_layout()
    figura.savefig(ruta, dpi=130)
    plt.close(figura)
    return ruta


# ─────────────────────────────  Evaluación  ───────────────────────────
def evaluar() -> None:
    asegurar_directorios()
    configurar_dispositivo()

    if not RUTA_CNN.exists():
        print(f"[ERROR] No existe {RUTA_CNN}. Entrena primero: python entrenar.py")
        sys.exit(1)

    modelo = tf.keras.models.load_model(RUTA_CNN)
    _, _, ds_prueba, _ = cargar_datasets()

    etiquetas_reales = []
    etiquetas_predichas = []
    for lote_imagenes, lote_etiquetas in ds_prueba:
        probabilidades = modelo.predict(lote_imagenes, verbose=0)
        etiquetas_predichas.extend(np.argmax(probabilidades, axis=1))
        etiquetas_reales.extend(lote_etiquetas.numpy())

    reporte = classification_report(etiquetas_reales, etiquetas_predichas,
                                    target_names=CLASES_GESTOS, digits=4)
    matriz = confusion_matrix(etiquetas_reales, etiquetas_predichas)
    ruta_matriz = guardar_matriz_confusion(matriz)

    ruta_reporte = DIR_RESULTADOS / "cnn_reporte_clasificacion.txt"
    ruta_reporte.write_text(reporte, encoding="utf-8")

    print(reporte)
    print(f"[OK] Reporte en {ruta_reporte}")
    print(f"[OK] Matriz de confusión en {ruta_matriz}")


# ──────────────────────────────  Entry  ──────────────────────────────
if __name__ == "__main__":
    evaluar()
