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
import tensorflow as tf

from datos_freihand import cargar_datasets_freihand
from modelo import configurar_dispositivo, construir_cnn_regresion
from vision_proyecto.config import (
    DIR_RESULTADOS,
    ORDEN_DEDOS,
    RUTA_CNN_REGRESION,
    RUTA_HISTORIAL_REGRESION,
)
from vision_proyecto.utils import asegurar_directorios


# ─────────────────────────────  Gráficas  ─────────────────────────────
def guardar_curvas(historial: dict) -> Path:
    plt.style.use("dark_background")
    figura, ejes = plt.subplots(1, 2, figsize=(12, 4.5))
    figura.patch.set_facecolor("#1e1e1e")

    for eje, metrica, titulo in zip(ejes, ("mae", "loss"),
                                    ("Error medio por dedo (MAE)", "Pérdida (MSE)")):
        eje.set_facecolor("#2d2d2d")
        eje.plot(historial[metrica], color="#2a82da", label="entrenamiento")
        eje.plot(historial[f"val_{metrica}"], color="#3ddc84", label="validación")
        eje.set_title(titulo, color="#dcdcdc")
        eje.set_xlabel("Época")
        eje.legend()
        eje.grid(alpha=0.2)

    ruta = DIR_RESULTADOS / "regresion_curvas_entrenamiento.png"
    figura.tight_layout()
    figura.savefig(ruta, dpi=130)
    plt.close(figura)
    return ruta


def mae_por_dedo(modelo, ds_validacion) -> dict:
    errores = np.zeros(len(ORDEN_DEDOS), dtype=np.float64)
    total = 0
    for imagenes, etiquetas in ds_validacion:
        pred = modelo.predict(imagenes, verbose=0)
        errores += np.abs(pred - etiquetas.numpy()).sum(axis=0)
        total += len(imagenes)
    return {dedo: round(float(e / total), 4)
            for dedo, e in zip(ORDEN_DEDOS, errores)}


# ──────────────────────────────  Entrenamiento  ───────────────────────
def entrenar(epocas: int, tam_lote: int) -> None:
    asegurar_directorios()
    dispositivo = configurar_dispositivo()

    ds_entrenamiento, ds_validacion, info = cargar_datasets_freihand(tam_lote)
    print(f"[DATOS] FreiHAND: {info['entrenamiento']} train / "
          f"{info['validacion']} val (de {info['imagenes_totales']} imágenes)")

    modelo = construir_cnn_regresion()
    modelo.summary()

    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            str(RUTA_CNN_REGRESION), monitor="val_mae",
            mode="min", save_best_only=True),
        tf.keras.callbacks.EarlyStopping(
            monitor="val_mae", mode="min", patience=6, restore_best_weights=True),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.5, patience=3, min_lr=1e-5),
    ]

    inicio = time.perf_counter()
    historia = modelo.fit(
        ds_entrenamiento,
        validation_data=ds_validacion,
        epochs=epocas,
        callbacks=callbacks,
    )
    duracion = time.perf_counter() - inicio

    perdida_val, mae_val = modelo.evaluate(ds_validacion, verbose=0)
    errores_dedo = mae_por_dedo(modelo, ds_validacion)

    registro = {
        "dispositivo": dispositivo,
        "epocas_ejecutadas": len(historia.history["loss"]),
        "segundos_entrenamiento": round(duracion, 1),
        "mae_validacion": round(float(mae_val), 4),
        "mse_validacion": round(float(perdida_val), 4),
        "mae_por_dedo": errores_dedo,
        **{clave: [float(v) for v in valores]
           for clave, valores in historia.history.items()},
    }
    RUTA_HISTORIAL_REGRESION.write_text(json.dumps(registro, indent=2, ensure_ascii=False),
                                        encoding="utf-8")
    ruta_curvas = guardar_curvas(historia.history)

    print(f"\n[OK] Modelo guardado en {RUTA_CNN_REGRESION}")
    print(f"[OK] Historial en {RUTA_HISTORIAL_REGRESION}")
    print(f"[OK] Curvas en {ruta_curvas}")
    print(f"[OK] MAE validación: {mae_val:.4f} ({duracion:.0f}s en {dispositivo})")
    print(f"[OK] MAE por dedo: {errores_dedo}")


# ──────────────────────────────  Entry  ──────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Entrena la CNN de regresión (5 flexiones por dedo) con FreiHAND")
    parser.add_argument("--epocas", type=int, default=25)
    parser.add_argument("--lote", type=int, default=32)
    args = parser.parse_args()
    entrenar(args.epocas, args.lote)
