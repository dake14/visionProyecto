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
import tensorflow as tf

from datos import cargar_datasets
from modelo import configurar_dispositivo, construir_cnn
from vision_proyecto.config import DIR_RESULTADOS, RUTA_CNN, RUTA_HISTORIAL_CNN
from vision_proyecto.utils import asegurar_directorios


# ─────────────────────────────  Gráficas  ─────────────────────────────
def guardar_curvas(historial: dict) -> Path:
    plt.style.use("dark_background")
    figura, ejes = plt.subplots(1, 2, figsize=(12, 4.5))
    figura.patch.set_facecolor("#1e1e1e")

    for eje, metrica, titulo in zip(ejes, ("accuracy", "loss"), ("Exactitud", "Pérdida")):
        eje.set_facecolor("#2d2d2d")
        eje.plot(historial[metrica], color="#2a82da", label="entrenamiento")
        eje.plot(historial[f"val_{metrica}"], color="#3ddc84", label="validación")
        eje.set_title(titulo, color="#dcdcdc")
        eje.set_xlabel("Época")
        eje.legend()
        eje.grid(alpha=0.2)

    ruta = DIR_RESULTADOS / "cnn_curvas_entrenamiento.png"
    figura.tight_layout()
    figura.savefig(ruta, dpi=130)
    plt.close(figura)
    return ruta


# ──────────────────────────────  Entrenamiento  ───────────────────────
def entrenar(epocas: int, tam_lote: int) -> None:
    asegurar_directorios()
    dispositivo = configurar_dispositivo()

    ds_entrenamiento, ds_validacion, ds_prueba, info = cargar_datasets(tam_lote)
    print(f"[DATOS] {info.name}: {info.splits['train'].num_examples} train / "
          f"{info.splits['test'].num_examples} test")

    modelo = construir_cnn()
    modelo.summary()

    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            str(RUTA_CNN), monitor="val_accuracy", save_best_only=True),
        tf.keras.callbacks.EarlyStopping(
            monitor="val_accuracy", patience=6, restore_best_weights=True),
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

    perdida_test, exactitud_test = modelo.evaluate(ds_prueba, verbose=0)

    registro = {
        "dispositivo": dispositivo,
        "epocas_ejecutadas": len(historia.history["loss"]),
        "segundos_entrenamiento": round(duracion, 1),
        "exactitud_test": round(float(exactitud_test), 4),
        "perdida_test": round(float(perdida_test), 4),
        **{clave: [float(v) for v in valores]
           for clave, valores in historia.history.items()},
    }
    RUTA_HISTORIAL_CNN.write_text(json.dumps(registro, indent=2), encoding="utf-8")
    ruta_curvas = guardar_curvas(historia.history)

    print(f"\n[OK] Modelo guardado en {RUTA_CNN}")
    print(f"[OK] Historial en {RUTA_HISTORIAL_CNN}")
    print(f"[OK] Curvas en {ruta_curvas}")
    print(f"[OK] Exactitud en test: {exactitud_test:.2%} "
          f"({duracion:.0f}s en {dispositivo})")


# ──────────────────────────────  Entry  ──────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Entrena la CNN de gestos desde cero")
    parser.add_argument("--epocas", type=int, default=20)
    parser.add_argument("--lote", type=int, default=32)
    args = parser.parse_args()
    entrenar(args.epocas, args.lote)
