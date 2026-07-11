# visionProyecto — Mano robótica controlada por visión artificial

Control de una mano robótica (ESP32 + PCA9685 + 5 servos) mediante tracking de manos
con MediaPipe, con interfaz gráfica interactiva, más una CNN entrenada **desde cero**
en TensorFlow para análisis y comparación de rendimiento.

## Estructura

```
visionProyecto/
├── main.py                     # Lanza la interfaz gráfica
├── requirements.txt
├── vision_proyecto/
│   ├── config.py               # Límites de servos, protocolo serial, rutas
│   ├── core/
│   │   ├── tracking.py         # MediaPipe HandLandmarker (LIVE_STREAM e IMAGE)
│   │   ├── flexion.py          # Landmarks → flexión 0..1 por dedo + gesto
│   │   ├── dibujo.py           # Esqueleto y HUD sobre el video
│   │   ├── serial_esp32.py     # Envío de tramas a la ESP32 (pyserial)
│   │   └── inferencia_cnn.py   # Inferencia en vivo con la CNN propia
│   └── ui/                     # Interfaz customtkinter (tema oscuro)
├── esp32/
│   ├── main.py                 # MicroPython: recibe tramas y mueve los servos
│   └── pca9685.py              # Driver del PCA9685
├── modelo_scratch/
│   ├── modelo.py               # Arquitectura CNN + detección GPU/CPU
│   ├── datos.py                # Dataset rock_paper_scissors (TFDS, descarga automática)
│   ├── entrenar.py             # Entrenamiento con curvas
│   ├── evaluar.py              # Reporte + matriz de confusión
│   └── comparar.py             # Benchmark MediaPipe vs CNN (latencia, FPS, exactitud)
├── modelos/                    # hand_landmarker.task, cnn_gestos.keras (generados)
└── resultados/                 # Gráficas y métricas (generados)
```

## Instalación

```powershell
cd visionProyecto
.venv\Scripts\activate
pip install -r requirements.txt
```

El modelo `hand_landmarker.task` (~8 MB) se descarga automáticamente del repositorio
oficial de MediaPipe la primera vez que se ejecuta la app o el benchmark.

## Uso de la interfaz

```powershell
python main.py            # cámara 0
python main.py --camara 1
```

Modos disponibles:

| Modo        | Comportamiento                                                        |
|-------------|-----------------------------------------------------------------------|
| Manual      | Sliders por dedo → la mano copia los valores                          |
| Visión      | MediaPipe calcula la flexión real de cada dedo y la envía a la ESP32  |
| CNN propia  | La CNN clasifica piedra/papel/tijera y aplica la postura equivalente  |

Conecta la ESP32 por USB, elige el puerto COM en el panel y pulsa **Conectar**.

## ESP32 (MicroPython)

Copia `esp32/main.py` y `esp32/pca9685.py` a la placa (Thonny o `mpremote cp`).
La placa espera tramas `S:p,i,c,a,m\n` con la flexión 0–100 de cada dedo
(pulgar, índice, corazón, anular, meñique) a 115200 baudios. Incluye:

- Movimiento suavizado (máx. 40 µs por ciclo de 20 ms).
- Failsafe: si no llegan datos en 3 s, la mano vuelve a posición abierta.
- Los límites `abierto`/`cerrado` de cada servo son los calibrados en el
  código de referencia original.

## Modelo desde cero (TensorFlow)

Dataset: [rock_paper_scissors](https://www.tensorflow.org/datasets/catalog/rock_paper_scissors)
(TFDS, ~220 MB, descarga automática). Piedra = mano cerrada, papel = mano abierta,
tijera = índice y corazón extendidos — el mismo vocabulario de gestos que usa la mano.

```powershell
python modelo_scratch/entrenar.py --epocas 20 --lote 32
python modelo_scratch/evaluar.py
python modelo_scratch/comparar.py --muestras 200
```

`comparar.py` mide latencia por imagen, FPS equivalente, exactitud y tasa de
detección tanto del pipeline MediaPipe+reglas como de la CNN, y guarda tabla,
JSON y gráfica en `resultados/`.

## GPU (RTX 3080)

El código detecta la GPU automáticamente (`tf.config.list_physical_devices`) y, si
existe, activa *memory growth* y precisión mixta (`mixed_float16`, ideal para los
Tensor Cores de la RTX 3080). Si no hay GPU, entrena en CPU sin cambios.

Importante en Windows: TensorFlow ≥ 2.11 ya no soporta CUDA en Windows nativo,
así que en esta instalación (Python 3.12 + Windows) entrena en CPU. Para
aprovechar la RTX 3080 hay dos opciones:

1. **WSL2 (recomendado)**: dentro de Ubuntu/WSL2 con drivers NVIDIA actuales:
   ```bash
   python3 -m venv .venv && source .venv/bin/activate
   pip install "tensorflow[and-cuda]" tensorflow-datasets matplotlib scikit-learn
   python modelo_scratch/entrenar.py
   ```
2. **Linux nativo**: mismo comando que en WSL2.

En cualquier otra laptop sin GPU el mismo código corre en CPU sin tocar nada.
