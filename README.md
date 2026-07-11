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

## GPU (RTX 3080 y otras)

El código detecta la GPU automáticamente y activa *memory growth* + precisión mixta
(`mixed_float16`, ideal para Tensor Cores).

### Windows nativo (sin WSL2)

**DirectML** (recomendado): abstracción de Microsoft para GPU en Windows.

```powershell
pip install tensorflow-directml
python modelo_scratch/entrenar.py
```

Funciona con RTX 3080, AMD Radeon, Intel Arc. ~2–3× más rápido que CPU.

### Windows con WSL2

```bash
pip install "tensorflow[and-cuda]"
python modelo_scratch/entrenar.py
```

Requiere drivers NVIDIA + CUDA Toolkit. Algo más configuración, pero máximo rendimiento.

### Linux nativo

```bash
pip install "tensorflow[and-cuda]"
python modelo_scratch/entrenar.py
```

### Laptop sin GPU

El mismo código corre en CPU sin cambios (más lento, pero funcional).
