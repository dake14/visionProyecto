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

## Modelo de regresión: 5 ángulos por dedo (FreiHAND)

A diferencia del clasificador (1 de 3 gestos), esta CNN produce **la flexión
continua 0–1 de cada dedo** directamente — igual que MediaPipe, pero aprendida
desde cero.

Dataset: [FreiHAND](https://lmb.informatik.uni-freiburg.de/resources/datasets/FreihandDataset.en.html)
(~130k imágenes con 21 keypoints 3D). Las etiquetas de flexión se generan
automáticamente pasando los keypoints por `calcular_flexiones_np` — la misma
matemática de ángulos que usa MediaPipe en vivo, así que la salida es
directamente compatible con el protocolo serial de la mano.

Descarga manual (no está en TFDS): baja `FreiHAND_pub_v2.zip` y descomprímelo en
`datasets/FreiHAND_pub_v2/` (debe quedar `training_xyz.json` y `training/rgb/*.jpg`).

```powershell
python modelo_scratch/entrenar_regresion.py --epocas 25 --lote 32
```

**GPU en un clic (Colab):** abre
[`notebooks/entrenar_regresion_colab.ipynb`](https://colab.research.google.com/github/dake14/visionProyecto/blob/master/notebooks/entrenar_regresion_colab.ipynb)
→ `Entorno de ejecución` → GPU → `Ejecutar todo`. El notebook clona el repo,
descarga FreiHAND, entrena en GPU y te descarga `cnn_regresion_dedos.keras`.

Arquitectura: mismos 4 bloques convolucionales, pero la última capa es
`Dense(5, sigmoid)` (regresión) en vez de `Dense(3, softmax)` (clasificación).
Pérdida MSE, métrica MAE por dedo. El modelo entrenado se guarda en
`modelos/cnn_regresion_dedos.keras` y `PredictorAngulosCNN` lo usa para inferencia
en vivo devolviendo las 5 flexiones listas para enviar a la ESP32.

## GPU (RTX 3080 y otras)

El código detecta GPU automáticamente si está disponible (memory growth + precisión mixta).

### Windows nativo: Google Colab (Recomendado)

Usa GPU gratis sin instalar nada localmente:

1. Ve a [colab.research.google.com](https://colab.research.google.com)
2. Nueva notebook
3. Copia y corre:

```python
!git clone https://github.com/dake14/visionProyecto.git
%cd visionProyecto
!pip install -q tensorflow tensorflow-datasets matplotlib scikit-learn

import sys
sys.path.insert(0, '.')
from modelo_scratch.entrenar import entrenar
entrenar(epocas=15, tam_lote=32)
```

Entrena ~50 seg/época con K80/T4/L4 gratis.

### WSL2 + CUDA (Máximo rendimiento local)

```bash
pip install "tensorflow[and-cuda]"
python modelo_scratch/entrenar.py
```

Requiere drivers NVIDIA + CUDA Toolkit (~30 min setup). RTX 3080 ~ 30–40 seg/época.

### Linux nativo

```bash
pip install "tensorflow[and-cuda]"
python modelo_scratch/entrenar.py
```

### Windows nativo sin GPU

```powershell
python modelo_scratch/entrenar.py
```

Entrena en CPU (~75 seg/época), funcional pero lento. Para acelerar, usa Colab o WSL2.
