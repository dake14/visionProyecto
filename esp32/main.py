# ─────────────────────────────  Imports  ─────────────────────────────
import select
import sys
import time

from machine import I2C, Pin
from pca9685 import PCA9685


# ─────────────────────────────  Config  ──────────────────────────────
PIN_SCL = 22
PIN_SDA = 21
FRECUENCIA_PWM = 50
PERIODO_US = 20000
RESOLUCION = 4096

PREFIJO_TRAMA = "S:"
TIMEOUT_SEGURIDAD_MS = 3000
PASO_MAX_US = 40
HZ_CONTROL = 50

DEDOS = [
    {"nombre": "pulgar",  "canal": 8, "abierto": 1840, "cerrado": 2420},
    {"nombre": "indice",  "canal": 6, "abierto": 2140, "cerrado": 540},
    {"nombre": "corazon", "canal": 4, "abierto": 2125, "cerrado": 510},
    {"nombre": "anular",  "canal": 2, "abierto": 1820, "cerrado": 520},
    {"nombre": "menique", "canal": 0, "abierto": 1540, "cerrado": 520},
]


# ──────────────────────────────  Setup  ──────────────────────────────
i2c = I2C(0, scl=Pin(PIN_SCL), sda=Pin(PIN_SDA))
pca = PCA9685(i2c)
pca.freq(FRECUENCIA_PWM)

entrada = select.poll()
entrada.register(sys.stdin, select.POLLIN)


# ──────────────────────────────  Core  ───────────────────────────────
def set_us(canal, us):
    pca.duty(canal, int(us * RESOLUCION / PERIODO_US))


def flexion_a_us(dedo, porcentaje):
    return dedo["abierto"] + (dedo["cerrado"] - dedo["abierto"]) * porcentaje / 100.0


def leer_trama():
    linea = None
    while entrada.poll(0):
        linea = sys.stdin.readline()
    if linea and linea.startswith(PREFIJO_TRAMA):
        try:
            valores = [int(v) for v in linea[len(PREFIJO_TRAMA):].strip().split(",")]
            if len(valores) == len(DEDOS):
                return [max(0, min(100, v)) for v in valores]
        except ValueError:
            pass
    return None


def acercar(actual, objetivo):
    delta = objetivo - actual
    if delta > PASO_MAX_US:
        return actual + PASO_MAX_US
    if delta < -PASO_MAX_US:
        return actual - PASO_MAX_US
    return objetivo


def bucle_principal():
    posiciones = [float(d["abierto"]) for d in DEDOS]
    objetivos = list(posiciones)

    for dedo, posicion in zip(DEDOS, posiciones):
        set_us(dedo["canal"], posicion)
    time.sleep(1)

    ultimo_dato = time.ticks_ms()
    periodo_ms = int(1000 / HZ_CONTROL)
    print("Mano lista. Esperando tramas " + PREFIJO_TRAMA + "p,i,c,a,m")

    while True:
        porcentajes = leer_trama()
        if porcentajes is not None:
            objetivos = [flexion_a_us(dedo, p) for dedo, p in zip(DEDOS, porcentajes)]
            ultimo_dato = time.ticks_ms()

        if time.ticks_diff(time.ticks_ms(), ultimo_dato) > TIMEOUT_SEGURIDAD_MS:
            objetivos = [float(d["abierto"]) for d in DEDOS]

        for i, dedo in enumerate(DEDOS):
            if posiciones[i] != objetivos[i]:
                posiciones[i] = acercar(posiciones[i], objetivos[i])
                set_us(dedo["canal"], posiciones[i])

        time.sleep_ms(periodo_ms)


# ──────────────────────────────  Entry  ──────────────────────────────
bucle_principal()
