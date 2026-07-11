# ─────────────────────────────  Imports  ─────────────────────────────
from __future__ import annotations

import threading
import time

import serial
from serial.tools import list_ports

from vision_proyecto.config import BAUDIOS, HZ_ENVIO_SERIAL, PREFIJO_TRAMA


# ──────────────────────────────  Core  ───────────────────────────────
def puertos_disponibles() -> list[str]:
    return [puerto.device for puerto in list_ports.comports()]


class ConexionESP32:

    def __init__(self):
        self._serial: serial.Serial | None = None
        self._lock = threading.Lock()
        self._ultimo_envio = 0.0
        self._intervalo_minimo = 1.0 / HZ_ENVIO_SERIAL

    @property
    def conectado(self) -> bool:
        return self._serial is not None and self._serial.is_open

    def conectar(self, puerto: str) -> None:
        self.desconectar()
        self._serial = serial.Serial(puerto, BAUDIOS, timeout=0.1, write_timeout=0.2)
        time.sleep(0.5)

    def desconectar(self) -> None:
        with self._lock:
            if self._serial is not None:
                try:
                    self._serial.close()
                finally:
                    self._serial = None

    def enviar_flexiones(self, flexiones: list[float], forzar: bool = False) -> bool:
        ahora = time.perf_counter()
        if not forzar and ahora - self._ultimo_envio < self._intervalo_minimo:
            return False
        if not self.conectado:
            return False

        porcentajes = [max(0, min(100, round(f * 100))) for f in flexiones]
        trama = PREFIJO_TRAMA + ",".join(str(p) for p in porcentajes) + "\n"
        with self._lock:
            if self._serial is None:
                return False
            try:
                self._serial.write(trama.encode("ascii"))
            except (serial.SerialException, OSError):
                self._serial = None
                return False
        self._ultimo_envio = ahora
        return True
