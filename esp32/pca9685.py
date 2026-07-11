# ─────────────────────────────  Imports  ─────────────────────────────
import time

import ustruct


# ─────────────────────────────  Config  ──────────────────────────────
REG_MODE1 = 0x00
REG_PRESCALE = 0xFE
REG_LED0_ON_L = 0x06
OSC_INTERNO_HZ = 25000000


# ──────────────────────────────  Driver  ─────────────────────────────
class PCA9685:

    def __init__(self, i2c, address=0x40):
        self.i2c = i2c
        self.address = address
        self.reset()

    def _write(self, reg, value):
        self.i2c.writeto_mem(self.address, reg, bytearray([value]))

    def _read(self, reg):
        return self.i2c.readfrom_mem(self.address, reg, 1)[0]

    def reset(self):
        self._write(REG_MODE1, 0x00)

    def freq(self, freq=None):
        if freq is None:
            return int(OSC_INTERNO_HZ / 4096 / (self._read(REG_PRESCALE) + 1))
        prescale = int(OSC_INTERNO_HZ / 4096.0 / freq + 0.5) - 1
        old_mode = self._read(REG_MODE1)
        self._write(REG_MODE1, (old_mode & 0x7F) | 0x10)
        self._write(REG_PRESCALE, prescale)
        self._write(REG_MODE1, old_mode)
        time.sleep_us(500)
        self._write(REG_MODE1, old_mode | 0xA1)

    def pwm(self, index, on=None, off=None):
        if on is None or off is None:
            data = self.i2c.readfrom_mem(self.address, REG_LED0_ON_L + 4 * index, 4)
            return ustruct.unpack("<HH", data)
        data = ustruct.pack("<HH", on, off)
        self.i2c.writeto_mem(self.address, REG_LED0_ON_L + 4 * index, data)

    def duty(self, index, value):
        if not 0 <= value <= 4095:
            raise ValueError("duty fuera de rango")
        if value == 0:
            self.pwm(index, 0, 4096)
        elif value == 4095:
            self.pwm(index, 4096, 0)
        else:
            self.pwm(index, 0, value)
