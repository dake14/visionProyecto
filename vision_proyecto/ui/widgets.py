# ─────────────────────────────  Imports  ─────────────────────────────
from __future__ import annotations

from typing import Callable

import customtkinter as ctk

from vision_proyecto.config import ORDEN_DEDOS
from vision_proyecto.ui import theme


# ──────────────────────────────  Widgets  ────────────────────────────
class PanelFlexiones(ctk.CTkFrame):

    def __init__(self, master, al_mover_slider: Callable[[list[float]], None]):
        super().__init__(master, fg_color=theme.SUPERFICIE, corner_radius=12)
        self._al_mover_slider = al_mover_slider
        self._sliders: list[ctk.CTkSlider] = []
        self._barras: list[ctk.CTkProgressBar] = []
        self._etiquetas_valor: list[ctk.CTkLabel] = []
        self._modo_manual = False

        ctk.CTkLabel(self, text="Flexión por dedo", font=("Segoe UI", 15, "bold"),
                     text_color=theme.TEXTO).grid(row=0, column=0, columnspan=3,
                                                  padx=12, pady=(10, 6), sticky="w")

        for fila, (dedo, color) in enumerate(zip(ORDEN_DEDOS, theme.COLORES_DEDOS), start=1):
            ctk.CTkLabel(self, text=dedo.capitalize(), width=70, anchor="w",
                         text_color=color).grid(row=fila, column=0, padx=(12, 6), pady=4)

            barra = ctk.CTkProgressBar(self, width=160, progress_color=color,
                                       fg_color=theme.FONDO)
            barra.set(0)
            barra.grid(row=fila, column=1, padx=6, pady=4)
            self._barras.append(barra)

            etiqueta = ctk.CTkLabel(self, text="0 %", width=45,
                                    text_color=theme.TEXTO_TENUE)
            etiqueta.grid(row=fila, column=2, padx=(6, 12), pady=4)
            self._etiquetas_valor.append(etiqueta)

            slider = ctk.CTkSlider(self, from_=0, to=1, width=160,
                                   button_color=color, progress_color=color,
                                   command=self._slider_movido)
            slider.set(0)
            self._sliders.append(slider)

    def _slider_movido(self, _valor) -> None:
        if self._modo_manual:
            self._al_mover_slider(self.valores_sliders())

    def valores_sliders(self) -> list[float]:
        return [slider.get() for slider in self._sliders]

    def activar_modo_manual(self, activo: bool) -> None:
        self._modo_manual = activo
        for fila, (slider, barra) in enumerate(zip(self._sliders, self._barras), start=1):
            if activo:
                barra.grid_forget()
                slider.grid(row=fila, column=1, padx=6, pady=4)
            else:
                slider.grid_forget()
                barra.grid(row=fila, column=1, padx=6, pady=4)

    def mostrar_flexiones(self, flexiones: list[float]) -> None:
        for barra, etiqueta, valor in zip(self._barras, self._etiquetas_valor, flexiones):
            barra.set(valor)
            etiqueta.configure(text=f"{round(valor * 100)} %")


class PanelSerial(ctk.CTkFrame):

    def __init__(self, master, al_conectar: Callable[[str], None],
                 al_desconectar: Callable[[], None],
                 al_refrescar: Callable[[], list[str]]):
        super().__init__(master, fg_color=theme.SUPERFICIE, corner_radius=12)
        self._al_conectar = al_conectar
        self._al_desconectar = al_desconectar
        self._al_refrescar = al_refrescar

        ctk.CTkLabel(self, text="Conexión ESP32", font=("Segoe UI", 15, "bold"),
                     text_color=theme.TEXTO).grid(row=0, column=0, columnspan=3,
                                                  padx=12, pady=(10, 6), sticky="w")

        self._combo_puertos = ctk.CTkComboBox(self, values=["(sin puertos)"], width=140)
        self._combo_puertos.grid(row=1, column=0, padx=(12, 6), pady=6)

        ctk.CTkButton(self, text="↻", width=34,
                      command=self.refrescar_puertos).grid(row=1, column=1, padx=4, pady=6)

        self._boton_conexion = ctk.CTkButton(self, text="Conectar", width=100,
                                             command=self._alternar_conexion)
        self._boton_conexion.grid(row=1, column=2, padx=(4, 12), pady=6)

        self._estado = ctk.CTkLabel(self, text="● Desconectado", text_color=theme.ERROR)
        self._estado.grid(row=2, column=0, columnspan=3, padx=12, pady=(0, 10), sticky="w")

        self._conectado = False
        self.refrescar_puertos()

    def refrescar_puertos(self) -> None:
        puertos = self._al_refrescar()
        self._combo_puertos.configure(values=puertos or ["(sin puertos)"])
        if puertos:
            self._combo_puertos.set(puertos[0])
        else:
            self._combo_puertos.set("(sin puertos)")

    def _alternar_conexion(self) -> None:
        if self._conectado:
            self._al_desconectar()
            self.marcar_desconectado()
        else:
            puerto = self._combo_puertos.get()
            if puerto and not puerto.startswith("("):
                self._al_conectar(puerto)

    def marcar_conectado(self, puerto: str) -> None:
        self._conectado = True
        self._boton_conexion.configure(text="Desconectar")
        self._estado.configure(text=f"● Conectado a {puerto}", text_color=theme.OK)

    def marcar_desconectado(self, error: str = "") -> None:
        self._conectado = False
        self._boton_conexion.configure(text="Conectar")
        texto = f"● {error}" if error else "● Desconectado"
        self._estado.configure(text=texto, text_color=theme.ERROR)


class PanelEstado(ctk.CTkFrame):

    def __init__(self, master):
        super().__init__(master, fg_color=theme.SUPERFICIE, corner_radius=12)

        ctk.CTkLabel(self, text="Estado", font=("Segoe UI", 15, "bold"),
                     text_color=theme.TEXTO).grid(row=0, column=0, columnspan=2,
                                                  padx=12, pady=(10, 6), sticky="w")

        self._fps = self._fila(1, "FPS", "0.0")
        self._manos = self._fila(2, "Manos", "0")
        self._gesto = self._fila(3, "Gesto", "—")
        self._fuente = self._fila(4, "Fuente", "MediaPipe")

    def _fila(self, fila: int, titulo: str, valor: str) -> ctk.CTkLabel:
        ctk.CTkLabel(self, text=titulo, width=70, anchor="w",
                     text_color=theme.TEXTO_TENUE).grid(row=fila, column=0,
                                                        padx=(12, 6), pady=3, sticky="w")
        etiqueta = ctk.CTkLabel(self, text=valor, anchor="w", text_color=theme.TEXTO)
        etiqueta.grid(row=fila, column=1, padx=(6, 12), pady=3, sticky="w")
        if fila == 4:
            etiqueta.grid_configure(pady=(3, 10))
        return etiqueta

    def actualizar(self, fps: float, manos: int, gesto: str, fuente: str) -> None:
        self._fps.configure(text=f"{fps:.1f}")
        self._manos.configure(text=str(manos))
        self._gesto.configure(text=gesto)
        self._fuente.configure(text=fuente)
