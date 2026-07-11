# ─────────────────────────────  Imports  ─────────────────────────────
from __future__ import annotations

import threading
import time

import cv2
import customtkinter as ctk
from PIL import Image

from vision_proyecto.config import (
    ALTO_CAMARA,
    ANCHO_CAMARA,
    POSTURAS_GESTO,
)
from vision_proyecto.core.dibujo import dibujar_esqueleto, dibujar_hud
from vision_proyecto.core.flexion import (
    SuavizadorFlexiones,
    calcular_flexiones,
    clasificar_gesto,
)
from vision_proyecto.core.serial_esp32 import ConexionESP32, puertos_disponibles
from vision_proyecto.core.tracking import RastreadorManos
from vision_proyecto.ui import theme
from vision_proyecto.ui.widgets import PanelEstado, PanelFlexiones, PanelSerial
from vision_proyecto.utils import asegurar_modelo_landmarker


# ─────────────────────────────  Config  ──────────────────────────────
MODO_MANUAL = "Manual"
MODO_VISION = "Visión"
MODO_CNN = "CNN propia"

TAM_VIDEO = (860, 484)
INTERVALO_UI_MS = 30
FRAMES_ENTRE_PREDICCIONES_CNN = 3
INDICES_CAMARA_FALLBACK = 4
SEGUNDOS_REINTENTO_CAMARA = 2.0


# ──────────────────────────  Estado compartido  ───────────────────────
class EstadoCompartido:

    def __init__(self):
        self.lock = threading.Lock()
        self.frame = None
        self.flexiones = [0.0] * 5
        self.gesto = "—"
        self.fps = 0.0
        self.manos = 0
        self.fuente = "MediaPipe"
        self.mensaje_video = "Iniciando cámara..."


# ──────────────────────────────  App  ─────────────────────────────────
class AppManoRobotica(ctk.CTk):

    def __init__(self, indice_camara: int = 0):
        super().__init__(fg_color=theme.FONDO)
        self.title("visionProyecto — Mano robótica por visión artificial")
        self.geometry("1280x760")
        self.minsize(1100, 640)

        self._estado = EstadoCompartido()
        self._conexion = ConexionESP32()
        self._modo = MODO_VISION
        self._indice_camara = indice_camara
        self._detener = threading.Event()
        self._clasificador_cnn = None

        self._construir_ui()
        self._hilo_video = threading.Thread(target=self._bucle_video, daemon=True)
        self._hilo_video.start()

        self.protocol("WM_DELETE_WINDOW", self._al_cerrar)
        self.after(INTERVALO_UI_MS, self._refrescar_ui)

    # ────────────────────────  Construcción UI  ───────────────────────
    def _construir_ui(self) -> None:
        self.grid_columnconfigure(0, weight=1)
        self.grid_rowconfigure(0, weight=1)

        self._etiqueta_video = ctk.CTkLabel(self, text="Iniciando cámara...",
                                            fg_color=theme.SUPERFICIE,
                                            corner_radius=12,
                                            text_color=theme.TEXTO_TENUE)
        self._etiqueta_video.grid(row=0, column=0, rowspan=2, padx=(16, 8),
                                  pady=16, sticky="nsew")

        columna = ctk.CTkFrame(self, fg_color="transparent")
        columna.grid(row=0, column=1, rowspan=2, padx=(8, 16), pady=16, sticky="ns")

        ctk.CTkLabel(columna, text="Modo de control", font=("Segoe UI", 15, "bold"),
                     text_color=theme.TEXTO).pack(anchor="w", pady=(0, 6))

        self._selector_modo = ctk.CTkSegmentedButton(
            columna,
            values=[MODO_MANUAL, MODO_VISION, MODO_CNN],
            command=self._cambiar_modo,
        )
        self._selector_modo.set(MODO_VISION)
        self._selector_modo.pack(fill="x", pady=(0, 12))

        self._panel_flexiones = PanelFlexiones(columna, self._sliders_movidos)
        self._panel_flexiones.pack(fill="x", pady=(0, 12))

        self._panel_serial = PanelSerial(columna, self._conectar_serial,
                                         self._desconectar_serial,
                                         puertos_disponibles)
        self._panel_serial.pack(fill="x", pady=(0, 12))

        self._panel_estado = PanelEstado(columna)
        self._panel_estado.pack(fill="x", pady=(0, 12))

        self._mensaje = ctk.CTkLabel(columna, text="", wraplength=280,
                                     text_color=theme.TEXTO_TENUE, justify="left")
        self._mensaje.pack(fill="x")

    # ─────────────────────────  Callbacks UI  ─────────────────────────
    def _cambiar_modo(self, modo: str) -> None:
        if modo == MODO_CNN and self._clasificador_cnn is None:
            if not self._cargar_cnn():
                self._selector_modo.set(self._modo)
                return
        self._modo = modo
        self._panel_flexiones.activar_modo_manual(modo == MODO_MANUAL)
        self._mensaje.configure(text="")

    def _cargar_cnn(self) -> bool:
        from vision_proyecto.core.inferencia_cnn import ClasificadorGestosCNN

        if not ClasificadorGestosCNN.disponible():
            self._mensaje.configure(
                text="No existe modelos/cnn_gestos.keras. "
                     "Entrena primero: python modelo_scratch/entrenar.py",
                text_color=theme.ERROR)
            return False
        self._mensaje.configure(text="Cargando modelo CNN...", text_color=theme.TEXTO_TENUE)
        self.update_idletasks()
        try:
            self._clasificador_cnn = ClasificadorGestosCNN()
        except Exception as error:
            self._mensaje.configure(text=f"Error al cargar CNN: {error}",
                                    text_color=theme.ERROR)
            return False
        self._mensaje.configure(text="")
        return True

    def _sliders_movidos(self, valores: list[float]) -> None:
        with self._estado.lock:
            self._estado.flexiones = valores
        self._conexion.enviar_flexiones(valores)

    def _conectar_serial(self, puerto: str) -> None:
        try:
            self._conexion.conectar(puerto)
            self._panel_serial.marcar_conectado(puerto)
        except Exception as error:
            self._panel_serial.marcar_desconectado(f"Error: {error}")

    def _desconectar_serial(self) -> None:
        self._conexion.desconectar()

    # ─────────────────────────  Hilo de video  ────────────────────────
    def _abrir_camara(self):
        indices = [self._indice_camara] + [
            i for i in range(INDICES_CAMARA_FALLBACK) if i != self._indice_camara
        ]
        for indice in indices:
            captura = cv2.VideoCapture(indice)
            if captura.isOpened():
                captura.set(cv2.CAP_PROP_FRAME_WIDTH, ANCHO_CAMARA)
                captura.set(cv2.CAP_PROP_FRAME_HEIGHT, ALTO_CAMARA)
                captura.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                return captura, indice
            captura.release()
        return None, -1

    def _bucle_video(self) -> None:
        ruta_modelo = asegurar_modelo_landmarker()
        rastreador = RastreadorManos(ruta_modelo)
        suavizador = SuavizadorFlexiones()

        captura, indice = self._abrir_camara()
        while captura is None and not self._detener.is_set():
            with self._estado.lock:
                self._estado.mensaje_video = (
                    "No se encontró ninguna cámara (índices 0–3).\n"
                    "Conecta una webcam; se reintenta automáticamente...")
            if self._detener.wait(SEGUNDOS_REINTENTO_CAMARA):
                break
            captura, indice = self._abrir_camara()

        if captura is None:
            rastreador.cerrar()
            return

        with self._estado.lock:
            self._estado.mensaje_video = ""

        contador_fps = 0
        fps_actual = 0.0
        t_referencia = time.perf_counter()
        contador_frames = 0

        while not self._detener.is_set():
            ok, frame = captura.read()
            if not ok:
                time.sleep(0.05)
                continue

            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            rastreador.procesar_frame(frame_rgb)
            resultado = rastreador.ultimo_resultado()

            flexiones = None
            gesto = "—"
            manos = 0
            fuente = "MediaPipe"

            if resultado and resultado.hand_landmarks:
                manos = len(resultado.hand_landmarks)
                landmarks = resultado.hand_landmarks[0]
                lateralidad = "Izquierda"
                if resultado.handedness:
                    categoria = resultado.handedness[0][0]
                    lateralidad = "Derecha" if categoria.category_name == "Right" else "Izquierda"
                dibujar_esqueleto(frame, landmarks, lateralidad)
                flexiones = suavizador.actualizar(calcular_flexiones(landmarks))
                gesto = clasificar_gesto(flexiones)

            if self._modo == MODO_CNN and self._clasificador_cnn is not None:
                fuente = "CNN propia"
                contador_frames += 1
                if contador_frames % FRAMES_ENTRE_PREDICCIONES_CNN == 0:
                    gesto_cnn, confianza = self._clasificador_cnn.predecir(frame_rgb)
                    gesto = f"{gesto_cnn} ({confianza:.0%})"
                    flexiones = suavizador.actualizar(POSTURAS_GESTO[gesto_cnn])
                else:
                    flexiones = None

            dibujar_hud(frame, fps_actual, gesto, self._modo)

            with self._estado.lock:
                self._estado.frame = frame
                self._estado.fps = fps_actual
                self._estado.manos = manos
                self._estado.gesto = gesto
                self._estado.fuente = fuente
                if flexiones is not None and self._modo != MODO_MANUAL:
                    self._estado.flexiones = flexiones

            if flexiones is not None and self._modo != MODO_MANUAL:
                self._conexion.enviar_flexiones(flexiones)

            contador_fps += 1
            t_ahora = time.perf_counter()
            if t_ahora - t_referencia >= 1.0:
                fps_actual = contador_fps / (t_ahora - t_referencia)
                contador_fps = 0
                t_referencia = t_ahora

        captura.release()
        rastreador.cerrar()

    # ─────────────────────────  Refresco de UI  ───────────────────────
    def _refrescar_ui(self) -> None:
        with self._estado.lock:
            frame = self._estado.frame
            flexiones = list(self._estado.flexiones)
            fps = self._estado.fps
            manos = self._estado.manos
            gesto = self._estado.gesto
            fuente = self._estado.fuente
            mensaje_video = self._estado.mensaje_video

        if frame is None and mensaje_video:
            self._etiqueta_video.configure(text=mensaje_video)

        if frame is not None:
            imagen = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            imagen_ctk = ctk.CTkImage(light_image=imagen, dark_image=imagen,
                                      size=TAM_VIDEO)
            self._etiqueta_video.configure(image=imagen_ctk, text="")
            self._etiqueta_video._image_ref = imagen_ctk

        if self._modo != MODO_MANUAL:
            self._panel_flexiones.mostrar_flexiones(flexiones)
        self._panel_estado.actualizar(fps, manos, gesto, fuente)

        if not self._detener.is_set():
            self.after(INTERVALO_UI_MS, self._refrescar_ui)

    # ────────────────────────────  Cierre  ────────────────────────────
    def _al_cerrar(self) -> None:
        self._detener.set()
        self._hilo_video.join(timeout=2.0)
        self._conexion.desconectar()
        self.destroy()
