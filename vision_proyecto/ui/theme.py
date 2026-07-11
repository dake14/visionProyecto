# ─────────────────────────────  Imports  ─────────────────────────────
from __future__ import annotations

import customtkinter as ctk


# ─────────────────────────────  Paleta  ──────────────────────────────
FONDO = "#1e1e1e"
SUPERFICIE = "#2d2d2d"
TEXTO = "#dcdcdc"
TEXTO_TENUE = "#828282"
ACENTO = "#2a82da"
BORDE = "#555555"
OK = "#3ddc84"
ERROR = "#e05252"

COLORES_DEDOS = ["#ff1493", "#ffa500", "#00ff00", "#4d7cff", "#ffff00"]


# ──────────────────────────────  Tema  ───────────────────────────────
def aplicar_tema_oscuro() -> None:
    ctk.set_appearance_mode("dark")
    ctk.set_default_color_theme("dark-blue")
