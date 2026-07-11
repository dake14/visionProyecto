# ─────────────────────────────  Imports  ─────────────────────────────
from __future__ import annotations

import argparse

from vision_proyecto.ui.app import AppManoRobotica
from vision_proyecto.ui.theme import aplicar_tema_oscuro


# ──────────────────────────────  Entry  ──────────────────────────────
def main() -> None:
    parser = argparse.ArgumentParser(description="Mano robótica por visión artificial")
    parser.add_argument("--camara", type=int, default=0)
    args = parser.parse_args()

    aplicar_tema_oscuro()
    app = AppManoRobotica(indice_camara=args.camara)
    app.mainloop()


if __name__ == "__main__":
    main()
