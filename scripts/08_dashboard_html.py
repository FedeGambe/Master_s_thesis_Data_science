"""Fase 8: dashboard di predizione in HTML statico.

Esporta il modello finale (``modelli/modello_finale.joblib``) e scrive ``docs/dashboard.html``:
un unico file che si apre con un doppio clic, senza Python né server.
La corrispondenza con il modello Python è verificata da ``tests/test_dashboard_html.py``.
"""
import _percorso  # noqa: F401

import json

from tesi_bev import archivio, config as C, esporta_web


def main():
    modello = archivio.carica_modello("modello_finale")
    meta = json.loads((C.DIR_MODELLI / "modello_finale.json").read_text(encoding="utf-8"))
    percorso = esporta_web.scrivi_pagina(modello, meta)
    print(f"Dashboard scritta in {percorso.relative_to(C.RADICE)} ({percorso.stat().st_size / 1024:.0f} KB)")


if __name__ == "__main__":
    main()
