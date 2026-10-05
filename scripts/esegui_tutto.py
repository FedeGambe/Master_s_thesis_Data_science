"""Esegue l'intera analisi in ordine e rigenera notebook e pagina dei risultati.

Uso:  python scripts/esegui_tutto.py [--veloce] [--senza-r]
"""
import argparse
import shutil
import subprocess
import sys
from pathlib import Path

RADICE = Path(__file__).resolve().parents[1]
FASI = ["01_prepara_dati", "02_statistica", "03_esplorativa", "04_cluster", "05_machine_learning", "06_reti_neurali"]


def esegui(comando):
    print(f"\n==> {' '.join(comando)}", flush=True)
    subprocess.run(comando, cwd=RADICE, check=True)


def main(veloce=False, senza_r=False):
    for fase in FASI:
        extra = ["--veloce"] if veloce and fase == "05_machine_learning" else []
        esegui([sys.executable, f"scripts/{fase}.py", *extra])
    if not senza_r and shutil.which("Rscript"):
        esegui(["Rscript", "R/esegui_tutto.R"])
    esegui([sys.executable, "scripts/genera_notebook.py", "--esegui"])
    esegui([sys.executable, "scripts/07_pagina_risultati.py"])
    esegui([sys.executable, "scripts/08_dashboard_html.py"])


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--veloce", action="store_true")
    p.add_argument("--senza-r", action="store_true")
    a = p.parse_args()
    main(a.veloce, a.senza_r)
