#!/usr/bin/env python3
"""Exporta los activos mínimos del demo: pesos del generador (EMA) y casos históricos.

Salida (carpeta demo/):
  generator_run48.pt  state_dict del generador, sin discriminador ni optimizadores
  cases.json          casos históricos curados + parámetros de conversión z-score

Uso:
    python scripts/11_export_demo_assets.py
"""
import json
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.config import get_config

CHECKPOINT = ROOT / "outputs" / "checkpoints" / "run48" / "checkpoint_best_final.pt"
OUT_DIR = ROOT / "demo"

# Los 8 casos de la defensa (un registro por dirección de viento)
DEFENSA = {
    5247: "BATUCO", 2915: "TABUNCO", 2939: "SANTA CRUZ", 2956: "CORONEL DE MAULE",
    2940: "LAS MÁQUINAS", 5249: "QUIVOLGO 1", 2992: "LAS CARDILLAS", 4932: "CORRAL DE PÉREZ",
}
DIRS = ["N", "NE", "E", "SE", "S", "SW", "W", "NW"]
MID_HA_TARGET = 200.0
BAD = "�"


def clean(value):
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return None
    value = str(value).strip()
    return None if BAD in value or not value else value


def case_entry(df, cond, mask, veg, idx, name):
    row = df.loc[idx]
    comuna = clean(row["Comuna"])
    ha = float(row["Sup_Total"])
    label = f"{name}" + (f" ({comuna.title()})" if comuna else "")
    wind_dir = clean(row["Dir_viento"])
    label += f" · {ha:,.0f}".replace(",", ".") + " ha" + (f" · viento {wind_dir}" if wind_dir else "")
    return {
        "label": label,
        "ha": ha,
        "temp": None if mask[idx, 0] else float(row["Temperatura"]),
        "hum": None if mask[idx, 1] else float(row["Humedad_Relativa"]),
        "wind_speed": None if mask[idx, 2] else float(row["Vel_Viento"]),
        "wind_dir": None if mask[idx, 14] else wind_dir,
        "topography": None if mask[idx, 24] else clean(row["Topografia"]),
        "condition": [round(float(x), 6) for x in cond[idx]],
        "mask": [float(x) for x in mask[idx]],
        "veg": [round(float(x), 6) for x in veg[idx]],
    }


def main():
    cfg = get_config()
    df = pd.read_csv(cfg.paths.processed_csv)
    cond = np.load(cfg.paths.condition_vectors)
    mask = np.load(cfg.paths.masks)
    veg = np.load(cfg.paths.vegetation_profiles)
    scaler = pickle.load(open(cfg.paths.processed_data / "scaler.pkl", "rb"))

    cases = [case_entry(df, cond, mask, veg, i, n) for i, n in DEFENSA.items()]

    # Un incendio mediano y completo por dirección, para mostrar el régimen no extremo
    complete = (mask.sum(1) == 0) & df["Nombre incendio"].map(lambda s: clean(s) is not None)
    used = set(DEFENSA)
    for d in DIRS:
        pool = df[complete & (df["Dir_viento"] == d) & ~df.index.isin(used)]
        idx = (pool["Sup_Total"] - MID_HA_TARGET).abs().idxmin()
        used.add(idx)
        cases.append(case_entry(df, cond, mask, veg, idx, clean(df.loc[idx, "Nombre incendio"])))

    imputed = [float(cond[mask[:, i] == 1, i][0]) for i in range(3)]
    ranges = {}
    for key, col in [("temp", "Temperatura"), ("hum", "Humedad_Relativa"), ("wind_speed", "Vel_Viento")]:
        lo, hi = np.nanpercentile(df[col], [0.5, 99.5])
        ranges[key] = [float(np.floor(lo)), float(np.ceil(hi))]

    meta = {
        "scaler_mean": [float(x) for x in scaler.mean_],
        "scaler_scale": [float(x) for x in scaler.scale_],
        "imputed_z": imputed,
        "ranges": ranges,
        "cases": cases,
    }

    OUT_DIR.mkdir(exist_ok=True)
    (OUT_DIR / "cases.json").write_text(json.dumps(meta, ensure_ascii=False, indent=1), encoding="utf-8")

    ckpt = torch.load(CHECKPOINT, map_location="cpu", weights_only=False)
    torch.save(ckpt["ema"] if "ema" in ckpt else ckpt["generator"], OUT_DIR / "generator_run48.pt")
    print(f"Epoca del checkpoint: {ckpt.get('epoch', '?')}")
    print(f"{len(cases)} casos | rangos: {ranges}")
    for f in OUT_DIR.iterdir():
        print(f"  {f.name}: {f.stat().st_size / 1e6:.2f} MB")


if __name__ == "__main__":
    main()
