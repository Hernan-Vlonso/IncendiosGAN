#!/usr/bin/env python3
"""Diagnóstico 3: Condiciones meteorológicas de registros E vs otras direcciones.

Verifica si los registros con viento E tienen condiciones que anulan la señal
direccional (viento débil, humedad alta, etc.).
"""

import sys
import pickle
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.config import get_config
from src.data.preprocessing import DIR_VIENTO_CATS

_DIR_VIENTO_START = 14
_DIRS = ["N", "NE", "E", "SE", "S", "SW", "W", "NW"]
_CONT_NAMES = ["Temperatura", "Humedad_Relativa", "Vel_Viento", "Sup_Total"]


def main():
    cfg = get_config()
    condition_vectors = np.load(cfg.paths.condition_vectors)

    with open(cfg.paths.scaler_path, "rb") as f:
        scaler = pickle.load(f)

    # Separar registros por dirección de viento
    wind_onehot = condition_vectors[:, _DIR_VIENTO_START:_DIR_VIENTO_START + 10]
    dir_idx = np.argmax(wind_onehot, axis=1)

    print(f"\n{'Dir':>4} {'N regs':>8} {'Temp(C)':>10} {'Hum(%)':>10} {'Vel(km/h)':>12} {'Area(ha)':>10}")
    print("-" * 60)

    for d, name in enumerate(_DIRS):
        mask = dir_idx == d
        n = mask.sum()
        if n == 0:
            print(f"  {name:>4} {0:>8}")
            continue

        z = condition_vectors[mask, :4].astype(np.float64)
        physical = z * scaler.scale_[:4] + scaler.mean_[:4]

        temp   = physical[:, 0]
        hum    = physical[:, 1]
        vel    = physical[:, 2]
        area   = np.expm1(physical[:, 3])  # invertir log1p

        print(f"  {name:>4} {n:>8} "
              f"{temp.mean():>7.1f}±{temp.std():4.1f} "
              f"{hum.mean():>7.1f}±{hum.std():4.1f} "
              f"{vel.mean():>8.1f}±{vel.std():5.1f} "
              f"{np.median(area):>8.1f}ha")

    # Cuántos registros E tienen vel_viento < 5 km/h (señal débil)
    e_mask = dir_idx == 2
    e_z = condition_vectors[e_mask, :4].astype(np.float64)
    e_phys = e_z * scaler.scale_[:4] + scaler.mean_[:4]
    e_vel = e_phys[:, 1+1]  # Vel_Viento es índice 2
    low_wind = (e_vel < 5).sum()
    print(f"\n  Registros E con vel < 5 km/h: {low_wind}/{e_mask.sum()} ({100*low_wind/e_mask.sum():.1f}%)")

    # Dirección E vs S: comparar vel_viento directamente
    s_mask = dir_idx == 4
    s_z = condition_vectors[s_mask, :4].astype(np.float64)
    s_phys = s_z * scaler.scale_[:4] + scaler.mean_[:4]
    s_vel = s_phys[:, 2]
    e_vel2 = e_phys[:, 2]
    print(f"\n  Vel_Viento media  E: {e_vel2.mean():.2f} km/h  vs  S: {s_vel.mean():.2f} km/h")
    print(f"  Vel_Viento mediana E: {np.median(e_vel2):.2f} km/h  vs  S: {np.median(s_vel):.2f} km/h")


if __name__ == "__main__":
    main()
