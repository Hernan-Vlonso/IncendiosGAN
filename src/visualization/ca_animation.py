"""Animación de la propagación del autómata celular."""

from pathlib import Path

import imageio
import numpy as np

from src.ca.cellular_automaton import CellularAutomaton
from src.ca.renderer import render_state


def create_ca_animation(ca: CellularAutomaton, veg_grid: np.ndarray,
                        output_path: str, n_steps: int = 80,
                        fps: int = 10, seed: int = 0):
    """Crear un GIF animado de la propagación del fuego.

    Args:
        ca: autómata celular ya inicializado e ignicionado
        veg_grid: grid de vegetación
        output_path: ruta del GIF de salida
        n_steps: pasos de simulación
        fps: frames por segundo
        seed: semilla
    """
    rng = np.random.default_rng(seed)
    frames = []

    # Frame inicial
    frame = render_state(ca.state, veg_grid, 0, n_steps)
    frames.append(frame)

    for step in range(n_steps):
        ca.step(rng)
        frame = render_state(ca.state, veg_grid, step + 1, n_steps)
        frames.append(frame)

        # Parar si no hay más fuego
        if not np.any(ca.state == 1):
            # Repetir último frame unos cuantos para pausa visual
            for _ in range(5):
                frames.append(frame)
            break

    # Guardar como GIF
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    imageio.mimsave(str(output_path), frames, fps=fps, loop=0)
    print(f"Animación guardada: {output_path} ({len(frames)} frames)")


def create_comparison_animation(snapshots_list: list, veg_grids: list,
                                labels: list, output_path: str,
                                fps: int = 5):
    """Crear animación comparativa de múltiples simulaciones lado a lado.

    Args:
        snapshots_list: lista de listas de snapshots
        veg_grids: lista de grids de vegetación
        labels: etiquetas por simulación
        output_path: ruta de salida
        fps: frames por segundo
    """
    n_sims = len(snapshots_list)
    max_steps = max(len(s) for s in snapshots_list)

    frames = []
    for step in range(max_steps):
        row = []
        for i in range(n_sims):
            snaps = snapshots_list[i]
            idx = min(step, len(snaps) - 1)
            frame = render_state(snaps[idx], veg_grids[i], step, max_steps)
            row.append(frame)

        # Concatenar horizontalmente
        combined = np.concatenate(row, axis=1)
        frames.append(combined)

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    imageio.mimsave(str(output_path), frames, fps=fps, loop=0)
    print(f"Animación comparativa guardada: {output_path}")
