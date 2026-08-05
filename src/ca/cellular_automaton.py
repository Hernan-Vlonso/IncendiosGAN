"""Motor de simulación de propagación de incendios con autómata celular.

Versión vectorizada con numpy para alto rendimiento.
"""

import numpy as np
from scipy.ndimage import maximum_filter

from src.config import CAConfig


# Estados de las celdas
UNBURNED = 0
BURNING = 1
BURNED_OUT = 2
NON_FLAMMABLE = -1

# Vecindario de Moore (8 direcciones): (dy, dx)
NEIGHBORS = [(-1, -1), (-1, 0), (-1, 1),
             (0, -1),           (0, 1),
             (1, -1),  (1, 0),  (1, 1)]

# Vectores unitarios normalizados para cada dirección vecina
_NEIGHBOR_DIRS = np.array([
    [-1, -1], [-1, 0], [-1, 1],
    [0, -1],           [0, 1],
    [1, -1],  [1, 0],  [1, 1],
], dtype=np.float32)
_NEIGHBOR_NORMS = np.linalg.norm(_NEIGHBOR_DIRS, axis=1, keepdims=True)
_NEIGHBOR_NORMS[_NEIGHBOR_NORMS == 0] = 1.0
_NEIGHBOR_UNIT = _NEIGHBOR_DIRS / _NEIGHBOR_NORMS

# Mapeo de dirección del viento a vector (dy, dx)
# Convención meteorológica estándar: "Dir_viento" = desde dónde viene el viento.
# "N" = viento del Norte → sopla hacia el Sur → fuego se propaga al Sur → dy=+1.
# El vector apunta en la dirección HACIA DONDE sopla el viento (= hacia donde va el fuego).
WIND_VECTORS = {
    "N":  (1,  0),  "NE": (1, -1),  "E": (0, -1), "SE": (-1, -1),
    "S":  (-1, 0),  "SW": (-1, 1),  "W": (0,  1),  "NW": (1,   1),
    "Calma": (0, 0), "Missing": (0, 0),
}


class CellularAutomaton:
    """Autómata celular vectorizado para simulación de propagación de fuego.

    Todas las operaciones de propagación se realizan con numpy sobre el grid
    completo, eliminando los loops de Python celda por celda.
    """

    def __init__(self, veg_grid: np.ndarray, elevation: np.ndarray,
                 slope: np.ndarray, slope_dir: np.ndarray,
                 wind_dir: str, wind_speed: float,
                 humidity: float, temperature: float,
                 cfg: CAConfig):
        # Nota: slope_dir (dirección de pendiente) se acepta por compatibilidad
        # de API pero actualmente no se usa. El efecto de pendiente se calcula
        # directamente desde elevation usando np.roll (diferencias de elevación
        # entre vecinos). slope tampoco se usa: el cálculo de slope_bonus usa
        # elev_diff directamente para mayor precisión direccional.
        # TODO: incorporar slope_dir para penalizar propagación cuesta abajo.
        self.cfg = cfg
        self.size = cfg.grid_size
        self.veg_grid = veg_grid
        self.elevation = elevation

        # Estado del autómata
        self.state = np.zeros((self.size, self.size), dtype=np.int32)
        self.state[veg_grid == 8] = NON_FLAMMABLE

        # Precalcular flammabilidad por celda
        flam = np.array(cfg.flammability, dtype=np.float32)
        self.flammability = flam[np.clip(veg_grid, 0, 9)]

        # Precalcular probabilidad base por celda (no depende de dirección)
        humidity_mod = 1.0 - cfg.humidity_factor * humidity
        temp_bonus = cfg.temperature_factor * temperature * 0.1
        self.base_prob = (cfg.base_spread_prob * self.flammability * humidity_mod
                          + temp_bonus)
        self.base_prob = np.clip(self.base_prob, 0.0, 0.95).astype(np.float32)

        # Precalcular contribución del viento para cada dirección vecina.
        # alignment > 0: dirección a favor del viento → bonus de propagación.
        # alignment < 0: dirección contra el viento → penalización de propagación.
        # Usar alignment sin clamp (no max(0,...)) para capturar ambos efectos.
        wind_vec = np.array(WIND_VECTORS.get(wind_dir, (0, 0)), dtype=np.float32)
        wind_norm = np.linalg.norm(wind_vec)
        self.wind_bonus = np.zeros(8, dtype=np.float32)
        if wind_norm > 0:
            wind_unit = wind_vec / wind_norm
            for i, n_unit in enumerate(_NEIGHBOR_UNIT):
                alignment = np.dot(n_unit, wind_unit)
                self.wind_bonus[i] = cfg.wind_factor * wind_speed * alignment

        # Precalcular diferencias de elevación para cada dirección vecina.
        # np.roll sin wrap-around: se anulan los bordes desplazados para
        # evitar que los bordes opuestos del grid se conecten artificialmente.
        self.elev_diff_grids = []
        for dy, dx in NEIGHBORS:
            shifted = np.roll(np.roll(elevation, -dy, axis=0), -dx, axis=1)
            diff = shifted - elevation
            # Anular bordes envueltos por np.roll(-dy): dy>0 → última fila, dy<0 → primera fila
            if dy > 0:
                diff[-1, :] = 0.0
            elif dy < 0:
                diff[0, :] = 0.0
            if dx > 0:
                diff[:, -1] = 0.0
            elif dx < 0:
                diff[:, 0] = 0.0
            slope_bonus = cfg.slope_factor * np.maximum(0, diff * 5)
            self.elev_diff_grids.append(slope_bonus.astype(np.float32))

        # Precalcular probabilidad de burnout por celda
        self.burnout_prob = (0.15 + 0.1 * (1.0 - self.flammability)).astype(np.float32)

        # Precalcular spread probability total para cada dirección
        # spread_prob[i] = probabilidad de propagación hacia vecino i
        self.spread_probs = []
        for i in range(8):
            prob = self.base_prob + self.wind_bonus[i] + self.elev_diff_grids[i]
            prob = np.clip(prob, 0.0, 0.95)
            self.spread_probs.append(prob)

        self.history = []
        self.step_count = 0

    def ignite(self, center_y: int = None, center_x: int = None,
               radius: int = None):
        """Iniciar fuego en una zona central."""
        if center_y is None:
            center_y = self.size // 2
        if center_x is None:
            center_x = self.size // 2
        if radius is None:
            radius = self.cfg.ignition_radius

        y_lo = max(0, center_y - radius)
        y_hi = min(self.size, center_y + radius + 1)
        x_lo = max(0, center_x - radius)
        x_hi = min(self.size, center_x + radius + 1)

        patch = self.state[y_lo:y_hi, x_lo:x_hi]
        patch[patch == UNBURNED] = BURNING

        self.history.append(self.state.copy())

    def step(self, rng: np.random.Generator):
        """Ejecutar un paso de simulación (vectorizado)."""
        size = self.size
        burning = (self.state == BURNING)

        if not burning.any():
            self.step_count += 1
            return self.state

        unburned = (self.state == UNBURNED)

        # Para cada dirección vecina, calcular qué celdas unburned tienen
        # un vecino burning Y el random < spread_prob
        new_burning = np.zeros((size, size), dtype=bool)

        for i, (dy, dx) in enumerate(NEIGHBORS):
            # Shift burning mask: si burning[y,x] entonces shifted[y+dy, x+dx] = True.
            # Anular bordes envueltos por np.roll(dy): dy>0 → primera fila, dy<0 → última fila.
            shifted_burning = np.roll(np.roll(burning, dy, axis=0), dx, axis=1)
            if dy > 0:
                shifted_burning[0, :] = False
            elif dy < 0:
                shifted_burning[-1, :] = False
            if dx > 0:
                shifted_burning[:, 0] = False
            elif dx < 0:
                shifted_burning[:, -1] = False

            # Celdas candidatas: unburned Y tiene vecino burning en dirección i
            candidates = unburned & shifted_burning

            if not candidates.any():
                continue

            # Test probabilístico solo en candidatos
            rand = rng.random((size, size)).astype(np.float32)
            ignited = candidates & (rand < self.spread_probs[i])
            new_burning |= ignited

        # Burnout: celdas burning → burned_out
        rand_burnout = rng.random((size, size)).astype(np.float32)
        burned_out = burning & (rand_burnout < self.burnout_prob)

        # Actualizar estado
        self.state[new_burning] = BURNING
        self.state[burned_out] = BURNED_OUT

        self.step_count += 1
        return self.state

    def run(self, n_steps: int = None, seed: int = 0) -> list:
        """Ejecutar simulación completa.

        Returns:
            Lista de snapshots del estado en intervalos regulares.
        """
        if n_steps is None:
            n_steps = self.cfg.n_steps

        rng = np.random.default_rng(seed)
        n_snapshots = self.cfg.n_snapshots
        snapshot_interval = max(1, n_steps // n_snapshots)

        snapshots = []
        for step_i in range(n_steps):
            self.step(rng)

            if not np.any(self.state == BURNING):
                while len(snapshots) < n_snapshots:
                    snapshots.append(self.state.copy())
                break

            if (step_i + 1) % snapshot_interval == 0 and len(snapshots) < n_snapshots:
                snapshots.append(self.state.copy())

        while len(snapshots) < n_snapshots:
            snapshots.append(self.state.copy())

        return snapshots


def run_simulation(veg_grid, elevation, slope, slope_dir,
                   wind_dir, wind_speed, humidity, temperature,
                   cfg: CAConfig, seed: int = 0):
    """Ejecutar una simulación completa y retornar snapshots."""
    ca = CellularAutomaton(
        veg_grid=veg_grid, elevation=elevation,
        slope=slope, slope_dir=slope_dir,
        wind_dir=wind_dir, wind_speed=wind_speed,
        humidity=humidity, temperature=temperature,
        cfg=cfg,
    )
    ca.ignite()
    snapshots = ca.run(seed=seed)
    return snapshots, ca
