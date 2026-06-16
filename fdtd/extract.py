from __future__ import annotations

import numpy as np


UM = 1e-6
C0 = 3e8


def as_1d(value) -> np.ndarray:
    return np.asarray(value, dtype=np.float64).reshape(-1)


def transmission(fdtd, monitor: str) -> tuple[np.ndarray, np.ndarray]:
    spec = as_1d(fdtd.transmission(monitor))
    freq = as_1d(fdtd.getdata(monitor, "f"))
    wavelength_um = C0 / (freq * UM)
    return wavelength_um, spec


def extract_rt(fdtd, pol: str) -> dict[str, np.ndarray]:
    wavelength_um, r_raw = transmission(fdtd, "R")
    _, t_raw = transmission(fdtd, "T")
    r = r_raw
    t = -t_raw if np.nanmean(t_raw) < 0 else t_raw
    a = 1.0 - r - t
    order = np.argsort(wavelength_um)
    return {
        "wavelength_um": wavelength_um[order],
        f"R_{pol}": r[order],
        f"T_{pol}": t[order],
        f"A_{pol}": a[order],
        f"R_raw_{pol}": r_raw[order],
        f"T_raw_{pol}": t_raw[order],
    }

