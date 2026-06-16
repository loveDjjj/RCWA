from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def save_csv(path: str | Path, table: dict[str, np.ndarray]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    names = list(table.keys())
    data = np.column_stack([table[name] for name in names])
    np.savetxt(path, data, delimiter=",", header=",".join(names), comments="")


def plot_spectrum(path: str | Path, table: dict[str, np.ndarray]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    wl = table["wavelength_um"]
    has_te = all(name in table for name in ("R_TE", "T_TE", "A_TE"))
    has_tm = all(name in table for name in ("R_TM", "T_TM", "A_TM"))
    fig, ax = plt.subplots(figsize=(7.2, 4.8))
    if has_te and has_tm:
        r = 0.5 * (table["R_TE"] + table["R_TM"])
        t = 0.5 * (table["T_TE"] + table["T_TM"])
        a = 0.5 * (table["A_TE"] + table["A_TM"])
        title = "FDTD averaged TE/TM, normal incidence"
    elif has_te:
        r, t, a = table["R_TE"], table["T_TE"], table["A_TE"]
        title = "FDTD TE, normal incidence"
    elif has_tm:
        r, t, a = table["R_TM"], table["T_TM"], table["A_TM"]
        title = "FDTD TM, normal incidence"
    else:
        raise ValueError("Spectrum table must contain TE and/or TM columns")
    ax.plot(wl, r, label="R", linewidth=2)
    ax.plot(wl, t, label="T", linewidth=2)
    ax.plot(wl, a, label="A", linewidth=2)
    ax.set_title(title)
    ax.set_xlabel("Wavelength (um)")
    ax.set_ylabel("R / T / A")
    ax.set_ylim(-0.1, 1.1)
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=300)
    plt.close(fig)
