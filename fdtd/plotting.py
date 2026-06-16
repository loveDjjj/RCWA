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
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6), sharey=True)
    for ax, pol in zip(axes, ["TE", "TM"]):
        ax.plot(wl, table[f"R_{pol}"], label=f"{pol} R", linewidth=2)
        ax.plot(wl, table[f"T_{pol}"], label=f"{pol} T", linewidth=2)
        ax.plot(wl, table[f"A_{pol}"], label=f"{pol} A", linewidth=2)
        ax.set_title(f"FDTD {pol}, normal incidence")
        ax.set_xlabel("Wavelength (um)")
        ax.set_ylabel("R / T / A")
        ax.set_ylim(-0.1, 1.1)
        ax.grid(True, alpha=0.3)
        ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=300)
    plt.close(fig)

