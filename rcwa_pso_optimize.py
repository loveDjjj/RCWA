"""
Particle-swarm optimization for normal-incidence RCWA reflection spectra.

The script optimizes a configured square-pillar metasurface using the local
batched torch_rcwa implementation. It evaluates particle x wavelength batches
at normal incidence and writes each generation's best spectrum and target plot.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import yaml
from tqdm.auto import tqdm

import torch_rcwa
from rcwa_spectrum import (
    channel_power_batch,
    check_rayleigh_points,
    diffraction_orders,
    iter_slices,
    load_material_eps,
    propagating_order_mask_tensor,
    save_table,
    wavelength_axis,
)


DEFAULT_CONFIG = "configs/pso.yaml"


def parse_args():
    parser = argparse.ArgumentParser(description="PSO optimization for normal-incidence RCWA reflection spectra.")
    parser.add_argument("--config", default=DEFAULT_CONFIG, help="YAML configuration file.")
    return parser.parse_args()


def load_config(path: str | Path) -> dict:
    with open(path, "r", encoding="utf-8") as handle:
        cfg = yaml.safe_load(handle)
    if not isinstance(cfg, dict):
        raise ValueError(f"Invalid YAML config: {path}")
    return cfg


def build_pillar_masks(
    side_um: torch.Tensor,
    geometry: dict,
    rcwa_cfg: dict,
    geo_dtype: torch.dtype,
    device: torch.device,
) -> torch.Tensor:
    nx, ny = [int(v) for v in rcwa_cfg["grid"]]
    period_x, period_y = [float(v) for v in geometry["period_um"]]
    edge_sharpness = float(geometry.get("edge_sharpness", 1000.0))

    x = (period_x / nx) * (torch.arange(nx, dtype=geo_dtype, device=device) + 0.5)
    y = (period_y / ny) * (torch.arange(ny, dtype=geo_dtype, device=device) + 0.5)
    x_grid, y_grid = torch.meshgrid(x, y, indexing="ij")
    side = side_um[:, None, None]
    level = 1.0 - torch.maximum(
        torch.abs((x_grid[None, :, :] - period_x / 2.0) / (side / 2.0)),
        torch.abs((y_grid[None, :, :] - period_y / 2.0) / (side / 2.0)),
    )
    return torch.sigmoid(edge_sharpness * level)


def prepare_air_conv(order: int, geometry: dict, rcwa_cfg: dict, sim_dtype: torch.dtype, geo_dtype: torch.dtype, device: torch.device):
    nx, ny = [int(v) for v in rcwa_cfg["grid"]]
    builder = torch_rcwa.rcwa(
        freq=torch.ones(1, dtype=geo_dtype, device=device),
        order=[order, order],
        L=[float(v) for v in geometry["period_um"]],
        dtype=sim_dtype,
        device=device,
        linalg_batch_mode=rcwa_cfg.get("linalg_batch_mode", "auto"),
        linalg_batch_threshold=int(rcwa_cfg.get("linalg_batch_threshold", 512)),
    )
    return builder.convolution_matrix(torch.ones((nx, ny), dtype=sim_dtype, device=device))[0]


def target_reflection(cfg: dict, wavelengths_um: np.ndarray) -> np.ndarray:
    target = cfg["target"]
    target_type = target.get("type", "constant")
    if target_type == "constant":
        return np.full(len(wavelengths_um), float(target.get("R", 0.0)), dtype=np.float64)
    if target_type == "csv":
        data = np.genfromtxt(target["csv"], delimiter=",", names=True)
        wl_col = target.get("wavelength_column", "wavelength_um")
        r_col = target.get("reflection_column", "R")
        return np.interp(wavelengths_um, data[wl_col], data[r_col]).astype(np.float64)
    raise ValueError("target.type must be constant or csv")


def load_material_candidates(cfg: dict, wavelengths_um: np.ndarray):
    unit = cfg["materials"].get("csv_wavelength_unit", "um")
    structure = cfg["materials"]["structure_candidates"]
    film = cfg["materials"]["film_candidates"]
    structure_eps = np.stack([load_material_eps(item["csv"], wavelengths_um, unit) for item in structure])
    film_eps = np.stack([load_material_eps(item["csv"], wavelengths_um, unit) for item in film])
    substrate_eps = load_material_eps(cfg["materials"]["substrate_csv"], wavelengths_um, unit)
    return structure, film, structure_eps, film_eps, substrate_eps


def decode_particles(position: np.ndarray, cfg: dict) -> dict[str, np.ndarray]:
    n_structure = len(cfg["materials"]["structure_candidates"])
    n_film = len(cfg["materials"]["film_candidates"])
    period_x = float(cfg["geometry"]["period_um"][0])
    fill_factor = position[:, 0]
    decoded = {
        "period_um": np.full(len(position), period_x, dtype=np.float64),
        "fill_factor": fill_factor,
        "pillar_side_um": fill_factor * period_x,
        "pillar_thickness_um": position[:, 1],
        "film_thickness_um": position[:, 2],
        "structure_material_idx": np.clip(np.floor(position[:, 3] + 0.5).astype(np.int64), 0, n_structure - 1),
        "film_material_idx": np.clip(np.floor(position[:, 4] + 0.5).astype(np.int64), 0, n_film - 1),
    }
    return decoded


def particle_bounds(cfg: dict) -> tuple[np.ndarray, np.ndarray]:
    bounds = cfg["bounds"]
    n_structure = len(cfg["materials"]["structure_candidates"])
    n_film = len(cfg["materials"]["film_candidates"])
    lower = np.array(
        [
            bounds["fill_factor"][0],
            bounds["pillar_thickness_um"][0],
            bounds["film_thickness_um"][0],
            -0.5,
            -0.5,
        ],
        dtype=np.float64,
    )
    upper = np.array(
        [
            bounds["fill_factor"][1],
            bounds["pillar_thickness_um"][1],
            bounds["film_thickness_um"][1],
            max(n_structure - 1, 0) + 0.5,
            max(n_film - 1, 0) + 0.5,
        ],
        dtype=np.float64,
    )
    return lower, upper


def evaluate_particles(
    position: np.ndarray,
    wavelengths_um: np.ndarray,
    target_r: np.ndarray,
    cfg: dict,
    material_data: tuple,
    device: torch.device,
    sim_dtype: torch.dtype,
    geo_dtype: torch.dtype,
    return_spectra: bool = False,
    progress_desc: str | None = None,
    show_progress: bool = True,
) -> tuple[np.ndarray, np.ndarray | None]:
    structure_items, film_items, structure_eps, film_eps, substrate_eps = material_data
    rcwa_cfg = cfg["rcwa"]
    order = int(rcwa_cfg["fourier_order"])
    geometry = cfg["geometry"]
    period = [float(v) for v in geometry["period_um"]]
    decoded = decode_particles(position, cfg)
    particle_count = len(position)
    wavelength_count = len(wavelengths_um)
    batch_size = int(rcwa_cfg.get("batch_size", particle_count * wavelength_count))

    wl_t_all = torch.as_tensor(wavelengths_um, dtype=geo_dtype, device=device)
    target_t = torch.as_tensor(target_r, dtype=geo_dtype, device=device)
    substrate_eps_t_all = torch.as_tensor(substrate_eps, dtype=sim_dtype, device=device)
    air_conv = prepare_air_conv(order, geometry, rcwa_cfg, sim_dtype, geo_dtype, device)
    orders_np = diffraction_orders(order)
    orders_t = torch.as_tensor(orders_np, dtype=torch.int64, device=device)

    side_t = torch.as_tensor(decoded["pillar_side_um"], dtype=geo_dtype, device=device)
    masks = build_pillar_masks(side_t, geometry, rcwa_cfg, geo_dtype, device).to(sim_dtype)
    conv_builder = torch_rcwa.rcwa(
        freq=torch.ones(particle_count, dtype=geo_dtype, device=device),
        order=[order, order],
        L=period,
        dtype=sim_dtype,
        device=device,
        linalg_batch_mode=rcwa_cfg.get("linalg_batch_mode", "auto"),
        linalg_batch_threshold=int(rcwa_cfg.get("linalg_batch_threshold", 512)),
    )
    pillar_shape_conv = conv_builder.convolution_matrix(masks)

    structure_eps_selected = structure_eps[decoded["structure_material_idx"]]
    film_eps_selected = film_eps[decoded["film_material_idx"]]
    structure_eps_t = torch.as_tensor(structure_eps_selected, dtype=sim_dtype, device=device)
    film_eps_t = torch.as_tensor(film_eps_selected, dtype=sim_dtype, device=device)
    pillar_thickness_t = torch.as_tensor(decoded["pillar_thickness_um"], dtype=sim_dtype, device=device)
    film_thickness_t = torch.as_tensor(decoded["film_thickness_um"], dtype=sim_dtype, device=device)

    r_all = torch.empty((particle_count, wavelength_count), dtype=geo_dtype, device=device)
    invalid = np.zeros(particle_count, dtype=bool)
    penalty_loss = float(cfg.get("target", {}).get("penalty_loss", 1.0e6))
    total = particle_count * wavelength_count
    batch_count = (total + batch_size - 1) // batch_size
    flat_slices = iter_slices(total, batch_size)
    if show_progress:
        flat_slices = tqdm(
            flat_slices,
            total=batch_count,
            desc=progress_desc or "RCWA batches",
            leave=False,
        )
    for flat_slice in flat_slices:
        flat = torch.arange(flat_slice.start, flat_slice.stop, dtype=torch.int64, device=device)
        particle_idx = flat // wavelength_count
        wavelength_idx = flat % wavelength_count

        wl_t = wl_t_all[wavelength_idx]
        freq = 1.0 / wl_t
        eps_structure = structure_eps_t[particle_idx, wavelength_idx]
        eps_film = film_eps_t[particle_idx, wavelength_idx]
        eps_substrate = substrate_eps_t_all[wavelength_idx]
        eps_conv = air_conv[None, :, :] + (eps_structure - 1.0)[:, None, None] * pillar_shape_conv[particle_idx]

        try:
            sim = torch_rcwa.rcwa(
                freq=freq,
                order=[order, order],
                L=period,
                dtype=sim_dtype,
                device=device,
                linalg_batch_mode=rcwa_cfg.get("linalg_batch_mode", "auto"),
                linalg_batch_threshold=int(rcwa_cfg.get("linalg_batch_threshold", 512)),
            )
            sim.add_input_layer(eps=1.0)
            sim.add_output_layer(eps=eps_substrate)
            sim.set_incident_angle(
                inc_ang=torch.zeros(freq.numel(), dtype=geo_dtype, device=device),
                azi_ang=torch.zeros(freq.numel(), dtype=geo_dtype, device=device),
            )
            sim.add_layer_conv(thickness=pillar_thickness_t[particle_idx], eps_conv=eps_conv)
            sim.add_layer(thickness=film_thickness_t[particle_idx], eps=eps_film)
            sim.solve_global_smatrix()

            mask = propagating_order_mask_tensor(
                freq,
                torch.ones_like(eps_structure),
                torch.zeros_like(freq.real),
                0.0,
                orders_t,
                geometry,
            )
            r_te = channel_power_batch(sim, "reflection", ("ss", "ps"), orders_t, mask)
            r_tm = channel_power_batch(sim, "reflection", ("pp", "sp"), orders_t, mask)
            r = 0.5 * (r_te + r_tm)
            if not torch.isfinite(r).all():
                raise FloatingPointError("non-finite RCWA reflection")
            r_all[particle_idx, wavelength_idx] = r
        except Exception as exc:
            failed_particles = torch.unique(particle_idx).detach().cpu().numpy().astype(np.int64)
            invalid[failed_particles] = True
            r_all[particle_idx, wavelength_idx] = penalty_loss
            tqdm.write(
                "RCWA batch failed; penalizing particles "
                f"{failed_particles.tolist()}: {type(exc).__name__}: {exc}"
            )

    loss = torch.mean((r_all - target_t[None, :]) ** 2, dim=1)
    if np.any(invalid):
        invalid_t = torch.as_tensor(invalid, dtype=torch.bool, device=device)
        loss = torch.where(invalid_t, torch.full_like(loss, penalty_loss), loss)
    spectra = r_all.detach().cpu().numpy() if return_spectra else None
    return loss.detach().cpu().numpy(), spectra


def write_json(path: str | Path, payload: dict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def plot_generation(path: str | Path, wavelengths_um: np.ndarray, reflection: np.ndarray, target_r: np.ndarray, title: str) -> None:
    import matplotlib.pyplot as plt

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    ax.plot(wavelengths_um, reflection, label="Best structure R", linewidth=2)
    ax.plot(wavelengths_um, target_r, label="Target R", linewidth=2, linestyle="--")
    ax.set_xlabel("Wavelength (um)")
    ax.set_ylabel("Reflection")
    ax.set_ylim(-0.05, 1.05)
    ax.grid(True, alpha=0.3)
    ax.set_title(title)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=220)
    plt.close(fig)


def structure_payload(position: np.ndarray, loss: float, cfg: dict) -> dict:
    decoded = decode_particles(position[None, :], cfg)
    structure_items = cfg["materials"]["structure_candidates"]
    film_items = cfg["materials"]["film_candidates"]
    s_idx = int(decoded["structure_material_idx"][0])
    f_idx = int(decoded["film_material_idx"][0])
    return {
        "loss": float(loss),
        "period_um": float(decoded["period_um"][0]),
        "fill_factor": float(decoded["fill_factor"][0]),
        "pillar_side_um": float(decoded["pillar_side_um"][0]),
        "pillar_thickness_um": float(decoded["pillar_thickness_um"][0]),
        "film_thickness_um": float(decoded["film_thickness_um"][0]),
        "structure_material": structure_items[s_idx]["name"],
        "structure_material_idx": s_idx,
        "film_material": film_items[f_idx]["name"],
        "film_material_idx": f_idx,
    }


def run_pso(cfg: dict) -> None:
    runtime = cfg.get("runtime", {})
    device = torch.device("cpu" if runtime.get("cpu", False) or not torch.cuda.is_available() else "cuda")
    use_double = bool(runtime.get("double_precision", False))
    sim_dtype = torch.complex128 if use_double else torch.complex64
    geo_dtype = torch.float64 if use_double else torch.float32
    torch.backends.cuda.matmul.allow_tf32 = False

    wavelengths_um = wavelength_axis(cfg)
    check_rayleigh_points(wavelengths_um, np.array([0.0], dtype=np.float64), cfg["geometry"], cfg["rcwa"])
    target_r = target_reflection(cfg, wavelengths_um)
    material_data = load_material_candidates(cfg, wavelengths_um)
    lower, upper = particle_bounds(cfg)
    pso = cfg["pso"]
    rng = np.random.default_rng(int(pso.get("seed", 0)))
    particle_count = int(pso["particles"])
    iterations = int(pso["iterations"])
    velocity_span = (upper - lower) * float(pso.get("velocity_clamp_fraction", 0.25))

    position = lower[None, :] + rng.random((particle_count, len(lower))) * (upper - lower)[None, :]
    velocity = rng.uniform(-velocity_span[None, :], velocity_span[None, :], size=position.shape)
    pbest_position = position.copy()
    pbest_loss = np.full(particle_count, np.inf, dtype=np.float64)
    gbest_position = position[0].copy()
    gbest_loss = np.inf

    run_dir = Path(cfg["output"]["run_dir"])
    gen_dir = run_dir / "generations"
    gen_dir.mkdir(parents=True, exist_ok=True)
    history = []

    print(f"PSO device={device}, particles={particle_count}, iterations={iterations}")
    print(f"Optimizing normal-incidence reflection over {wavelengths_um[0]:.3g}-{wavelengths_um[-1]:.3g} um")
    for generation in tqdm(range(iterations), desc="PSO generations"):
        losses, _ = evaluate_particles(
            position,
            wavelengths_um,
            target_r,
            cfg,
            material_data,
            device,
            sim_dtype,
            geo_dtype,
            return_spectra=False,
            progress_desc=f"generation {generation + 1}/{iterations} RCWA",
        )
        improved = losses < pbest_loss
        pbest_loss[improved] = losses[improved]
        pbest_position[improved] = position[improved]
        best_idx = int(np.argmin(pbest_loss))
        if pbest_loss[best_idx] < gbest_loss:
            gbest_loss = float(pbest_loss[best_idx])
            gbest_position = pbest_position[best_idx].copy()

        _, best_spectrum = evaluate_particles(
            gbest_position[None, :],
            wavelengths_um,
            target_r,
            cfg,
            material_data,
            device,
            sim_dtype,
            geo_dtype,
            return_spectra=True,
            progress_desc=f"generation {generation + 1}/{iterations} best spectrum",
            show_progress=False,
        )
        best_reflection = best_spectrum[0]
        payload = structure_payload(gbest_position, gbest_loss, cfg)
        payload["generation"] = generation
        history.append(payload.copy())
        save_table(
            gen_dir / f"generation_{generation:04d}_best_spectrum.csv",
            {"wavelength_um": wavelengths_um, "R": best_reflection, "target_R": target_r},
        )
        write_json(gen_dir / f"generation_{generation:04d}_best_structure.json", payload)
        plot_generation(
            gen_dir / f"generation_{generation:04d}_best_spectrum.png",
            wavelengths_um,
            best_reflection,
            target_r,
            f"Generation {generation} best, loss={gbest_loss:.6g}",
        )
        tqdm.write(
            f"generation {generation + 1}/{iterations}: loss={gbest_loss:.6g}, "
            f"structure={payload['structure_material']}, film={payload['film_material']}, "
            f"p={payload['period_um']:.4g}, f={payload['fill_factor']:.4g}, "
            f"w={payload['pillar_side_um']:.4g}, d={payload['pillar_thickness_um']:.4g}, "
            f"film_h={payload['film_thickness_um']:.4g}"
        )

        r1 = rng.random(position.shape)
        r2 = rng.random(position.shape)
        velocity = (
            float(pso["inertia"]) * velocity
            + float(pso["cognitive"]) * r1 * (pbest_position - position)
            + float(pso["social"]) * r2 * (gbest_position[None, :] - position)
        )
        velocity = np.clip(velocity, -velocity_span[None, :], velocity_span[None, :])
        position = np.clip(position + velocity, lower[None, :], upper[None, :])

    history_path = run_dir / cfg["output"].get("history_csv", "history.csv")
    names = list(history[0].keys()) if history else []
    with history_path.open("w", encoding="utf-8") as handle:
        handle.write(",".join(names) + "\n")
        for row in history:
            handle.write(",".join(str(row[name]) for name in names) + "\n")

    save_table(
        run_dir / cfg["output"].get("best_csv", "best_spectrum.csv"),
        {"wavelength_um": wavelengths_um, "R": best_reflection, "target_R": target_r},
    )
    write_json(run_dir / cfg["output"].get("best_json", "best_structure.json"), structure_payload(gbest_position, gbest_loss, cfg))
    plot_generation(run_dir / "best_spectrum.png", wavelengths_um, best_reflection, target_r, f"Final best, loss={gbest_loss:.6g}")


def main():
    args = parse_args()
    cfg = load_config(args.config)
    run_pso(cfg)


if __name__ == "__main__":
    main()
