"""
Config-driven metasurface spectra using the local batched torch_rcwa.

The script reads all geometry, material, scan, RCWA, and output settings from a
YAML file. Wavelength points are evaluated in true batched matrix operations by
torch_rcwa.rcwa, with the chunk size controlled by rcwa.batch_size.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch
import yaml

import torch_rcwa


DEFAULT_CONFIG = "configs/spectrum.yaml"


def parse_args():
    parser = argparse.ArgumentParser(
        description="Calculate batched RCWA R/T/A spectra and angle maps for a configured metasurface."
    )
    parser.add_argument("--config", default=DEFAULT_CONFIG, help="YAML configuration file.")
    parser.add_argument("--no_plots", action="store_true", help="Skip figure generation.")
    parser.add_argument("--no_spectrum", action="store_true", help="Skip normal-incidence spectrum for this process.")
    parser.add_argument("--no_angle_sweep", action="store_true", help="Skip angle sweep for this process.")
    parser.add_argument(
        "--angle_shards",
        type=int,
        default=1,
        help="Split the configured angle axis into this many independent shards for this process.",
    )
    parser.add_argument(
        "--angle_shard_index",
        type=int,
        default=0,
        help="Zero-based angle shard index to run in this process.",
    )
    parser.add_argument(
        "--output_suffix",
        default=None,
        help="Suffix for sharded angle outputs. Defaults to angle_shard_<index>_of_<count>.",
    )
    parser.add_argument(
        "--merge_angle_shards",
        type=int,
        default=0,
        help="Merge this many sharded angle CSV files and exit without running RCWA.",
    )
    parser.add_argument(
        "--run_angle_shards",
        type=int,
        default=0,
        help="Launch this many child processes for angle shards, then merge and plot.",
    )
    parser.add_argument(
        "--keep_angle_shards",
        action="store_true",
        help="Keep per-shard angle CSV files after automatic shard merge.",
    )
    return parser.parse_args()


def load_config(path: str | Path) -> dict:
    with open(path, "r", encoding="utf-8") as handle:
        cfg = yaml.safe_load(handle)
    if not isinstance(cfg, dict):
        raise ValueError(f"Invalid YAML config: {path}")
    return cfg


def material_path(materials: dict, key: str, legacy_key: str | None = None) -> str:
    if key in materials:
        return materials[key]
    if legacy_key and legacy_key in materials:
        return materials[legacy_key]
    raise KeyError(f"materials.{key} is required")


def material_name_from_path(path: str | Path) -> str:
    return Path(path).stem


def stack_summary(cfg: dict):
    geometry = cfg["geometry"]
    materials = cfg.get("materials", {})
    structure_name = materials.get(
        "structure_name",
        material_name_from_path(material_path(materials, "structure_csv", "zns_csv")),
    )
    substrate_name = materials.get(
        "substrate_name",
        material_name_from_path(material_path(materials, "substrate_csv", "si_csv")),
    )
    return [
        ("input", "Air", None),
        ("layer", f"patterned {structure_name} square pillar in air", float(geometry["pillar_thickness_um"])),
        ("layer", f"uniform {structure_name} film", float(geometry["film_thickness_um"])),
        ("output", f"semi-infinite {substrate_name}", None),
    ]


def wavelength_axis(cfg: dict) -> np.ndarray:
    scan = cfg["scan"]["wavelength_um"]
    return np.linspace(float(scan["start"]), float(scan["stop"]), int(scan["points"]), dtype=np.float64)


def angle_axis(cfg: dict) -> np.ndarray:
    scan = cfg["scan"].get("angle_deg", {"start": 0.0, "stop": 60.0, "points": 31})
    return np.linspace(float(scan["start"]), float(scan["stop"]), int(scan["points"]), dtype=np.float64)


def rayleigh_matches(
    wavelengths_um: np.ndarray,
    angles_deg: np.ndarray,
    geometry: dict,
    order: int,
    *,
    atol_um: float = 1e-9,
) -> list[dict]:
    """Find scan points on input-side Rayleigh diffraction thresholds."""
    period_x, period_y = [float(v) for v in geometry["period_um"]]
    if period_x <= 0.0 or period_y <= 0.0:
        raise ValueError("geometry.period_um values must be positive")

    matches: list[dict] = []
    wavelengths_um = np.asarray(wavelengths_um, dtype=np.float64)
    angles_deg = np.asarray(angles_deg, dtype=np.float64)
    orders = diffraction_orders(int(order))
    nonzero_orders = orders[np.any(orders != 0, axis=1)]

    for angle_deg in angles_deg:
        theta = np.deg2rad(float(angle_deg))
        sin_theta = float(np.sin(theta))
        for mx, my in nonzero_orders:
            gx = float(mx) / period_x
            gy = float(my) / period_y
            g_norm_sq = gx * gx + gy * gy
            # Input-side air threshold: |k_parallel + G| = k0.
            # With azimuth fixed to x and after multiplying by lambda^2:
            # (sin(theta) + gx * lambda)^2 + (gy * lambda)^2 = 1.
            a = g_norm_sq
            b = 2.0 * sin_theta * gx
            c = sin_theta * sin_theta - 1.0
            roots: list[float] = []
            if abs(a) < 1e-14:
                if abs(b) > 1e-14:
                    roots.append(-c / b)
            else:
                disc = b * b - 4.0 * a * c
                if disc >= -1e-14:
                    disc = max(disc, 0.0)
                    sqrt_disc = float(np.sqrt(disc))
                    roots.extend([(-b - sqrt_disc) / (2.0 * a), (-b + sqrt_disc) / (2.0 * a)])
            for lambda_c in roots:
                if lambda_c <= 0.0:
                    continue
                hit_indices = np.flatnonzero(np.isclose(wavelengths_um, lambda_c, rtol=0.0, atol=atol_um))
                for idx in hit_indices:
                    matches.append(
                        {
                            "wavelength_index": int(idx),
                            "wavelength_um": float(wavelengths_um[idx]),
                            "critical_wavelength_um": float(lambda_c),
                            "angle_deg": float(angle_deg),
                            "order": (int(mx), int(my)),
                        }
                    )
    matches.sort(key=lambda item: (item["wavelength_index"], item["angle_deg"], item["order"]))
    return matches


def check_rayleigh_points(
    wavelengths_um: np.ndarray,
    angles_deg: np.ndarray,
    geometry: dict,
    rcwa_cfg: dict,
) -> None:
    check_cfg = rcwa_cfg.get("rayleigh_check", {})
    if check_cfg is False or (isinstance(check_cfg, dict) and not check_cfg.get("enabled", True)):
        return
    atol_um = float(check_cfg.get("atol_um", 1e-9)) if isinstance(check_cfg, dict) else 1e-9
    matches = rayleigh_matches(
        wavelengths_um,
        angles_deg,
        geometry,
        int(rcwa_cfg["fourier_order"]),
        atol_um=atol_um,
    )
    if not matches:
        return

    lines = [
        "Rayleigh critical scan point detected; stopping before RCWA solve.",
        "These points can make the RCWA linear system singular. Adjust scan.wavelength_um start/stop/points,",
        "or move the wavelength grid slightly away from the listed critical values.",
    ]
    max_lines = 20
    for match in matches[:max_lines]:
        lines.append(
            "  "
            f"wavelength[{match['wavelength_index']}]={match['wavelength_um']:.12g} um "
            f"matches lambda_c={match['critical_wavelength_um']:.12g} um, "
            f"angle={match['angle_deg']:.12g} deg, order={match['order']}"
        )
    if len(matches) > max_lines:
        lines.append(f"  ... {len(matches) - max_lines} more matches omitted")
    raise ValueError("\n".join(lines))


def select_angle_shard(angles_deg: np.ndarray, shard_count: int, shard_index: int) -> np.ndarray:
    if shard_count < 1:
        raise ValueError("--angle_shards must be >= 1")
    if shard_index < 0 or shard_index >= shard_count:
        raise ValueError("--angle_shard_index must satisfy 0 <= index < --angle_shards")
    return np.array_split(angles_deg, shard_count)[shard_index]


def suffixed_path(path: str | Path, suffix: str | None) -> str:
    if not suffix:
        return str(path)
    path = Path(path)
    return str(path.with_name(f"{path.stem}_{suffix}{path.suffix}"))


def angle_shard_suffix(shard_count: int, shard_index: int, output_suffix: str | None) -> str | None:
    if output_suffix is not None:
        return output_suffix
    if shard_count <= 1:
        return None
    return f"angle_shard_{shard_index}_of_{shard_count}"


def merge_angle_shards(cfg: dict, shard_count: int, pols: list[str], no_plots: bool = False) -> None:
    base_path = Path(cfg["output"]["angle_csv"])
    tables = []
    names = None
    for shard_index in range(shard_count):
        shard_path = Path(suffixed_path(base_path, angle_shard_suffix(shard_count, shard_index, None)))
        if not shard_path.exists():
            raise FileNotFoundError(f"Missing angle shard CSV: {shard_path}")
        data = np.genfromtxt(shard_path, delimiter=",", names=True)
        if data.ndim == 0:
            data = np.array([data], dtype=data.dtype)
        shard_names = list(data.dtype.names or [])
        if names is None:
            names = shard_names
        elif shard_names != names:
            raise ValueError(f"Shard columns differ in {shard_path}")
        tables.append(data)

    if names is None:
        raise ValueError("No angle shards to merge")
    merged = np.concatenate(tables)
    order = np.lexsort((merged["angle_deg"], merged["wavelength_um"]))
    merged = merged[order]
    table = {name: np.asarray(merged[name]) for name in names}
    save_table(base_path, table)

    if not no_plots and not cfg.get("output", {}).get("no_plots", False):
        plot_angle_maps(cfg["output"]["angle_fig"], table, pols)


def run_angle_shards(args) -> None:
    if args.run_angle_shards < 2:
        raise ValueError("--run_angle_shards must be >= 2")

    script_path = Path(__file__).resolve()
    base_cmd = [
        sys.executable,
        str(script_path),
        "--config",
        args.config,
        "--angle_shards",
        str(args.run_angle_shards),
    ]
    if args.no_plots:
        base_cmd.append("--no_plots")

    processes = []
    for shard_index in range(args.run_angle_shards):
        cmd = [*base_cmd, "--angle_shard_index", str(shard_index)]
        if shard_index > 0:
            cmd.append("--no_spectrum")
        print("Launching:", " ".join(cmd))
        processes.append(subprocess.Popen(cmd))

    failed = []
    for shard_index, process in enumerate(processes):
        returncode = process.wait()
        if returncode != 0:
            failed.append((shard_index, returncode))
    if failed:
        details = ", ".join(f"shard {idx} exit {code}" for idx, code in failed)
        raise RuntimeError(f"Angle shard run failed: {details}")

    merge_cmd = [
        sys.executable,
        str(script_path),
        "--config",
        args.config,
        "--merge_angle_shards",
        str(args.run_angle_shards),
    ]
    if args.no_plots:
        merge_cmd.append("--no_plots")
    print("Merging:", " ".join(merge_cmd))
    subprocess.run(merge_cmd, check=True)

    if not args.keep_angle_shards:
        cfg = load_config(args.config)
        base_path = Path(cfg["output"]["angle_csv"])
        for shard_index in range(args.run_angle_shards):
            shard_path = Path(suffixed_path(base_path, angle_shard_suffix(args.run_angle_shards, shard_index, None)))
            if shard_path.exists():
                shard_path.unlink()


def selected_pols(cfg: dict) -> list[str]:
    pol = cfg["scan"].get("polarization", "both")
    if pol == "both":
        return ["TE", "TM"]
    if pol in ["TE", "TM"]:
        return [pol]
    raise ValueError("scan.polarization must be TE, TM, or both")


def load_material_eps(
    csv_path: str | Path,
    wavelengths_um: np.ndarray,
    wavelength_unit: str = "um",
) -> np.ndarray:
    path = Path(csv_path)
    if not path.exists():
        raise FileNotFoundError(f"Material CSV not found: {path}")
    data = np.genfromtxt(path, delimiter=",", names=True)
    names = data.dtype.names or ()
    if not {"wl", "n", "k"}.issubset(names):
        raise ValueError(f"{path} must contain columns: wl,n,k")
    wl = np.asarray(data["wl"], dtype=np.float64)
    n = np.asarray(data["n"], dtype=np.float64)
    k = np.asarray(data["k"], dtype=np.float64)
    order = np.argsort(wl)
    wl = wl[order]
    n = n[order]
    k = k[order]
    if wavelength_unit == "nm":
        wl = wl / 1000.0
    elif wavelength_unit != "um":
        raise ValueError("wavelength_unit must be 'um' or 'nm'")
    n_interp = np.interp(wavelengths_um, wl, n)
    k_interp = np.interp(wavelengths_um, wl, k)
    return (n_interp + 1j * k_interp) ** 2


def input_channels(pol: str) -> tuple[str, str]:
    if pol == "TE":
        return ("ss", "ps")
    if pol == "TM":
        return ("pp", "sp")
    raise ValueError("pol must be TE or TM")


def diffraction_orders(order: int) -> np.ndarray:
    order_axis = np.arange(-order, order + 1, dtype=np.int64)
    return np.stack(np.meshgrid(order_axis, order_axis, indexing="ij"), axis=-1).reshape(-1, 2)


def get_runtime_cache(
    runtime_cache: dict,
    geometry: dict,
    rcwa_cfg: dict,
    geo_dtype: torch.dtype,
    sim_dtype: torch.dtype,
    device: torch.device,
) -> dict:
    cache_key = (
        tuple(float(v) for v in geometry["period_um"]),
        float(geometry["pillar_side_um"]),
        float(geometry.get("edge_sharpness", 1000.0)),
        int(rcwa_cfg["fourier_order"]),
        tuple(int(v) for v in rcwa_cfg["grid"]),
        str(geo_dtype),
        str(sim_dtype),
        str(device),
    )
    cached = runtime_cache.get("geometry_cache")
    if cached is not None and cached.get("key") == cache_key:
        return cached

    order = int(rcwa_cfg["fourier_order"])
    pillar_mask = build_pillar_mask(geometry, rcwa_cfg, geo_dtype, device).to(sim_dtype)
    nx, ny = [int(v) for v in rcwa_cfg["grid"]]
    conv_builder = torch_rcwa.rcwa(
        freq=torch.ones(1, dtype=geo_dtype, device=device),
        order=[order, order],
        L=[float(v) for v in geometry["period_um"]],
        dtype=sim_dtype,
        device=device,
        linalg_batch_mode=rcwa_cfg.get("linalg_batch_mode", "auto"),
        linalg_batch_threshold=int(rcwa_cfg.get("linalg_batch_threshold", 512)),
    )
    air_conv = conv_builder.convolution_matrix(torch.ones((nx, ny), dtype=sim_dtype, device=device))[0]
    pillar_shape_conv = conv_builder.convolution_matrix(pillar_mask)[0]
    cached = {
        "key": cache_key,
        "order": order,
        "period": [float(v) for v in geometry["period_um"]],
        "azimuth_deg": float(rcwa_cfg.get("azimuth_deg", 0.0)),
        "pillar_mask": pillar_mask,
        "air_conv": air_conv,
        "pillar_shape_conv": pillar_shape_conv,
        "orders": diffraction_orders(order),
    }
    cached["orders_t"] = torch.tensor(cached["orders"], dtype=torch.int64, device=device)
    runtime_cache["geometry_cache"] = cached
    return cached


def build_pillar_mask(geometry: dict, rcwa_cfg: dict, geo_dtype: torch.dtype, device: torch.device) -> torch.Tensor:
    nx, ny = [int(v) for v in rcwa_cfg["grid"]]
    period_x, period_y = [float(v) for v in geometry["period_um"]]
    side = float(geometry["pillar_side_um"])
    edge_sharpness = float(geometry.get("edge_sharpness", 1000.0))

    x = (period_x / nx) * (torch.arange(nx, dtype=geo_dtype, device=device) + 0.5)
    y = (period_y / ny) * (torch.arange(ny, dtype=geo_dtype, device=device) + 0.5)
    x_grid, y_grid = torch.meshgrid(x, y, indexing="ij")
    level = 1.0 - torch.maximum(
        torch.abs((x_grid - period_x / 2.0) / (side / 2.0)),
        torch.abs((y_grid - period_y / 2.0) / (side / 2.0)),
    )
    return torch.sigmoid(edge_sharpness * level)


def propagating_order_mask_batch(
    wavelengths_um: np.ndarray,
    eps_medium: np.ndarray,
    angle_deg: float,
    azimuth_deg: float,
    orders: np.ndarray,
    geometry: dict,
    device: torch.device,
) -> torch.Tensor:
    period_x, period_y = [float(v) for v in geometry["period_um"]]
    eps_real = np.real(eps_medium).astype(np.float64)
    n_medium = np.sqrt(np.maximum(eps_real, 0.0))
    theta = np.deg2rad(angle_deg)
    phi = np.deg2rad(azimuth_deg)
    qx = (
        1.0 / wavelengths_um[:, None] * np.sin(theta) * np.cos(phi)
        + orders[None, :, 0] / period_x
    )
    qy = (
        1.0 / wavelengths_um[:, None] * np.sin(theta) * np.sin(phi)
        + orders[None, :, 1] / period_y
    )
    q_parallel_sq = qx**2 + qy**2
    propagation_limit = (n_medium[:, None] / wavelengths_um[:, None]) ** 2
    mask = q_parallel_sq <= propagation_limit + 1e-12
    return torch.tensor(mask, dtype=torch.bool, device=device)


def propagating_order_mask_tensor(
    freq: torch.Tensor,
    eps_medium: torch.Tensor,
    angle_rad: torch.Tensor,
    azimuth_rad: float,
    orders: torch.Tensor,
    geometry: dict,
) -> torch.Tensor:
    period_x, period_y = [float(v) for v in geometry["period_um"]]
    real_dtype = freq.real.dtype
    orders = orders.to(dtype=real_dtype, device=freq.device)
    n_medium = torch.sqrt(torch.clamp(torch.real(eps_medium).to(real_dtype), min=0.0))
    sin_theta = torch.sin(angle_rad)
    cos_phi = float(np.cos(azimuth_rad))
    sin_phi = float(np.sin(azimuth_rad))
    freq_real = freq.real
    qx = freq_real[:, None] * sin_theta[:, None] * cos_phi + orders[None, :, 0] / period_x
    qy = freq_real[:, None] * sin_theta[:, None] * sin_phi + orders[None, :, 1] / period_y
    q_parallel_sq = qx**2 + qy**2
    propagation_limit = (n_medium[:, None] * freq_real[:, None]) ** 2
    return q_parallel_sq <= propagation_limit + 1e-12


def channel_power_batch(sim, port: str, channels: tuple[str, str], orders: np.ndarray, mask: torch.Tensor) -> torch.Tensor:
    total = torch.zeros(sim.batch_size, dtype=sim._real_dtype, device=sim._device)
    for channel in channels:
        coeff = sim.S_parameters(
            orders=orders,
            direction="forward",
            port=port,
            polarization=channel,
            ref_order=[0, 0],
        )
        if coeff.dim() == 1:
            coeff = coeff[:, None]
        total = total + torch.sum((torch.abs(coeff) ** 2).real * mask, dim=1)
    return total


def prepare_scan_tensors(
    wavelengths_um: np.ndarray,
    eps_structure: np.ndarray,
    eps_substrate: np.ndarray,
    angles_deg: np.ndarray,
    device: torch.device,
    sim_dtype: torch.dtype,
    geo_dtype: torch.dtype,
) -> dict[str, torch.Tensor]:
    return {
        "wavelengths_um": torch.as_tensor(wavelengths_um, dtype=geo_dtype, device=device),
        "eps_structure": torch.as_tensor(eps_structure, dtype=sim_dtype, device=device),
        "eps_substrate": torch.as_tensor(eps_substrate, dtype=sim_dtype, device=device),
        "angles_rad": torch.as_tensor(np.deg2rad(angles_deg), dtype=geo_dtype, device=device),
    }


def simulate_batch_tensor(
    wavelengths_um_t: torch.Tensor,
    eps_structure_t_base: torch.Tensor,
    eps_substrate_t_base: torch.Tensor,
    angles_rad_t: torch.Tensor,
    pols: list[str],
    geometry: dict,
    rcwa_cfg: dict,
    runtime_cache: dict,
    device: torch.device,
    sim_dtype: torch.dtype,
    geo_dtype: torch.dtype,
) -> dict[str, dict[str, torch.Tensor]]:
    cache = get_runtime_cache(runtime_cache, geometry, rcwa_cfg, geo_dtype, sim_dtype, device)
    order = cache["order"]
    period = cache["period"]
    azimuth_deg = cache["azimuth_deg"]
    air_conv = cache["air_conv"]
    pillar_shape_conv = cache["pillar_shape_conv"]
    orders_t = cache["orders_t"]

    wavelength_count = int(wavelengths_um_t.numel())
    angle_count = int(angles_rad_t.numel())
    freq = (1.0 / wavelengths_um_t).repeat_interleave(angle_count)
    eps_structure_t = eps_structure_t_base.repeat_interleave(angle_count)
    eps_substrate_t = eps_substrate_t_base.repeat_interleave(angle_count)
    angle_rad = angles_rad_t.repeat(wavelength_count)
    patterned_eps_conv = air_conv[None, :, :] + (eps_structure_t - 1.0)[:, None, None] * pillar_shape_conv[None, :, :]

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
    sim.add_output_layer(eps=eps_substrate_t)
    sim.set_incident_angle(
        inc_ang=angle_rad,
        azi_ang=torch.full((freq.numel(),), np.deg2rad(azimuth_deg), dtype=geo_dtype, device=device),
    )
    sim.add_layer_conv(thickness=float(geometry["pillar_thickness_um"]), eps_conv=patterned_eps_conv)
    sim.add_layer(thickness=float(geometry["film_thickness_um"]), eps=eps_structure_t)
    sim.solve_global_smatrix()

    reflection_mask = propagating_order_mask_tensor(
        freq,
        torch.ones_like(eps_structure_t),
        angle_rad,
        np.deg2rad(azimuth_deg),
        orders_t,
        geometry,
    )
    transmission_mask = propagating_order_mask_tensor(
        freq,
        eps_substrate_t,
        angle_rad,
        np.deg2rad(azimuth_deg),
        orders_t,
        geometry,
    )

    out: dict[str, dict[str, torch.Tensor]] = {}
    for pol in pols:
        channels = input_channels(pol)
        r = channel_power_batch(sim, "reflection", channels, orders_t, reflection_mask)
        t = channel_power_batch(sim, "transmission", channels, orders_t, transmission_mask)
        a = torch.ones_like(r) - r - t
        out[pol] = {
            "R": r.detach(),
            "T": t.detach(),
            "A": a.detach(),
        }
    return out


def simulate_batch(
    wavelengths_um: np.ndarray,
    eps_structure: np.ndarray,
    eps_substrate: np.ndarray,
    angles_deg: np.ndarray,
    pols: list[str],
    geometry: dict,
    rcwa_cfg: dict,
    runtime_cache: dict,
    device: torch.device,
    sim_dtype: torch.dtype,
    geo_dtype: torch.dtype,
) -> dict[str, dict[str, torch.Tensor]]:
    scan_tensors = prepare_scan_tensors(wavelengths_um, eps_structure, eps_substrate, angles_deg, device, sim_dtype, geo_dtype)
    return simulate_batch_tensor(
        scan_tensors["wavelengths_um"],
        scan_tensors["eps_structure"],
        scan_tensors["eps_substrate"],
        scan_tensors["angles_rad"],
        pols,
        geometry,
        rcwa_cfg,
        runtime_cache,
        device,
        sim_dtype,
        geo_dtype,
    )


def iter_slices(length: int, batch_size: int):
    for start in range(0, length, batch_size):
        yield slice(start, min(start + batch_size, length))


def iter_angle_slices(wavelength_count: int, angle_count: int, max_batch_elements: int | None):
    if max_batch_elements is None:
        yield slice(0, angle_count)
        return
    if max_batch_elements < wavelength_count:
        raise ValueError("rcwa.max_batch_elements must be >= the wavelength chunk size")
    angle_chunk_size = max(1, int(max_batch_elements) // wavelength_count)
    for start in range(0, angle_count, angle_chunk_size):
        yield slice(start, min(start + angle_chunk_size, angle_count))


def as_device_tensor(value, dtype: torch.dtype, device: torch.device) -> torch.Tensor:
    if isinstance(value, torch.Tensor):
        return value.to(dtype=dtype, device=device)
    return torch.as_tensor(value, dtype=dtype, device=device)


def run_spectrum(
    wavelengths_um: np.ndarray,
    eps_structure: np.ndarray,
    eps_substrate: np.ndarray,
    pols: list[str],
    geometry: dict,
    rcwa_cfg: dict,
    runtime_cache: dict,
    device: torch.device,
    sim_dtype: torch.dtype,
    geo_dtype: torch.dtype,
) -> dict[str, np.ndarray]:
    batch_size = int(rcwa_cfg.get("batch_size", len(wavelengths_um)))
    out_dtype = geo_dtype or torch.float64
    scan_tensors = prepare_scan_tensors(
        wavelengths_um,
        eps_structure,
        eps_substrate,
        np.array([0.0], dtype=np.float64),
        device,
        sim_dtype,
        geo_dtype,
    )
    out_t: dict[str, torch.Tensor] = {}
    for pol in pols:
        out_t[f"R_{pol}"] = torch.empty(len(wavelengths_um), dtype=out_dtype, device=device)
        out_t[f"T_{pol}"] = torch.empty(len(wavelengths_um), dtype=out_dtype, device=device)
        out_t[f"A_{pol}"] = torch.empty(len(wavelengths_um), dtype=out_dtype, device=device)

    for chunk in iter_slices(len(wavelengths_um), batch_size):
        result = simulate_batch_tensor(
            scan_tensors["wavelengths_um"][chunk],
            scan_tensors["eps_structure"][chunk],
            scan_tensors["eps_substrate"][chunk],
            scan_tensors["angles_rad"],
            pols,
            geometry,
            rcwa_cfg,
            runtime_cache,
            device,
            sim_dtype,
            geo_dtype,
        )
        for pol in pols:
            out_t[f"R_{pol}"][chunk] = as_device_tensor(result[pol]["R"], out_dtype, device)
            out_t[f"T_{pol}"][chunk] = as_device_tensor(result[pol]["T"], out_dtype, device)
            out_t[f"A_{pol}"][chunk] = as_device_tensor(result[pol]["A"], out_dtype, device)
        print(
            f"spectrum {'/'.join(pols)}: wavelengths {chunk.start + 1}-{chunk.stop}/{len(wavelengths_um)}, angle 0 deg"
        )
    out = {"wavelength_um": wavelengths_um.copy()}
    for name, value in out_t.items():
        out[name] = value.cpu().numpy()
    return out


def run_angle_sweep(
    wavelengths_um: np.ndarray,
    angles_deg: np.ndarray,
    eps_structure: np.ndarray,
    eps_substrate: np.ndarray,
    pols: list[str],
    geometry: dict,
    rcwa_cfg: dict,
    runtime_cache: dict,
    device: torch.device,
    sim_dtype: torch.dtype,
    geo_dtype: torch.dtype,
    spectrum_table: dict[str, np.ndarray] | None = None,
) -> dict[str, np.ndarray]:
    batch_size = int(rcwa_cfg.get("batch_size", len(wavelengths_um)))
    out_dtype = geo_dtype or torch.float64
    max_batch_elements = rcwa_cfg.get("max_batch_elements")
    max_batch_elements = None if max_batch_elements is None else int(max_batch_elements)
    scan_tensors = prepare_scan_tensors(wavelengths_um, eps_structure, eps_substrate, angles_deg, device, sim_dtype, geo_dtype)
    out: dict[str, np.ndarray] = {
        "wavelength_um": np.repeat(wavelengths_um, len(angles_deg)),
        "angle_deg": np.tile(angles_deg, len(wavelengths_um)),
    }
    out_t: dict[str, torch.Tensor] = {}
    for pol in pols:
        out_t[f"R_{pol}"] = torch.empty((len(wavelengths_um), len(angles_deg)), dtype=out_dtype, device=device)
        out_t[f"T_{pol}"] = torch.empty((len(wavelengths_um), len(angles_deg)), dtype=out_dtype, device=device)
        out_t[f"A_{pol}"] = torch.empty((len(wavelengths_um), len(angles_deg)), dtype=out_dtype, device=device)

    zero_angle_indices = np.flatnonzero(np.isclose(angles_deg, 0.0, rtol=0.0, atol=1e-12))
    reused_zero_angle = spectrum_table is not None and len(zero_angle_indices) > 0
    if reused_zero_angle:
        zero_idx = int(zero_angle_indices[0])
        for pol in pols:
            for quantity in ["R", "T", "A"]:
                out_t[f"{quantity}_{pol}"][:, zero_idx] = torch.as_tensor(
                    spectrum_table[f"{quantity}_{pol}"], dtype=out_dtype, device=device
                )

    tasks = []
    for chunk in iter_slices(len(wavelengths_um), batch_size):
        for angle_chunk in iter_angle_slices(chunk.stop - chunk.start, len(angles_deg), max_batch_elements):
            angle_subset = angles_deg[angle_chunk]
            if len(angle_subset) == 0:
                continue
            if reused_zero_angle:
                keep = ~np.isclose(angle_subset, 0.0, rtol=0.0, atol=1e-12)
                if not np.any(keep):
                    continue
                global_angle_indices = np.arange(angle_chunk.start, angle_chunk.stop)[keep]
            else:
                global_angle_indices = np.arange(angle_chunk.start, angle_chunk.stop)
            tasks.append((chunk, global_angle_indices))

    for chunk, global_angle_indices in tasks:
        result = simulate_batch_tensor(
            scan_tensors["wavelengths_um"][chunk],
            scan_tensors["eps_structure"][chunk],
            scan_tensors["eps_substrate"][chunk],
            scan_tensors["angles_rad"][global_angle_indices],
            pols,
            geometry,
            rcwa_cfg,
            runtime_cache,
            device,
            sim_dtype,
            geo_dtype,
        )
        shape = (chunk.stop - chunk.start, len(global_angle_indices))
        for pol in pols:
            out_t[f"R_{pol}"][chunk, global_angle_indices] = as_device_tensor(result[pol]["R"], out_dtype, device).reshape(shape)
            out_t[f"T_{pol}"][chunk, global_angle_indices] = as_device_tensor(result[pol]["T"], out_dtype, device).reshape(shape)
            out_t[f"A_{pol}"][chunk, global_angle_indices] = as_device_tensor(result[pol]["A"], out_dtype, device).reshape(shape)
        angle_subset = angles_deg[global_angle_indices]
        print(
            f"angle {'/'.join(pols)}: wavelengths {chunk.start + 1}-{chunk.stop}/{len(wavelengths_um)}, "
            f"angles {angle_subset[0]:.6g}-{angle_subset[-1]:.6g} deg"
        )

    for pol in pols:
        out[f"R_{pol}"] = out_t[f"R_{pol}"].reshape(-1).cpu().numpy()
        out[f"T_{pol}"] = out_t[f"T_{pol}"].reshape(-1).cpu().numpy()
        out[f"A_{pol}"] = out_t[f"A_{pol}"].reshape(-1).cpu().numpy()
    return out


def save_table(path: str | Path, table: dict[str, np.ndarray]):
    names = list(table.keys())
    data = np.column_stack([table[name] for name in names])
    np.savetxt(path, data, delimiter=",", header=",".join(names), comments="")
    print(f"Saved {path}")


def plot_spectrum(path: str | Path, spectrum: dict[str, np.ndarray], pols: list[str]):
    import matplotlib.pyplot as plt

    wl = spectrum["wavelength_um"]
    fig, axes = plt.subplots(1, len(pols), figsize=(7 * len(pols), 4.8), squeeze=False)
    for ax, pol in zip(axes[0], pols):
        ax.plot(wl, spectrum[f"R_{pol}"], label=f"{pol} R", linewidth=2)
        ax.plot(wl, spectrum[f"T_{pol}"], label=f"{pol} T", linewidth=2)
        ax.plot(wl, spectrum[f"A_{pol}"], label=f"{pol} A", linewidth=2)
        ax.set_title(f"Normal incidence {pol}")
        ax.set_xlabel("Wavelength (um)")
        ax.set_ylabel("R / T / A")
        ax.set_ylim(-0.05, 1.05)
        ax.grid(True, alpha=0.3)
        ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=300)
    plt.close(fig)
    print(f"Saved {path}")


def plot_angle_maps(path: str | Path, angle_table: dict[str, np.ndarray], pols: list[str]):
    import matplotlib.pyplot as plt

    wl = np.unique(angle_table["wavelength_um"])
    angles = np.unique(angle_table["angle_deg"])
    fig, axes = plt.subplots(len(pols), 3, figsize=(13.5, 4.2 * len(pols)), squeeze=False)
    for row, pol in enumerate(pols):
        for col, quantity in enumerate(["R", "T", "A"]):
            values = angle_table[f"{quantity}_{pol}"].reshape(len(wl), len(angles))
            im = axes[row, col].imshow(
                values,
                origin="lower",
                aspect="auto",
                extent=[angles.min(), angles.max(), wl.min(), wl.max()],
                vmin=-0.05,
                vmax=1.05,
                cmap="magma" if quantity == "A" else "viridis",
            )
            axes[row, col].set_title(f"{pol} {quantity}")
            axes[row, col].set_xlabel("Incident angle (deg)")
            axes[row, col].set_ylabel("Wavelength (um)")
            fig.colorbar(im, ax=axes[row, col], fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(path, dpi=300)
    plt.close(fig)
    print(f"Saved {path}")


def main():
    args = parse_args()
    if args.run_angle_shards:
        run_angle_shards(args)
        return

    cfg = load_config(args.config)
    rcwa_cfg = cfg["rcwa"]
    runtime = cfg.get("runtime", {})
    device = torch.device("cpu" if runtime.get("cpu", False) or not torch.cuda.is_available() else "cuda")
    use_double = bool(runtime.get("double_precision", False))
    sim_dtype = torch.complex128 if use_double else torch.complex64
    geo_dtype = torch.float64 if use_double else torch.float32
    torch.backends.cuda.matmul.allow_tf32 = False

    wavelengths_um = wavelength_axis(cfg)
    angles_deg = angle_axis(cfg)
    pols = selected_pols(cfg)

    if args.merge_angle_shards:
        merge_angle_shards(cfg, int(args.merge_angle_shards), pols, no_plots=args.no_plots)
        return

    angle_suffix = angle_shard_suffix(args.angle_shards, args.angle_shard_index, args.output_suffix)
    if args.angle_shards > 1:
        angles_deg = select_angle_shard(angles_deg, args.angle_shards, args.angle_shard_index)
    check_rayleigh_points(wavelengths_um, angles_deg, cfg["geometry"], rcwa_cfg)

    material_wavelength_unit = cfg["materials"].get("csv_wavelength_unit", "um")
    eps_structure = load_material_eps(
        material_path(cfg["materials"], "structure_csv", "zns_csv"),
        wavelengths_um,
        material_wavelength_unit,
    )
    eps_substrate = load_material_eps(
        material_path(cfg["materials"], "substrate_csv", "si_csv"),
        wavelengths_um,
        material_wavelength_unit,
    )

    print(
        f"Using torch_rcwa, device={device}, dtype={sim_dtype}, "
        f"order={rcwa_cfg['fourier_order']}, grid={rcwa_cfg['grid']}, batch_size={rcwa_cfg.get('batch_size')}"
    )
    if args.angle_shards > 1:
        print(
            f"Angle shard {args.angle_shard_index + 1}/{args.angle_shards}: "
            f"{len(angles_deg)} angles from {angles_deg[0]:.6g} to {angles_deg[-1]:.6g} deg"
        )
    print("Stack:")
    for kind, name, thickness in stack_summary(cfg):
        suffix = "" if thickness is None else f", thickness={thickness} um"
        print(f"  {kind}: {name}{suffix}")

    run_spectrum_enabled = bool(runtime.get("run_spectrum", True)) and not args.no_spectrum
    run_angle_sweep_enabled = bool(runtime.get("run_angle_sweep", True)) and not args.no_angle_sweep
    runtime_cache: dict = {}
    spectrum = None
    angle_table = None

    with torch.no_grad():
        if run_spectrum_enabled:
            spectrum = run_spectrum(
                wavelengths_um,
                eps_structure,
                eps_substrate,
                pols,
                cfg["geometry"],
                rcwa_cfg,
                runtime_cache,
                device,
                sim_dtype,
                geo_dtype,
            )
            save_table(cfg["output"]["spectrum_csv"], spectrum)

        if run_angle_sweep_enabled:
            angle_table = run_angle_sweep(
                wavelengths_um,
                angles_deg,
                eps_structure,
                eps_substrate,
                pols,
                cfg["geometry"],
                rcwa_cfg,
                runtime_cache,
                device,
                sim_dtype,
                geo_dtype,
                spectrum_table=spectrum,
            )
            save_table(suffixed_path(cfg["output"]["angle_csv"], angle_suffix), angle_table)

    if not args.no_plots and not cfg.get("output", {}).get("no_plots", False):
        if spectrum is not None:
            plot_spectrum(cfg["output"]["spectrum_fig"], spectrum, pols)
        if angle_table is not None:
            plot_angle_maps(suffixed_path(cfg["output"]["angle_fig"], angle_suffix), angle_table, pols)


if __name__ == "__main__":
    main()
