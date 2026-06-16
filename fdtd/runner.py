from __future__ import annotations

import json
from pathlib import Path

import yaml

from .builder import build_normal_incidence_square_pillar
from .extract import extract_rt
from .lumapi_loader import load_lumapi
from .plotting import plot_spectrum, save_csv


def load_config(path: str | Path) -> dict:
    with open(path, "r", encoding="utf-8") as handle:
        cfg = yaml.safe_load(handle)
    if not isinstance(cfg, dict):
        raise ValueError(f"Invalid YAML config: {path}")
    return cfg


def find_project_root(start: str | Path) -> Path:
    path = Path(start).resolve()
    for candidate in [path, *path.parents]:
        if (candidate / "fdtd").exists() and (candidate / "database").exists():
            return candidate
    raise FileNotFoundError(f"Could not locate project root from {path}")


def _merge_dict(base: dict, override: dict) -> dict:
    result = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = _merge_dict(result[key], value)
        else:
            result[key] = value
    return result


def resolve_config(config_path: str | Path) -> tuple[dict, Path]:
    config_path = Path(config_path).resolve()
    cfg = load_config(config_path)
    defaults_path = cfg.get("defaults")
    if defaults_path:
        base = load_config((config_path.parent / defaults_path).resolve())
        cfg = _merge_dict(base, {k: v for k, v in cfg.items() if k != "defaults"})
    return cfg, config_path


def resolve_main_configs(defaults_path: str | Path, structure_path: str | Path) -> tuple[dict, Path]:
    defaults_path = Path(defaults_path).resolve()
    structure_path = Path(structure_path).resolve()
    defaults_cfg = load_config(defaults_path)
    structure_cfg = load_config(structure_path)
    cfg = _merge_dict(defaults_cfg, structure_cfg)
    cfg["_config_sources"] = {
        "defaults": str(defaults_path),
        "structure": str(structure_path),
    }
    return cfg, structure_path


def _run_fdtd_cfg(cfg: dict, resolved_path: Path) -> dict:
    root = find_project_root(resolved_path)
    template_path = (root / cfg["template"]["fsp"]).resolve()
    output_dir = (root / cfg["output"]["run_dir"]).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    runtime_cfg = cfg.get("runtime", {})
    smoke_mode = runtime_cfg.get("smoke_mode", "off")
    if smoke_mode not in {"off", "build_only"}:
        raise ValueError("runtime.smoke_mode must be one of: off, build_only")

    lumapi = load_lumapi(runtime_cfg.get("lumapi_dir"))

    def reset_fdtd():
        fdtd = lumapi.FDTD(str(template_path))
        if fdtd.layoutmode() != 1:
            fdtd.switchtolayout()
        return fdtd

    table = {}
    layout = {}
    pol_mode = cfg["scan"].get("polarization", "both")
    pols = ["TE", "TM"] if pol_mode == "both" else [pol_mode]
    executed_run = smoke_mode == "off"
    for pol in pols:
        fdtd = reset_fdtd()
        fdtd.deleteall()
        layout[pol] = build_normal_incidence_square_pillar(fdtd, cfg, pol)
        if bool(cfg["output"].get("save_fsp", True)):
            fdtd.save(str(output_dir / f"{cfg['case_name']}_{pol}.fsp"))
        if smoke_mode == "build_only":
            continue
        fdtd.run()
        pol_table = extract_rt(fdtd, pol)
        if not table:
            table["wavelength_um"] = pol_table["wavelength_um"]
        for key, value in pol_table.items():
            if key == "wavelength_um":
                continue
            table[key] = value

    if executed_run and bool(cfg["output"].get("save_csv", True)):
        save_csv(output_dir / "fdtd_spectrum.csv", table)
    if executed_run and bool(cfg["output"].get("save_png", True)):
        plot_spectrum(output_dir / "fdtd_spectrum.png", table)
    if bool(cfg["output"].get("save_json", True)):
        config_sources = cfg.get("_config_sources")
        metadata = {
            "config": config_sources["structure"] if config_sources else str(resolved_path),
            "config_sources": config_sources,
            "case_name": cfg["case_name"],
            "geometry": cfg["geometry"],
            "materials": cfg["materials"],
            "layout": layout,
            "smoke_mode": smoke_mode,
            "executed_run": executed_run,
        }
        (output_dir / "case_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return {
        "config": str(resolved_path),
        "config_sources": cfg.get("_config_sources"),
        "output_dir": str(output_dir),
        "pols": pols,
        "smoke_mode": smoke_mode,
        "executed_run": executed_run,
    }


def run_fdtd_case(config_path: str | Path) -> dict:
    cfg, resolved_path = resolve_config(config_path)
    return _run_fdtd_cfg(cfg, resolved_path)


def run_fdtd_main(defaults_path: str | Path, structure_path: str | Path) -> dict:
    cfg, resolved_path = resolve_main_configs(defaults_path, structure_path)
    return _run_fdtd_cfg(cfg, resolved_path)
