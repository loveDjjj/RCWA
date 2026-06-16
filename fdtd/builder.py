from __future__ import annotations


UM = 1e-6


def build_normal_incidence_square_pillar(fdtd, cfg: dict, pol: str) -> dict[str, float]:
    geometry = cfg["geometry"]
    materials = cfg["materials"]
    sim_cfg = cfg["simulation"]
    source_cfg = cfg["source"]
    monitor_cfg = cfg["monitors"]

    period_x, period_y = [float(v) * UM for v in geometry["period_um"]]
    pillar_side = float(geometry["pillar_side_um"]) * UM
    pillar_h = float(geometry["pillar_thickness_um"]) * UM
    film_h = float(geometry["film_thickness_um"]) * UM
    substrate_h = float(geometry["substrate_thickness_um"]) * UM

    y_sub_max = 0.0
    y_film_min = 0.0
    y_film_max = film_h
    y_pillar_min = film_h
    y_pillar_max = film_h + pillar_h
    y_top = y_pillar_max
    y_source = y_top + float(source_cfg["offset_above_structure_um"]) * UM
    y_r = y_top + float(monitor_cfg["reflection_offset_um"]) * UM
    y_t = float(monitor_cfg["transmission_y_um"]) * UM
    y_min = min(-0.8 * substrate_h, y_t - 0.5 * UM)
    y_max = y_top + 2.2 * UM
    y_sub_min = y_min - 1.0 * UM

    fdtd.deleteall()
    fdtd.setglobalmonitor("frequency points", int(cfg["scan"]["wavelength_um"]["points"]))
    fdtd.setglobalmonitor("use wavelength spacing", 1)
    fdtd.setglobalmonitor("use source limits", 1)

    fdtd.addrect(
        name="substrate",
        x=0,
        z=0,
        x_span=period_x,
        z_span=period_y,
        y_min=y_sub_min,
        y_max=y_sub_max,
        material=materials["substrate"],
    )
    fdtd.addrect(
        name="film",
        x=0,
        z=0,
        x_span=period_x,
        z_span=period_y,
        y_min=y_film_min,
        y_max=y_film_max,
        material=materials["film"],
    )
    fdtd.addrect(
        name="pillar",
        x=0,
        z=0,
        x_span=pillar_side,
        z_span=pillar_side,
        y_min=y_pillar_min,
        y_max=y_pillar_max,
        material=materials["structure"],
    )
    fdtd.addfdtd(
        dimension=sim_cfg["dimension"],
        x=0,
        z=0,
        x_span=period_x,
        z_span=period_y,
        y_min=y_min,
        y_max=y_max,
        x_min_bc="Periodic",
        x_max_bc="Periodic",
        z_min_bc="Periodic",
        z_max_bc="Periodic",
        y_min_bc="PML",
        y_max_bc="PML",
        simulation_time=float(sim_cfg["simulation_time_s"]),
        PML_layers=int(sim_cfg["pml_layers"]),
        Mesh_accuracy=int(sim_cfg["mesh_accuracy"]),
    )

    mesh_override = sim_cfg.get("mesh_override", {})
    film_divisions = int(mesh_override.get("film_divisions", 0))
    pillar_divisions = int(mesh_override.get("pillar_divisions", 0))
    if pillar_divisions > 0:
        fdtd.addmesh(
            name="mesh_pillar",
            based_on_a_structure=1,
            structure="pillar",
            override_x_mesh=0,
            override_z_mesh=0,
            dy=pillar_h / pillar_divisions,
        )
    if film_divisions > 0:
        fdtd.addmesh(
            name="mesh_film",
            based_on_a_structure=1,
            structure="film",
            override_x_mesh=0,
            override_z_mesh=0,
            dy=film_h / film_divisions,
        )

    polarization_angle = 0 if pol == "TE" else 90
    wl_cfg = cfg["scan"]["wavelength_um"]
    fdtd.addplane(
        injection_axis=source_cfg.get("injection_axis", "y"),
        direction=source_cfg.get("direction", "backward"),
        polarization_angle=polarization_angle,
        angle_theta=float(cfg["scan"].get("angle_deg", 0.0)),
        x=0,
        z=0,
        x_span=period_x,
        z_span=period_y,
        y=y_source,
        wavelength_start=float(wl_cfg["start"]) * UM,
        wavelength_stop=float(wl_cfg["stop"]) * UM,
    )
    freq_points = int(wl_cfg["points"])
    fdtd.addpower(
        name="R",
        monitor_type="2D Y-Normal",
        x=0,
        z=0,
        y=y_r,
        x_span=period_x,
        z_span=period_y,
        override_global_monitor_settings=1,
        use_wavelength_spacing=1,
        frequency_points=freq_points,
    )
    fdtd.addpower(
        name="T",
        monitor_type="2D Y-Normal",
        x=0,
        z=0,
        y=y_t,
        x_span=period_x,
        z_span=period_y,
        override_global_monitor_settings=1,
        use_wavelength_spacing=1,
        frequency_points=freq_points,
    )
    return {
        "source_y_um": y_source / UM,
        "r_monitor_y_um": y_r / UM,
        "t_monitor_y_um": y_t / UM,
        "substrate_y_min_um": y_sub_min / UM,
        "substrate_y_max_um": y_sub_max / UM,
        "film_y_min_um": y_film_min / UM,
        "film_y_max_um": y_film_max / UM,
        "pillar_y_min_um": y_pillar_min / UM,
        "pillar_y_max_um": y_pillar_max / UM,
    }

