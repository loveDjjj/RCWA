from pathlib import Path

import numpy as np
import yaml


def test_load_config_reads_yaml_without_hardcoded_geometry(tmp_path):
    import rcwa_spectrum as script

    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "geometry": {
                    "period_um": [3.1, 3.2],
                    "pillar_side_um": 1.7,
                    "pillar_thickness_um": 0.5,
                    "film_thickness_um": 0.25,
                    "edge_sharpness": 800.0,
                },
                "materials": {"structure_csv": "ZnS.csv", "substrate_csv": "Si.csv"},
                "scan": {"wavelength_um": {"start": 3.0, "stop": 4.0, "points": 3}},
                "rcwa": {"fourier_order": 2, "grid": [24, 24], "batch_size": 3},
                "output": {},
            }
        ),
        encoding="utf-8",
    )

    cfg = script.load_config(config_path)

    assert cfg["geometry"]["period_um"] == [3.1, 3.2]
    assert cfg["geometry"]["pillar_side_um"] == 1.7
    assert cfg["rcwa"]["batch_size"] == 3
    assert script.stack_summary(cfg)[1] == ("layer", "patterned ZnS square pillar in air", 0.5)


def test_run_spectrum_batches_multiple_wavelengths(monkeypatch):
    import rcwa_spectrum as script

    calls = []

    def fake_simulate_batch_tensor(wavelengths_um, eps_structure, eps_substrate, angles_rad, pols, geometry, rcwa_cfg, runtime_cache, device, sim_dtype, geo_dtype):
        calls.append((tuple(wavelengths_um.cpu().numpy().tolist()), tuple(np.rad2deg(angles_rad.cpu().numpy()).tolist()), tuple(pols)))
        size = len(wavelengths_um) * len(angles_rad)
        return {
            "TE": {"R": np.full(size, 0.1), "T": np.full(size, 0.2), "A": np.full(size, 0.7)},
        }

    monkeypatch.setattr(script, "simulate_batch_tensor", fake_simulate_batch_tensor)

    table = script.run_spectrum(
        wavelengths_um=np.array([3.0, 3.5, 4.0, 4.5]),
        eps_structure=np.ones(4, dtype=np.complex128),
        eps_substrate=np.ones(4, dtype=np.complex128) * 2.25,
        pols=["TE"],
        geometry={"period_um": [2.8, 2.8]},
        rcwa_cfg={"batch_size": 2},
        runtime_cache={},
        device="cpu",
        sim_dtype=None,
        geo_dtype=None,
    )

    assert calls == [((3.0, 3.5), (0.0,), ("TE",)), ((4.0, 4.5), (0.0,), ("TE",))]
    assert table["R_TE"].tolist() == [0.1, 0.1, 0.1, 0.1]


def test_load_material_eps_respects_nm_unit(tmp_path):
    import rcwa_spectrum as script

    csv_path = tmp_path / "mat.csv"
    csv_path.write_text("wl,n,k\n3000,2.0,0\n4000,3.0,0\n", encoding="utf-8")

    eps = script.load_material_eps(csv_path, np.array([3.0, 4.0]), wavelength_unit="nm")

    np.testing.assert_allclose(eps, np.array([4.0 + 0.0j, 9.0 + 0.0j]))


def test_rayleigh_check_reports_period_match():
    import pytest
    import rcwa_spectrum as script

    wavelengths = np.linspace(2.0, 8.0, 121)

    with pytest.raises(ValueError) as exc_info:
        script.check_rayleigh_points(
            wavelengths,
            np.array([0.0]),
            {"period_um": [2.8, 2.8]},
            {"fourier_order": 1},
        )

    message = str(exc_info.value)
    assert "Rayleigh critical scan point detected" in message
    assert "wavelength[16]=2.8 um" in message
    assert "order=(-1, 0)" in message or "order=(1, 0)" in message


def test_rayleigh_check_allows_pso_300_point_grid():
    import rcwa_spectrum as script

    wavelengths = np.linspace(2.0, 8.0, 300)

    script.check_rayleigh_points(
        wavelengths,
        np.array([0.0]),
        {"period_um": [2.8, 2.8]},
        {"fourier_order": 5},
    )


def test_run_spectrum_computes_te_and_tm_in_one_batch(monkeypatch):
    import rcwa_spectrum as script

    calls = []

    def fake_simulate_batch_tensor(wavelengths_um, eps_structure, eps_substrate, angles_rad, pols, geometry, rcwa_cfg, runtime_cache, device, sim_dtype, geo_dtype):
        calls.append(tuple(pols))
        size = len(wavelengths_um) * len(angles_rad)
        return {
            "TE": {"R": np.full(size, 0.1), "T": np.full(size, 0.2), "A": np.full(size, 0.7)},
            "TM": {"R": np.full(size, 0.3), "T": np.full(size, 0.4), "A": np.full(size, 0.3)},
        }

    monkeypatch.setattr(script, "simulate_batch_tensor", fake_simulate_batch_tensor)
    table = script.run_spectrum(
        wavelengths_um=np.array([3.0, 3.5]),
        eps_structure=np.ones(2, dtype=np.complex128),
        eps_substrate=np.ones(2, dtype=np.complex128) * 2.25,
        pols=["TE", "TM"],
        geometry={"period_um": [2.8, 2.8]},
        rcwa_cfg={"batch_size": 2},
        runtime_cache={},
        device="cpu",
        sim_dtype=None,
        geo_dtype=None,
    )

    assert calls == [("TE", "TM")]
    assert table["R_TE"].tolist() == [0.1, 0.1]
    assert table["R_TM"].tolist() == [0.3, 0.3]


def test_run_angle_sweep_reuses_zero_degree_spectrum(monkeypatch):
    import rcwa_spectrum as script

    calls = []

    def fake_simulate_batch_tensor(wavelengths_um, eps_structure, eps_substrate, angles_rad, pols, geometry, rcwa_cfg, runtime_cache, device, sim_dtype, geo_dtype):
        calls.append(tuple(np.round(np.rad2deg(angles_rad.cpu().numpy()), 12).tolist()))
        size = len(wavelengths_um) * len(angles_rad)
        return {
            "TE": {"R": np.full(size, 0.9), "T": np.full(size, 0.05), "A": np.full(size, 0.05)},
        }

    monkeypatch.setattr(script, "simulate_batch_tensor", fake_simulate_batch_tensor)
    spectrum = {
        "wavelength_um": np.array([3.0, 4.0]),
        "R_TE": np.array([0.1, 0.2]),
        "T_TE": np.array([0.3, 0.4]),
        "A_TE": np.array([0.6, 0.4]),
    }
    table = script.run_angle_sweep(
        wavelengths_um=np.array([3.0, 4.0]),
        angles_deg=np.array([0.0, 30.0]),
        eps_structure=np.ones(2, dtype=np.complex128),
        eps_substrate=np.ones(2, dtype=np.complex128) * 2.25,
        pols=["TE"],
        geometry={"period_um": [2.8, 2.8]},
        rcwa_cfg={"batch_size": 2},
        runtime_cache={},
        device="cpu",
        sim_dtype=None,
        geo_dtype=None,
        spectrum_table=spectrum,
    )

    assert calls == [(30.0,)]
    assert table["R_TE"].reshape(2, 2)[:, 0].tolist() == [0.1, 0.2]
    assert table["R_TE"].reshape(2, 2)[:, 1].tolist() == [0.9, 0.9]


def test_run_angle_sweep_splits_angles_by_max_batch_elements(monkeypatch):
    import rcwa_spectrum as script

    calls = []

    def fake_simulate_batch_tensor(wavelengths_um, eps_structure, eps_substrate, angles_rad, pols, geometry, rcwa_cfg, runtime_cache, device, sim_dtype, geo_dtype):
        calls.append((tuple(wavelengths_um.cpu().numpy().tolist()), tuple(np.round(np.rad2deg(angles_rad.cpu().numpy()), 12).tolist())))
        size = len(wavelengths_um) * len(angles_rad)
        return {
            "TE": {"R": np.arange(size), "T": np.arange(size) + 10, "A": np.arange(size) + 20},
        }

    monkeypatch.setattr(script, "simulate_batch_tensor", fake_simulate_batch_tensor)
    script.run_angle_sweep(
        wavelengths_um=np.array([3.0, 4.0]),
        angles_deg=np.array([10.0, 20.0, 30.0]),
        eps_structure=np.ones(2, dtype=np.complex128),
        eps_substrate=np.ones(2, dtype=np.complex128) * 2.25,
        pols=["TE"],
        geometry={"period_um": [2.8, 2.8]},
        rcwa_cfg={"batch_size": 2, "max_batch_elements": 4},
        runtime_cache={},
        device="cpu",
        sim_dtype=None,
        geo_dtype=None,
    )

    assert calls == [((3.0, 4.0), (10.0, 20.0)), ((3.0, 4.0), (30.0,))]


def test_main_respects_run_switches(monkeypatch, tmp_path):
    import rcwa_spectrum as script

    cfg = {
        "geometry": {
            "period_um": [2.8, 2.8],
            "pillar_side_um": 1.5,
            "pillar_thickness_um": 0.6,
            "film_thickness_um": 0.3,
        },
        "materials": {"structure_csv": "ZnS.csv", "substrate_csv": "Si.csv"},
        "scan": {
            "wavelength_um": {"start": 3.0, "stop": 4.0, "points": 2},
            "angle_deg": {"start": 0.0, "stop": 20.0, "points": 2},
            "polarization": "TE",
        },
        "rcwa": {"fourier_order": 1, "grid": [8, 8], "batch_size": 2},
        "runtime": {"cpu": True, "run_spectrum": False, "run_angle_sweep": True},
        "output": {"angle_csv": str(tmp_path / "angle.csv"), "no_plots": True},
    }
    calls = []

    monkeypatch.setattr(
        script,
        "parse_args",
        lambda: type(
            "Args",
            (),
            {
                "config": "unused.yaml",
                "no_plots": True,
                "no_spectrum": False,
                "no_angle_sweep": False,
                "angle_shards": 1,
                "angle_shard_index": 0,
                "output_suffix": None,
                "merge_angle_shards": 0,
                "run_angle_shards": 0,
                "keep_angle_shards": False,
            },
        )(),
    )
    monkeypatch.setattr(script, "load_config", lambda path: cfg)
    monkeypatch.setattr(script, "load_material_eps", lambda path, wavelengths, unit="um": np.ones_like(wavelengths, dtype=np.complex128))
    monkeypatch.setattr(script, "run_spectrum", lambda *args, **kwargs: calls.append("spectrum"))

    def fake_run_angle_sweep(*args, **kwargs):
        calls.append("angle")
        return {"wavelength_um": np.array([3.0]), "angle_deg": np.array([0.0]), "R_TE": np.array([0.1]), "T_TE": np.array([0.2]), "A_TE": np.array([0.7])}

    monkeypatch.setattr(script, "run_angle_sweep", fake_run_angle_sweep)

    script.main()

    assert calls == ["angle"]
    assert Path(cfg["output"]["angle_csv"]).exists()


def test_select_angle_shard_splits_configured_angles():
    import rcwa_spectrum as script

    angles = np.arange(0.0, 6.0)

    assert script.select_angle_shard(angles, 2, 0).tolist() == [0.0, 1.0, 2.0]
    assert script.select_angle_shard(angles, 2, 1).tolist() == [3.0, 4.0, 5.0]
    assert script.angle_shard_suffix(2, 1, None) == "angle_shard_1_of_2"
    assert script.suffixed_path("angle.csv", "angle_shard_1_of_2") == "angle_angle_shard_1_of_2.csv"


def test_merge_angle_shards_sorts_by_wavelength_then_angle(tmp_path, monkeypatch):
    import rcwa_spectrum as script

    base = tmp_path / "angle.csv"
    shard0 = tmp_path / "angle_angle_shard_0_of_2.csv"
    shard1 = tmp_path / "angle_angle_shard_1_of_2.csv"
    shard0.write_text(
        "wavelength_um,angle_deg,R_TE,T_TE,A_TE\n"
        "4,0,0.4,0.5,0.1\n"
        "3,0,0.3,0.6,0.1\n",
        encoding="utf-8",
    )
    shard1.write_text(
        "wavelength_um,angle_deg,R_TE,T_TE,A_TE\n"
        "4,30,0.7,0.2,0.1\n"
        "3,30,0.6,0.3,0.1\n",
        encoding="utf-8",
    )
    cfg = {"output": {"angle_csv": str(base), "angle_fig": str(tmp_path / "angle.png"), "no_plots": True}}

    script.merge_angle_shards(cfg, 2, ["TE"])

    merged = np.genfromtxt(base, delimiter=",", names=True)
    assert merged["wavelength_um"].tolist() == [3.0, 3.0, 4.0, 4.0]
    assert merged["angle_deg"].tolist() == [0.0, 30.0, 0.0, 30.0]


def test_run_angle_shards_launches_children_and_merge(monkeypatch, tmp_path):
    import rcwa_spectrum as script

    commands = []
    removed = []

    class FakeProcess:
        def __init__(self, cmd):
            self.cmd = cmd

        def wait(self):
            return 0

    def fake_popen(cmd):
        commands.append(cmd)
        return FakeProcess(cmd)

    def fake_run(cmd, check):
        commands.append(cmd)
        assert check is True

    cfg = {"output": {"angle_csv": str(tmp_path / "angle.csv")}}
    for idx in range(2):
        shard = tmp_path / f"angle_angle_shard_{idx}_of_2.csv"
        shard.write_text("wavelength_um,angle_deg,R_TE,T_TE,A_TE\n", encoding="utf-8")

    monkeypatch.setattr(script.subprocess, "Popen", fake_popen)
    monkeypatch.setattr(script.subprocess, "run", fake_run)
    monkeypatch.setattr(script, "load_config", lambda path: cfg)
    monkeypatch.setattr(Path, "unlink", lambda self: removed.append(self))

    args = type(
        "Args",
        (),
        {
            "run_angle_shards": 2,
            "config": "config.yaml",
            "no_plots": True,
            "keep_angle_shards": False,
        },
    )()

    script.run_angle_shards(args)

    assert len(commands) == 3
    assert commands[0][-2:] == ["--angle_shard_index", "0"]
    assert "--no_spectrum" not in commands[0]
    assert commands[1][-3:] == ["--angle_shard_index", "1", "--no_spectrum"]
    assert "--merge_angle_shards" in commands[2]
    assert len(removed) == 2
