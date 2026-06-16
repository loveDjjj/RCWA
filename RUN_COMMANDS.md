# 运行命令

所有命令都在项目根目录下运行：

```powershell
cd "O:\Optics Code\RCWA"
```

## 单角度光谱

在 `configs/rcwa/spectrum.yaml` 中设置：

```yaml
runtime:
  run_spectrum: true
  run_angle_sweep: false
```

然后运行：

```powershell
python rcwa_spectrum.py --config configs\rcwa\spectrum.yaml
```

如果只需要输出数据、不生成图：

```powershell
python rcwa_spectrum.py --config configs\rcwa\spectrum.yaml --no_plots
```

正式光谱输出默认写入：

```text
outputs/rcwa/
```

## 多角度光谱

在 `configs/rcwa/spectrum.yaml` 中设置：

```yaml
runtime:
  run_angle_sweep: true
```

然后运行：

```powershell
python rcwa_spectrum.py --config configs\rcwa\spectrum.yaml
```

如果需要把角度扫描拆成多个子任务并行运行：

```powershell
python rcwa_spectrum.py --config configs\rcwa\spectrum.yaml --run_angle_shards 2 --no_plots
```

## PSO 结构优化

```powershell
python rcwa_pso_optimize.py --config configs\pso\pso.yaml
```

输出目录由 `output.run_dir` 控制，默认写入 `outputs/pso/reflection`。

当前默认 PSO 波长网格为 2 到 8 um、共 300 个采样点，用于避开 `period_um = 2.8 um` 对应的 Rayleigh 临界点。

如果需要诊断 GPU 利用率或每个 batch 的耗时分布，可在 `configs/pso/pso.yaml` 中临时打开：

```yaml
runtime:
  profile: true
```

程序会在每代 RCWA 后输出 `index`、`rcwa_init`、`layers`、`solve`、`powers`、`assign` 等阶段耗时。诊断结束后建议改回 `false`，避免频繁 `cuda synchronize` 影响速度。

## FDTD 结构验证

使用根目录入口：

```powershell
python run_fdtd.py
```

轻量 smoke 验证：

```powershell
python run_fdtd.py --defaults tests\fixtures\fdtd\defaults_smoke.yaml --structure tests\fixtures\fdtd\structure_smoke.yaml
```

## Smoke 检查

```powershell
python rcwa_spectrum.py --config tests\fixtures\rcwa\spectrum_smoke.yaml --no_plots
python rcwa_pso_optimize.py --config tests\fixtures\pso\pso_smoke.yaml
python -m pytest -q
```
