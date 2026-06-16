# FDTD Framework

本目录提供当前项目使用的 FDTD 共享框架，目标是把结构建模、光谱提取、画图和 YAML 配置分离开。

## 目录职责

`lumapi_loader.py`
- 加载 Lumerical `lumapi`

`builder.py`
- 根据 YAML 配置在模板 `.fsp` 上重建结构

`extract.py`
- 提取 `R/T/A` 和原始 monitor 通量

`plotting.py`
- 保存光谱 CSV 和 PNG

`runner.py`
- FDTD 统一运行入口

## 配置位置

FDTD 配置统一放在：

```text
configs/fdtd/
```

当前已有：

- `defaults.yaml`
- `structure.yaml`

模板工程统一放在：

```text
fdtd_templates/
```

## 运行方式

正式运行：

```powershell
python run_fdtd.py
```

轻量 smoke 验证：

```powershell
python run_fdtd.py --defaults tests\fixtures\fdtd\defaults_smoke.yaml --structure tests\fixtures\fdtd\structure_smoke.yaml
```

其中 smoke 配置使用：

```yaml
runtime:
  smoke_mode: build_only
```

这个模式只验证：

- 模板 `.fsp` 能否打开
- YAML 能否正确驱动建模
- `.fsp` 和 `case_metadata.json` 能否导出

它不会执行 `fdtd.run()`，因此适合作为入口级验证。
