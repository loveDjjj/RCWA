# RCWA

本项目提供基于 PyTorch 的批量 RCWA 计算流程，用于周期超表面光谱计算和粒子群结构优化。

## 主要功能

1. 对给定结构计算单角度光谱：
   `R/T/A` 随波长变化。
2. 对给定结构计算多角度光谱：
   `R/T/A` 随波长和入射角变化。
3. 使用粒子群算法优化结构：
   以 particle by wavelength 的方式批量调用 RCWA。

## 项目结构

```text
RCWA/
  rcwa_spectrum.py          # 单角度光谱和角度扫描入口
  rcwa_pso_optimize.py      # PSO 结构优化入口
  run_fdtd.py               # FDTD 验证入口
  torch_rcwa/               # 本地批量 RCWA 实现
  fdtd/                     # FDTD 共享框架
  fdtd/templates/           # FDTD 模板工程
  configs/
    rcwa/
      spectrum.yaml         # 正式 RCWA 光谱配置
    pso/
      pso.yaml              # 正式 PSO 配置
    fdtd/
      defaults.yaml         # FDTD 默认配置
      structure.yaml        # FDTD 结构配置
  database/                 # 材料 n/k CSV 数据
  outputs/                  # RCWA/PSO/FDTD 输出
  tests/
    fixtures/               # smoke/调试测试配置
```

## 配置说明

给定结构的光谱计算使用 `configs/rcwa/spectrum.yaml`。材料字段使用通用命名，不再绑定具体材料名：

```yaml
materials:
  structure_name: ZnS
  structure_csv: database/ZnS.csv
  substrate_name: Si
  substrate_csv: database/Si.csv
  csv_wavelength_unit: um
```

光谱计算和 PSO 优化在启动时都会执行 Rayleigh 临界点检查。如果波长网格精确命中临界衍射条件，程序会在 RCWA 求解前停止，并报告波长索引、入射角和衍射阶次。长时间任务开始前，应通过微调波长网格或修改采样点数避开这些临界点。

PSO 优化使用 `configs/pso/pso.yaml`。粒子编码为：

```text
[fill_factor, pillar_thickness_um, film_thickness_um, structure_material_idx, film_material_idx]
```

周期由配置文件固定，不放入粒子变量中，这样可以保持 particle by wavelength 的批量 RCWA 计算效率。

## 快速开始

可直接复制运行的命令见 [RUN_COMMANDS.md](RUN_COMMANDS.md)。

FDTD 重构后的说明见 [fdtd/README.md](</O:/Optics Code/RCWA/fdtd/README.md>)。
