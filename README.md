# 科学模型多端出图验证工具

让**同一确定场景**分别经过 CPU 参考路径与可用的 GPU/向量化渲染路径，输出
**浮点颜色（线性 HDR + sRGB）、深度、对象标识**，再比较实际差异。全部后端
计算仅依赖 Python 3 标准库，无需安装第三方包。

## 两条真实不同的计算实现（本环境始终可运行）

| 设备 ID | 实现 | 精度 | 遍历方式 | 填充约定 |
|---|---|---|---|---|
| `cpu_ref` | CPU 参考光栅化 | 原生 float64 标量 | 逐三角形包围盒扫描 | 全包含（所有边 ≥0） |
| `vector_f32` | 向量化计算路径 | 每步舍入到 binary32 | 8×8 分块、逐像素测试所有三角形（GPU SIMD 形态） | GPU top-left（非 top-left 边严格 >0） |
| `gpu_cuda` | 真实 GPU 路径 | float32 | 仅当检测到 GPU 硬件 **且** 存在 CUDA 计算模块时启用 | top-left |

两路径**共享完全相同**的相机矩阵、右手坐标系、透视校正插值、过程纹理双线性
采样与 `Blinn-Phong → Reinhard → sRGB` 颜色变换；差异只来自光栅化遍历顺序、
f32/f64 运算舍入和填充约定，不注入任何脚本化误差。

### 缺失设备时的诚实策略
`backend/raster/devices.py` 真实探测 `nvidia-smi` 与 CUDA Python 模块。探测不到
GPU 时：`gpu_cuda` 在界面上禁用、API 返回 `skipped`、**绝不生成 GPU 图像**，并
在覆盖说明中写明“未执行”。本仓库当前环境无 GPU，因此实际执行的是两条 CPU 实现。

## 怎样启动界面

```bash
# 无需 pip install，标准库即可
python3 backend/server.py            # 默认 http://127.0.0.1:8000
python3 backend/server.py 0.0.0.0 9000
```

浏览器打开 `http://127.0.0.1:8000`：
1. 在**出图任务面板**选择测试场景、分辨率与设备（GPU 不可用时为禁用态）；
2. 点击 **渲染对照**，按钮直接 `POST /api/render`，后端实时计算；
3. 结果区显示每个设备的 sRGB / 线性 HDR / 深度 / 对象标识图，以及颜色绝对误差、
   相对误差、深度误差、边缘分布、覆盖差异五张诊断通道；
4. 点击任一设备图像上的像素，触发带 `trace_pixel` 的真实重算，查看该像素两条
   路径的逐三角形边权重、候选深度、着色浮点输入与最终值。

界面里没有任何固定图像或误差数据——所有内容均来自后端响应。

## 无头运行（不启动界面也能复现）

```bash
python3 -m backend.cli --scene all --width 128 --out reports/run
python3 -m backend.cli --scene sliver --width 192 --trace 96,96 --out reports/trace
```

产出每场景每设备的 PNG、五张诊断 PNG、`*_report.json` 与 `summary.json`。

## 差异如何分类（不把误差都甩给“硬件不同”）

`backend/raster/compare.py` 区分两类：

- **像素对应错误（structural）**：覆盖掩码不一致（参考独有/被测独有）、对象
  标识不同、或深度差超过硬容差——属于“这个像素归谁”的几何/约定问题；
- **数值精度偏差（precision_only）**：两路径覆盖与对象标识一致，仅浮点值在
  f32/f64 可解释范围内不同。

统计口径：
- 仅在**模型有效区域**（两路径覆盖并集）统计；背景零值像素单独计数并排除，
  不能用大片黑背景稀释模型误差；
- 绝对误差、相对误差（暗部以 eps 下限归一）、均值/最大/P99/RMSE；
- 非有限值（NaN/Inf）单独收集；
- 边缘带（8 邻域覆盖变化）与内部分桶统计，定位边缘归属问题。

容差及来源见 `backend/raster/tolerance.py`（如 sRGB 半量化步长 `0.5/255`、
深度硬限 `1e-4`），界面“容差来源”区块逐条展示。

## 四个测试场景

| key | 针对的失败模式 |
|---|---|
| `depth_near` | 远距双平面深度差约 9e-4，深度缓冲精度与近重合表面 |
| `highlight` | 高光泽强镜面尖峰，微小插值差异移动高光位置 |
| `sliver` | 26 个窄楔形 + 针状三角，亚像素覆盖与边缘函数 |
| `hdr_gamut` | 线性光最高 3.0，经色调映射压回 [0,1]，检验越界钳制 |

## 测试

```bash
python3 -m unittest tests.test_pipeline -v
```

覆盖 f32 舍入、四场景双路径执行、GPU 不伪造、确定性复现、背景排除、结构/精度
分类、非有限值检测、像素轨迹与 PNG 编码。

## 复现配置
每次任务返回相机、光照、曝光、纹理、坐标/填充约定、Python 与平台信息，并对该
配置计算 `sha256`；相同输入两次渲染的浮点结果逐位一致。

## 目录
```
backend/raster/   几何/场景/两条光栅化路径/对比/容差/设备探测/PNG 编码/服务
backend/server.py 标准库 HTTP + REST API + 前端托管
backend/cli.py    无头出图与报告
frontend/         出图任务面板（原生 HTML/CSS/JS，调用真实 API）
tests/            单元测试
reports/run/      已生成的真实渲染与诊断产物
```
