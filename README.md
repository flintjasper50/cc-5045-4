# 科学模型多端出图验证工具

让**同一确定场景**分别经过 CPU 参考路径与可用的 GPU/向量化渲染路径，输出
**浮点颜色（线性 HDR + sRGB）、深度、对象标识**，再按**版本化的分项验收规则**
比较实际差异。全部后端计算仅依赖 Python 3 标准库，无需安装第三方包。

> 图形入口就是本工具的主入口：启动服务后浏览器打开首页的**出图任务面板**。
> 命令行（`backend/cli.py`）仅作为批量/CI 辅助，见下文“无头运行”。

## 图形入口（主要使用方式）

```bash
python3 backend/server.py            # 默认 http://127.0.0.1:8000
python3 backend/server.py 0.0.0.0 9000
./run.sh                             # 等价快捷方式
```

浏览器打开 `http://127.0.0.1:8000`，在**出图任务面板**中：

1. 选择测试场景、分辨率与**每个设备的能力/精度配置**（GPU 不可用时为禁用态）；
   预设下拉可一键切换“两种精度配置对照 / 严格 / 快速数学 / 降级组合”；
2. 点击 **渲染对照**，按钮直接 `POST /api/render`，后端实时计算；
3. 结果区先显示**运行状态**：每个设备是“已执行 / 配置不支持 / 未执行”，
   并单独列出各设备**图像缺失、非有限值、分项失败**等异常；
4. 随后显示每个设备的 sRGB / 线性 HDR / 深度 / 对象标识图，五张诊断通道，
   外加**归因阶段图**（每个不同像素始于哪个管线阶段）；
5. **差异归因**表给出首个差异阶段、最小复现窗口与可直接复制的复现命令；
6. **分项验收规则**表逐操作列出失败类、允许偏差、有效像素与来源；
7. **尚未验证的设备 × 配置组合**单独列出——绿色只代表本次真正执行过的组合；
8. 点击任一设备图像上的像素，触发带 `trace_pixel` 的真实重算，查看该像素两条
   路径的逐三角形边权重、候选深度、反照率/着色浮点输入与最终值。

界面里没有任何固定图像或误差数据——所有内容均来自后端响应。

## 两条真实不同的计算实现（本环境始终可运行）

| 设备 ID | 实现 | 默认精度 | 遍历方式 | 填充约定 |
|---|---|---|---|---|
| `cpu_ref` | CPU 参考光栅化 | 原生 float64 标量 | 逐三角形包围盒扫描 | 全包含（所有边 ≥0） |
| `vector_f32` | 向量化计算路径 | 每步舍入到 binary32 | 8×8 分块、逐像素测试所有三角形（GPU SIMD 形态） | GPU top-left |
| `gpu_cuda` | 真实 GPU 路径 | float32 | 仅当检测到 GPU 硬件 **且** 存在 CUDA 计算模块时启用 | top-left |

两路径**共享完全相同**的相机矩阵、右手坐标系、透视校正插值、过程纹理采样与
`Blinn-Phong → Reinhard → sRGB` 颜色变换；差异只来自光栅化遍历顺序、f32/f64
运算舍入、填充约定，以及**配置档显式声明的能力**（见下节），不注入脚本化误差。

### 缺失设备时的诚实策略
`backend/raster/devices.py` 真实探测 `nvidia-smi` 与 CUDA Python 模块。探测不到
GPU 时：`gpu_cuda` 在界面上禁用、API 返回 `skipped`、**绝不生成 GPU 图像**，并在
覆盖说明中写明“未执行”。本仓库当前环境无 GPU，因此实际执行的是两条 CPU 实现。

## 能力 / 精度配置档（可追溯）

`backend/raster/profiles.py` 把影响像素的能力做成**不可变配置档**，每个字段都带
来源说明，解析后计算 `sha256` 并写入报告：

| 配置档 | 精度 | 快速数学 | 纹理格式 | 混合写回 | 深度约定 |
|---|---|---|---|---|---|
| `reference_f64` | f64 | 关 | float32 双线性 | 浮点写回 | [-1,1] |
| `strict_f32` | f32 | 关 | float32 双线性 | 浮点写回 | [-1,1] |
| `fast_math_f32` | f32 | 开 | float32 双线性 | 浮点写回 | [-1,1] |
| `rgba8_texture_f32` | f32 | 关 | **rgba8 线性格点** | 浮点写回 | [-1,1] |
| `fixed8_blend_f32` | f32 | 关 | float32 双线性 | **8 位线性写回** | [-1,1] |
| `degraded_rgba8_fixed8_f32` | f32 | 开 | **rgba8** | **8 位写回** | [-1,1] |
| `f32_ndc01_depth`（仅诊断） | f32 | 关 | float32 双线性 | 浮点写回 | **0..1（刻意错误）** |

- **设备声明能力集合**，只有请求档的每个字段都在设备能力内才执行；否则报告
  `unsupported_capability` 并写明缺哪些字段，**绝不静默降级替代**。
- 例如 `gpu_cuda` 只声明 `float32_bilinear`/`float_blend`，在它上面请求 rgba8
  纹理会被标为不支持，而不会偷偷换成浮点纹理再宣称“已验证”。
- 快速数学是**显式声明的近似通路**（指数近似 + 重结合点积），不是隐藏系数。

## 验收规则（初版逐像素继续保留，新增规则版本化）

规则与来源见 `backend/raster/tolerance.py`。版本字符串 `acceptance-2026.11-v2`
与完整规则文本**冻结写入每份报告**，因此后续调整阈值不会改写旧报告结论。

三层口径互相独立：

1. **初版逐像素 v1（保留）**：模型有效区域的绝对/相对误差，背景零值像素单独
   计数并排除。仍在每次比较中执行并产出热图，但**不单独决定通过**。
2. **差异归因**：按管线顺序为每个不同像素标注**首个差异阶段**：
   `non_finite → coverage → object_id → depth → texture_sampling → shading →
   blend_write → color_mapping`。给出最小复现窗口、复现命令，并记录该像素的
   反照率 / 线性 HDR / sRGB / 深度中间通道——可直接判断差异始于变换、采样还是
   最终颜色映射。
3. **分项验收（决定结论）**：每项操作声明**有效像素集合**与允许偏差：

| 操作 | 有效像素 | 允许偏差 | 失败类 |
|---|---|---|---|
| `coverage` | 模型并集 | 0（不允许多/少一个模型像素） | `coverage_struct` |
| `object_id` | 共同覆盖 | 0 | `object_id` |
| `depth` | 共同覆盖且有限 | 内部/边缘均 ≤1e-4（NDC） | `depth_convention` / `depth_correspondence` |
| `final_color_inherited` | 已被上游同一对象覆盖 | 内 0.5/255、边 1/255、rel ≤2% | `precision_exceeded` |
| `final_color_primary` | 本设备首次引入 | 同上 | `precision_exceeded` |
| `blend_write_quantized` | 被测档声明 `fixed8_blend` | ≤4/255、rel ≤5% | `blend_quantization_exceeded` |
| `linear_hdr` | 共同覆盖且对象一致 | rel ≤2% | `precision_exceeded` |
| `finite` | 全部输出 | 非有限值 ≤0 | `non_finite` |

**结构问题不会被宽松颜色阈值吞掉**：覆盖缺失/新增、对象标识不一致、深度约定
错误、通道交换、非有限值各自是独立的失败类，并带有自己的像素集合；不会把
“缺失几何”“换通道”“NaN”混进一个全局 max 里。

两个**显式结构探测**：
- **通道交换探测**：若某个通道置换把平均误差降到 0.25/255 以下、且比原始误差
  改善 ≥8 倍，而原始误差 >2/255，则判 `channel_swap`，而不是数值噪声；
- **深度约定探测**：深度差普遍 >1e-3 且同向比例 >0.9，判 `depth_convention`。

判定策略是“按操作分别判定、逐项失败”：**不会**在看到失败后自动放宽阈值直到全绿，
也**不要求**所有硬件天然逐位相等——允许偏差是按操作和像素类别**事先**声明的。

## 无头运行（不启动界面也能复现）

命令行**仅辅助**多端出图；图形入口见上文“出图任务面板”。

```bash
# 默认：两条真实计算路径，四个场景，产出 PNG + 报告
python3 -m backend.cli --scene all --width 128 --out reports/run

# 交付两种精度配置的真实渲染对照（f64 参考 vs f32、f32 vs f32），
# 外加跨设备同配置用例（同一 strict_f32 在两套实现上仍会因填充约定不同）
python3 -m backend.cli --compare-configs --scene all --width 128 --out reports/precision

# 指定能力配置档（写入报告）并追踪一个像素
python3 -m backend.cli --scene depth_near --width 128 \
    --profile cpu_ref=reference_f64 --profile vector_f32=f32_ndc01_depth \
    --trace 64,64 --out reports/profiles

# 自检：注入通道交换 / 深度约定错误 / 非有限值，验证分类可区分
# 默认 --scene all 会对四个场景各跑一遍四类自检
python3 -m backend.cli --self-check --width 96 --out reports/selfcheck
```

已提交的真实产物：
- `reports/run/`：四场景 × 两设备图像 + 五张诊断图 + 归因阶段图 + 报告；
- `reports/precision/`：两种精度配置对照 + 跨设备同配置用例；
- `reports/selfcheck/`：正常舍入（通过）、通道交换、深度约定错误、非有限值四类
  结论，证明分类与失败结论可区分；
- `reports/trace/`：带像素轨迹复现。

## 差异如何分类（不把误差都甩给“硬件不同”）

`backend/raster/compare.py` 输出三类信息：

- **归因**：每个不同像素的**首个差异阶段** + 最小复现窗口 + 中间通道样本；
- **分项失败**：`coverage_struct` / `object_id` / `depth_convention` /
  `depth_correspondence` / `channel_swap` / `non_finite` /
  `blend_quantization_exceeded` / `precision_exceeded`；
- **v1 统计**：绝对/相对误差的均值、最大、P99、RMSE，按内部/边缘分桶；仅在
  **模型有效区域**统计，背景零值单独计数，不能用大片黑背景稀释模型误差。

## 四个测试场景

| key | 针对的失败模式 |
|---|---|
| `depth_near` | 远距双平面深度差约 9e-4（测得实际 NDC 差约 4.4e-7，f32 可分辨），近重合表面与深度缓冲精度；外圈保留可纹理区域以暴露纹理格式差异 |
| `highlight` | 高光泽强镜面尖峰，微小插值差异移动高光位置；8 位线性写回在此产生可见偏差 |
| `sliver` | 26 个窄楔形 + 针状三角，亚像素覆盖与边缘函数 |
| `hdr_gamut` | 线性光最高 3.0，经色调映射压回 [0,1]，检验越界钳制 |

## 测试

```bash
python3 -m unittest tests.test_pipeline -v
```

覆盖面包括：f32 舍入、四场景双路径执行、GPU 不伪造、确定性复现、背景排除、
**正常舍入通过 / 刻意深度约定错误归为 `depth_convention` / 通道交换单独识别 /
非有限值不被吞掉 / 缺失像素判为结构错误**、配置档能力不支持时拒绝执行而不降级、
规则版本冻结不随阈值改动而变、归因窗口与中间通道、PNG 编码。

## 复现配置

每次任务返回相机、光照、曝光、纹理、坐标/填充约定、Python 与平台信息、本次所用
每个配置档的 `sha256`，以及 `rules_version`；对输入计算 `config_sha256`。相同输入
两次渲染的浮点结果逐位一致。

## 跨设备比较用例与核验

- **同配置跨实现**（`fill_convention_same_profile`）：`cpu_ref` 与 `vector_f32`
  都设为 `strict_f32` 在 `depth_near` 128×128 上比较。因遍历顺序/填充约定不同，
  剪影归属仍会分歧，应判 `coverage_struct`/`object_id` 而非数值误差——用来说明
  “不是所有硬件天然逐位相等”，且结构差异必须被单独识别。
- **两种精度配置对照**（`--compare-configs`）：f64 参考 vs f32、以及 f32 vs f32，
  分别真实渲染并各自给出结论。
- **浮点路径与设备差别核验颜色误差及可复现性**：报告记录每条路径的精度/快速数学/
  纹理格式/写回/深度约定，以及 `elapsed_ms`、中间通道与 `config_sha256`。

## 尚未验证 / 不做什么

- 未声明能力的组合不执行、不替代，界面与报告都列为“尚未验证”；
- 无 GPU 时 `gpu_cuda` 全部组合标为未验证，不生成图像；
- 不采用“看到失败就放宽阈值直到全绿”；阈值按操作事先声明、版本化冻结；
- 不要求所有硬件逐位相等——差异分类后才判断是否可接受。

## 目录
```
backend/raster/   profiles(能力/精度档) 几何/场景/两条光栅化路径/归因与验收/容差/
                  设备探测与能力声明/PNG 编码/服务/JSON 安全序列化
backend/server.py 标准库 HTTP + REST API + 前端托管
backend/cli.py    无头出图与报告（辅助入口）
frontend/         出图任务面板（原生 HTML/CSS/JS，调用真实 API）
tests/            单元测试
reports/run/      四场景真实渲染与诊断产物
reports/precision/ 两种精度配置对照 + 跨设备同配置用例
reports/selfcheck/ 分类可区分性自检
```
