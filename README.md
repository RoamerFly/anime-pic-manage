# Anime Pic Manage

Windows 本地优先的动漫图片管理工具：**角色识别、相似度去重、分类整理、LoRA 训练集导出、AI 生图与 LoRA 训练联动**，图片始终留在本机，不上传云端。

设计原则是「本地优先 + 人工确认 + 不破坏文件」：识别与扫描只产出结果和计划，移动/删除必须先看到计划并确认；ComfyUI 与 kohya 都是**可选外置**，不装也不影响识别。

## 快速开始

### 用便携包（推荐）

1. 解压 `dist_windows_gpu`（显卡版）或 `dist_windows`（CPU 版，体积更小）
2. 双击根目录的 **`启动.bat`**（或直接双击 `anime-pic-manage.exe`）
3. 按 `使用说明.txt` 走一遍：选图库目录 → 识别 → 人工矫正

便携包自带 Python、推理引擎与模型，**目标机器不需要安装 Python / CUDA / 任何运行库**：

```text
AnimePicManage\
├─ 启动.bat / anime-pic-manage.exe / 使用说明.txt   ← 根目录只有这三个入口
├─ app\runtime\ai-worker.exe   内置推理引擎（不要删）
├─ app\env\                    兼容运行环境（可删，省约 1.3GB）
├─ app\cuda\                   显卡运行库：在设置页一键下载后出现在这里
├─ models\recognizer\...       识别模型（可切换、可下载）
├─ resources\                  角色集合、生图工作流模板、模型清单
├─ data\                       数据库、扫描结果、参考库、日志 ← 备份只需备份这一个
├─ output\{generated,loras,datasets}   出图 / LoRA / 训练集
└─ temp\
```

### 从源码运行

```powershell
pnpm install
pnpm dev                                                   # 浏览器预览前端
pnpm --filter @anime-pic-manage/desktop tauri:dev           # 桌面端（含 Rust 核心与 Worker）
```

## 功能与用法

### 1. 角色识别（识别 + 人工矫正）

1. 「角色识别」→ 选择图片文件夹（支持子目录）
2. 点「开始识别」，进度与结果实时刷新；扫描结果写入 `data\scan-results`，**随时可中断、暂停、续跑**
3. 点开某张图，在画布上拖动/缩放人物框，选择正确角色（或标记「未识别」），保存即写入标注库
4. 画布支持框选、拖动改框、缩放与滚轮缩放；框与角色名都可以覆盖模型判断

识别结果分三类：**高置信 / 待复核 / 未识别**。只有高置信会直接采用，其余都需要你确认——这是后面所有流程的数据来源。

### 2. 参考图识别（模型标签表里没有的角色）

识别模型的标签表决定它能认出谁。像玖辛奈这类冷门角色不在默认模型的 12,476 个标签里，无论怎么扫描都是「未识别」。参考图识别就是为这种情况准备的：

1. 先人工矫正一批该角色的图（10~30 张足够起步）
2. 设置 → 识别配置 → **参考图识别**：选后端（CCIP 推荐 / embedding 免下载）→ 点「用矫正结果建立参考库」
3. 之后扫描时，应用只对**非高置信**的人物做相似度匹配，命中会给出「参考匹配 9x%」并归入待复核

实测对比（48 张玖辛奈参考库 + 其他角色对照）：CCIP 判别力 AUC **1.000**，识别模型自身 embedding 0.951~0.979。参考库存在 `data\reference-library.json`，可随时删除重建。

### 3. 相似度去重

1. 「相似度识别」→ 选择目录 → 设定阈值（默认 0.85）与并行度
2. 扫描用感知哈希 + 颜色直方图；**动图（GIF/WebP）按首帧/中间帧/末帧分别比对**，命中任意一帧即分组
3. 在分组里选择保留哪张，可「执行治理」按归档目录移动重复图（执行前会显示计划）

### 4. 分类整理

按识别结果生成移动计划（dry-run），确认后才真正移动文件；跨盘、重名冲突与失败原因都会逐条列出。

### 5. AI 生图（可选，驱动你自己的 ComfyUI）

1. 设置 → 生图配置：填 ComfyUI 安装目录（指向含 `ComfyUI` 与 `python` 两层的文件夹，如 `E:\...\ComfyUI-aki-v3`），点右上角**保存**
2. 「AI 生图」→ 点「启动 ComfyUI」（首次十几秒）；停止按钮对**手动启动的实例**同样有效（仅结束占用该端口的 Python 进程）
3. 选择工作流模板 → 在工作流面板里改节点参数 → 点生成

**工作流面板**把模板画成节点图（按数据流自动分层、连线连到具体输入行）：

- 右键/中键拖动平移，滚轮上下、Shift+滚轮左右、Ctrl+滚轮缩放；**双击画布全屏**，右上角也有全屏按钮
- 直接编辑节点里的文本/数字/开关：提示词改 `CLIPTextEncode`，尺寸改 `EmptyLatentImage`，采样参数改 `KSampler`
- 改动作为**本机覆盖值**保存在 `data\comfy-template-overrides.json`，模板文件不被修改；节点卡片变蓝表示已覆盖，点 ↺ 还原
- 生成时优先级：**工作流节点值 → 快捷参数（若你改过）→ 模板默认值**

工作流模板来自 `resources\comfy-workflows\`：

```powershell
# 把 ComfyUI 的 UI 格式工作流（File → Save）转成应用模板
python scripts/comfy_convert_workflow.py "<工作流.json>" --name my-workflow --output resources/comfy-workflows
```

本机导入的个人模板以 `*.private.json` 存放，只在你本机存在，不会进入版本库。

### 6. LoRA 训练集导出

「AI 生图 → 训练集导出」把人工矫正过的标注变成 kohya 可直接训练的数据：

```text
人工矫正的样本 → 按人物框裁剪（person/head/bust/large/整图）
→ 感知哈希去重 → WD14 打标 → <输出目录>/<角色>/NNN.png + NNN.txt + dataset.toml
```

caption 采用社区写法：质量前缀 + 触发词 + Danbooru 标签，逗号后不留空格。首次打标需从 Hugging Face 下载 WD14 模型（约 470MB，之后走缓存）。导出**只清理本工具自己生成的文件**，目标目录里有别人的文件会直接报错停止。

### 7. LoRA 训练（可选，驱动你本机安装的 kohya）

1. 设置 → 生图配置 → LoRA 训练：填 kohya_ss 或 sd-scripts 目录、训练底模（`.safetensors`）、输出目录；Python 留空会自动用训练器自带 `venv`
2. 「AI 生图 → LoRA 训练」：确认训练集配置（`dataset.toml`）、填 LoRA 名称、选预设
3. 开始训练：日志/进度/loss 实时显示，可随时停止；产出的 `.safetensors` 会即时列出
4. 点「安装到 ComfyUI」，LoRA 复制进 `models\loras`，随后可在生图页选用

显存参考（RTX 3060 Laptop 6GB 实测）：

| 配置 | 显存 | 2000 步耗时 |
| --- | --- | --- |
| SD1.5 · 512px | 约 2.8 GB | 约 16 分钟 |
| SDXL · 768px | 约 5.9 GB（贴上限，会明显变慢） | 约 40 分钟以上 |

小显存跑 SDXL 必须勾选「缓存文本编码器输出」（释放两个 CLIP 约 1.4GB）；它会自动生成 `dataset.cache-te.toml` 供本次训练使用。

### 8. 设置

| Tab | 内容 |
| --- | --- |
| 基础配置 | 运行方式、推理设备、**CUDA 运行时目录（含一键下载）**、后台任务优先级、界面字号、环境状态检查 |
| 识别配置 | ONNX 推理线程数、跳过已标注、**角色识别模型（切换/下载/删除）**、**参考图识别（开关/后端/建库）** |
| 相似度配置 | 并行度、默认阈值、是否含子目录、归档目录名 |
| 生图配置 | ComfyUI 目录与端口、生成图输出目录、LoRA 训练器配置 |

性能相关都可调：后台优先级（低调=让出 CPU，默认）、识别线程数、相似度并行度、训练数据加载进程。**保存是手动点击右上角「保存」**，保存后写库；重建便携包不会清空 `data\`。

## 完整实战：从识别到 LoRA

以「让模型认识某个冷门角色，并训练成 LoRA」为例，全流程如下。

### 第 0 步：选模型

| 用途 | 模型 | 说明 |
| --- | --- | --- |
| 头部检测 | `anime-head-v2.0-s` | 随包内置，负责找人头 |
| 角色识别（默认） | `animetimm-resnet101-dbv4-full` | 12,476 标签，热门角色准；不支持冷门角色 |
| 角色识别（可选） | `camie-initial`（deepghs Camie tagger） | **70,527 标签**，含 `uzumaki_kushina` 等冷门角色；824MB，装好后在识别配置里切换 |
| 参考图相似度 | CCIP（`deepghs/ccip_onnx`） | 与识别模型解耦，实测判别力最好；约 370MB，首次建库时下载 |
| 打标（训练集） | WD14 `SwinV2_v3` | 首次导出时下载约 470MB |
| 训练底模 | 你自己的 `.safetensors` | **必须与要训练的 LoRA 家族一致**：SD1.5 配 SD1.5，SDXL 配 SDXL |

冷门角色只有三种出路：换覆盖面更广的识别模型（Camie）、用参考图识别、或者自己训个人模型。前两者是即时的，后两者可以叠加。

### 第 1 步：识别 + 人工矫正

1. 「角色识别」选图库目录 → 开始识别（可用「跳过已标注」增量跑）
2. 逐张确认：改框、选对角色、把不认识的标为「未识别」
3. 目标：该角色积累 **≥3 张**（训个人模型要 ≥2 个角色各 ≥3 张，参考库/训练集则建议 20~40 张）

### 第 2 步：让识别模型认识这个角色

**方式 A（推荐，快）**：设置 → 识别配置 → 参考图识别 → 「用矫正结果建立参考库」。之后扫描会自动给出「参考匹配 xx%」。

**方式 B（更准，需要更多样本）**：用矫正结果训练**个人模型**（应用内「个人模型」入口）。

- 算法：`normalized_centroid_v1` —— 每个角色一个 L2 归一化质心，并用**留一验证**校准余弦阈值与 margin
- 入门条件：≥2 类、每类 ≥3 张、留一准确率 ≥90% 才会自动启用（不达标只给警告）
- 产物是版本化快照，可评估、可回滚；扫描时与基座模型分数融合

> 经验：先把参考库跑起来，样本攒够再训个人模型，两者共存时以个人模型为准。

### 第 3 步：导出训练集

「AI 生图 → 训练集导出」选角色 → 导出。建议：裁剪方式 `person`、分辨率与底模匹配（SD1.5 用 512，SDXL 用 1024）、质量前缀按底模选（Pony 系列用 `score_9,score_8_up,score_7_up`）、触发词用角色名（如 `kushina_uzumaki`）。

产物位置（应用产出的文件夹名统一带 `-manage` 后缀，便于和你在 ComfyUI/kohya 里手工做的区分）：

| 配置了 ComfyUI | 没配置 ComfyUI（回退到包内） |
| --- | --- |
| 训练集 → `<ComfyUI 上一级>\lora-datasets-manage\<角色>\` | `output\datasets\<角色>\` |
| LoRA → `<ComfyUI 上一级>\lora-models-manage\` | `output\loras\` |
| 出图 → `<ComfyUI 根>\Images\generated-manage\` | `output\generated\` |

导出内容为 `<角色>\NNN.png + NNN.txt` 与 `dataset.toml`；页面会显示 caption 抽样，打标质量不理想就调阈值或改 `remove_tags` 再导一次。

### 第 4 步：训练 LoRA

1. 设置 → 生图配置 → LoRA 训练：训练器目录（如 `E:\...\sd-scripts`）、底模、输出目录
2. 「AI 生图 → LoRA 训练」：训练集配置指向刚才的 `dataset.toml`、LoRA 名称、预设
3. 6GB 显存建议：**SD1.5 · 512px**；要跑 SDXL 就选 768 并勾选「缓存文本编码器输出」
4. 训练完成后点「安装到 ComfyUI」

### 第 5 步：用新 LoRA 出图

「AI 生图」选 checkpoint（与 LoRA 同家族）→ 选刚安装的 LoRA 与强度（0.8~0.9 起步）→ 提示词以触发词开头 → 生成。种子留 `-1` 表示随机，生成后会回显实际种子，方便复现。

**验证是否训成**：固定种子，只用触发词出图看角色是否稳定；再换姿势/服装描述，确认学到的是角色而不是某张训练图的构图。也可以用「角色识别」反过来Checkpoint生成图，看识别结果是否命中。

## 疑难排查

| 现象 | 处理 |
| --- | --- |
| 设置保存后不生效 | 保存是手动点击右上角「保存」；保存后看一眼状态提示。旧版本存在半写库问题，已修复 |
| GPU 版仍跑 CPU | 设置 → 基础配置看「CUDA 运行时」缺哪些 DLL，直接点**一键下载 CUDA 运行时**（约 1.4GB，落到 `app\cuda`），不需要装系统 CUDA |
| 启动 ComfyUI 报未配置目录 | 设置 → 生图配置重新选目录并**保存**；确认目录里同时有 `ComfyUI\main.py` 与 `python\python.exe` |
| 停止按钮点了没反应/变灰 | 现在只要 ComfyUI 在运行就能停；手动启动的实例会按端口找到 Python 进程后结束，其他程序占用端口时会拒绝并提示 |
| 工作流报节点缺失 | 该模板用了未安装的自定义节点（ControlNet、IPAdapter 等），在 ComfyUI 里装好对应插件后重试 |
| 首次导出训练集很慢 | 正在下载约 470MB 的 WD14 打标模型，之后走缓存 |
| 识别全是「未识别」 | 多数是模型标签表里没有这个角色，见上文参考图识别 / 个人模型 |

## 开发

```powershell
pnpm lint && pnpm test && pnpm build          # 前端
cargo test --manifest-path apps/desktop/src-tauri/Cargo.toml --locked
uv run --project apps/ai-worker --extra model --extra detector --extra dev python -m pytest apps/ai-worker/tests -q

build.bat              # CPU 便携包 -> dist_windows
build_gpu.bat          # CUDA 便携包 -> dist_windows_gpu（全量，含 CUDA 版 ONNX Runtime）
build_gpu_fast.bat     # 复用 GPU 环境，只重建程序与 Worker（改界面/逻辑时用这个）
```

打包脚本会保留已有的 `data\` 与 `output\`，并对内置推理引擎与兼容环境分别执行健康与能力探针。

架构边界：桌面端（Tauri/Rust）负责 IPC、任务生命周期与文件安全，Python Worker 负责推理，模型权重不进入 Git；ComfyUI（GPL-3.0）与 kohya 一律外置调用，NVIDIA 运行库按需下载而非随包分发。

## 文档索引

- [架构](docs/architecture.md) · [模型指南](docs/model-guide.md) · [角色 LoRA 训练](docs/lora-training.md)
- [文件安全](docs/file-safety.md) · [隐私](docs/privacy.md) · [故障排查](docs/troubleshooting.md)
- [基准测试](docs/benchmark.md) · [发布与自动更新](docs/release.md) · [风险清单](docs/risk-register.md) · [架构决策记录](docs/adr/README.md)

## 许可证

本项目原创代码采用 MIT，见 [LICENSE](LICENSE)。第三方依赖、识别模型、打标权重与训练器不自动继承本项目许可证，需分别遵守其上游许可。

开发者：[@RoamerFly](https://github.com/RoamerFly) · [项目仓库](https://github.com/RoamerFly/anime-pic-manage) · [提交问题](https://github.com/RoamerFly/anime-pic-manage/issues)
