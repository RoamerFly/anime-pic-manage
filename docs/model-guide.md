# 模型与模型适配指南

## 原则

`models/` 在 Git 中只保存说明、目录占位和 manifest 示例。权重由用户在本机下载或离线导入，安装器校验文件完整性后才允许激活。不要把“文件存在”当成安装成功，也不要把生成模型、LoRA 或 Embedding 当成角色识别器。

## V0.1 默认验证路线

```text
Anime Head Detector
  → Head / Bust / Large
  → AnimeTimm dbv4-full 角色适配器
  → category == character 的标签
  → 可选作品角色集合与实际标签交集
  → Top-5、Top1/Top2 margin、多尺度一致性
```

推荐先验证 `animetimm/resnet101.dbv4-full`，再根据 100～300 张自有测试图的准确率、待确认率、速度、内存和显存决定是否启用 ConvFormer B36。`dbv4-full` 是标签体系，不是可以硬编码的单一模型名称。作品角色集合通过独立 JSON 元数据注入，模型适配器不绑定特定作品。

## 官方来源（下载前重新核对）

| 用途 | 来源 | 说明 |
| --- | --- | --- |
| 头部检测 | [deepghs/anime_head_detection](https://huggingface.co/deepghs/anime_head_detection) | 需要同步模型元数据；优先使用仓库声明的 ONNX 文件 |
| 角色识别 | [animetimm/resnet101.dbv4-full](https://huggingface.co/animetimm/resnet101.dbv4-full) | 默认验证；页面访问条件和许可证以当前页面为准 |
| 角色识别（可选） | [animetimm/convformer_b36.dbv4-full](https://huggingface.co/animetimm/convformer_b36.dbv4-full) | 高精度候选，资源开销更高 |
| 相似度特征（后续） | [facebook/dinov2-small](https://huggingface.co/facebook/dinov2-small) | 仅用于候选检索，不直接判断重复 |
| 感知相似度（后续） | [LPIPS](https://github.com/richzhang/PerceptualSimilarity) | 只精查少量候选，不全量两两计算 |

AnimeTimm 受控仓库需要用户登录 Hugging Face 并接受模型页访问条件。模型页显示的许可证可能不同；下载前和发布前都要检查具体版本、权重、训练数据衍生限制及分发条件。项目 MIT 不覆盖第三方模型。

## 推荐目录

```text
models/
├─ detector/anime-head-v2.0-s/
├─ recognizer/animetimm-resnet101-dbv4-full/
├─ recognizer/animetimm-convformer-b36-dbv4-full/
└─ similarity/dinov2-small/
```

## 推理设备与 GPU 加速

角色识别流水线通过 ONNX Runtime 执行，可在“设置 → 推理运行环境 → 推理设备”中选择执行方式；相似度扫描使用感知哈希、颜色直方图等本地算法，不经过神经网络，因此不提供 GPU 路径。

| 选项 | 行为 |
| --- | --- |
| 自动（优先显卡） | 运行时提供 `CUDAExecutionProvider` 且模型元数据声明 `cuda` 时使用显卡；CUDA 会话创建失败时自动回退 CPU |
| 仅 CPU | 始终使用 `CPUExecutionProvider` |
| NVIDIA CUDA | 强制使用显卡；缺少 CUDA 执行提供器或模型未声明 `cuda` 时返回明确错误，不静默回退 |

模型元数据 `supported_devices` 需要包含 `cuda` 才会在“自动”模式下启用显卡，`resources/model-manifests/*.example.json` 与随仓库提供的 `models/*/metadata.json` 已包含该声明。

发行版默认携带 CPU 版 ONNX Runtime。GPU 版用 `scripts/package_windows.ps1 -ProjectRoot <仓库根目录> -Cuda` 打包，目标机器仍需安装匹配的 NVIDIA 驱动与 CUDA/cuDNN 运行时；未安装时请保持“仅 CPU”，或在“自动”模式下接受回退到 CPU。设置页的“运行能力”会显示当前检测到的执行提供器。

仓库根目录提供两个 GPU 构建入口：`build_gpu.bat`（全量，输出 `dist_windows_gpu`）与 `build_gpu_fast.bat`（复用上一次 GPU 构建的运行时与 Python 环境，只重建程序与 Worker 逻辑）。GPU 构建使用独立环境 `apps/ai-worker/.venv-gpu`，不会污染开发用的 `.venv`。构建完成后，设置页“环境配置”的“CUDA 实测”会用已安装模型真实加载一次会话，直接给出识别实际使用的执行提供器，而不是只看运行时声明。

设置页按用途分为三个页签：

| 页签 | 主要内容 |
| --- | --- |
| 基础配置 | 界面字号（0.5px 步进）、运行方式、推理设备与一次性环境状态检查 |
| 识别配置 | ONNX 推理线程数（CPU 并行度）、识别扫描默认是否跳过已标注图片 |
| 相似度配置 | 特征提取并行度、默认匹配灵敏度、默认包含子目录、默认归档文件夹名 |

界面字号以 14px 为基准整体缩放全部界面文字（滑块无级调整，± 按钮按 0.5px 步进），保存后下次启动继续生效。识别由单个 Worker 顺序执行，线程数只影响单次模型推理内部的并行度；相似度扫描是 CPU 感知哈希流水线，并行度直接决定特征提取速度。两项设置保存后立即对新任务生效。

权重目录中应包含上游要求的元数据，例如 AnimeTimm 的 `model.onnx`、`selected_tags.csv`、`preprocess.json`、`categories.json` 和 `thresholds.csv`。不要同时下载 ONNX、Safetensors 和 Pickle 三种权重；桌面首版优先 ONNX。相似度特征优先 Safetensors，避免不必要的 Pickle 权重。

## 模型锁定与校验

成功安装后，在应用数据目录写入本地 `model-lock.json`，至少记录：

```json
{
  "schema_version": 1,
  "repo_id": "animetimm/resnet101.dbv4-full",
  "resolved_revision": "<commit-or-tag>",
  "files": [
    {
      "file_name": "model.onnx",
      "file_size": 0,
      "sha256": "<sha256>",
      "downloaded_at": "<ISO-8601>",
      "license": "<upstream-license-or-unknown>"
    }
  ]
}
```

PowerShell 校验示例：

```powershell
Get-FileHash .\models\recognizer\animetimm-resnet101-dbv4-full\model.onnx -Algorithm SHA256
```

模型激活前必须检查标签文件存在、标签行数与输出维度一致、存在角色类别、预处理可解析且输入尺寸一致。缺失或错位时停用模型并报出原因。

## 打标模型（LoRA 训练集导出）

「AI 生图 → 训练集导出」用 WD14 打标器把裁剪后的角色图转成 Danbooru 标签 caption。当前固定使用 `SwinV2_v3`（`dghs-imgutils` 的默认档位，对应上游 `SmilingWolf/wd-swinv2-tagger-v3` 权重），通过 `huggingface_hub` 按需下载，模型与标签文件进入本机 Hugging Face 缓存（受 `HF_HOME` 控制），不写入仓库、不随安装包分发。

- 权重约 470MB，首次导出才会下载；离线或无法访问 huggingface.co 时导出会失败，其余识别功能不受影响。
- 通用标签阈值默认 `0.35`、角色标签阈值默认 `0.85`；`rating_*` 标签不写入 caption。
- 打标结果只是初稿，导出后建议抽查 caption 再交给 kohya 训练；训练器与权重许可证按各自上游执行。

## 作品角色集合

`resources/character-sets/<collection>.json` 只提供模型 Tag 到显示名、别名和相似组的元数据。运行时计算：

```text
当前作品集合 ∩ 当前模型的 character 标签 = 本次可识别集合
```

不存在于模型标签的角色必须显示为“当前模型不支持”，不能静默伪造标签或将其映射为相似角色。角色中文名缺失时回退到英文 Tag。

## 识别策略

裁剪和决策阈值配置化。默认初始策略可使用 Head/Bust/Large 权重 `0.4/0.4/0.2`、Top1 `0.8`、Top1-Top2 margin `0.15`、至少两个裁剪一致；这些是待基准校准的策略阈值，不是准确率保证。侧脸、背面、强遮挡、检测框质量差或相似角色 margin 不足时进入待确认；检测到人物但无可信标签时进入未识别。

## 下载命令（用户主动执行）

```powershell
py -m pip install --upgrade huggingface_hub
hf auth login
hf download deepghs/anime_head_detection `
  head_detect_v2.0_s/model.onnx `
  head_detect_v2.0_s/labels.json `
  head_detect_v2.0_s/model_type.json `
  head_detect_v2.0_s/model_artifacts.json `
  head_detect_v2.0_s/threshold.json `
  --local-dir models/detector/anime-head-v2.0-s
```

接受访问条件后再下载角色模型：

```powershell
hf download animetimm/resnet101.dbv4-full `
  model.onnx selected_tags.csv preprocess.json categories.json thresholds.csv `
  --local-dir models/recognizer/animetimm-resnet101-dbv4-full
```

这些命令不会由 CI 自动执行；Token 不得写入仓库、配置或日志。实际启用的作品角色集合由用户选择，并必须与当前模型标签交集后再参与决策。
