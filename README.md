# Anime Pic Manage

Windows 本地动漫图片相似度识别与角色识别分类工具。

项目定位是“本地优先、人工确认、安全文件操作”。图片、缩略图、识别结果和向量索引默认只保存在本机；识别阶段只生成结果和分类计划，不会自动移动、覆盖或永久删除源文件。

## 当前状态

阶段 0 的桌面端、Worker、版本化 IPC、SQLite 初始边界和健康状态页面已经落地。V0.1 的图片枚举、Head/Bust/Large 裁剪、结果融合/导出、模型清单校验与基准目录清点等核心组件已经落地；真实模型驱动的端到端检测界面、完整批量任务和可复核可视化仍在完善，不能把当前骨架当成完整识别产品。

已确定的边界：

- 桌面 UI 不直接运行重量级推理；Tauri/Rust 负责受控 IPC、任务生命周期和文件安全边界，Python Worker 负责 AI 推理。
- V0.1 不移动、复制、重命名或删除图片；真实文件操作必须先产生 dry-run 分类计划并经用户确认。
- 模型权重不提交到 Git。模型来源、许可证、版本和 SHA-256 应记录在本机锁文件中。
- `resources/character-sets/` 中的作品角色集合是可替换元数据，不把业务逻辑写死为单一作品；缺失于当前模型标签的角色必须显示为不支持。

未承诺的能力：完整批量扫描缓存、真实分类执行/撤销、相似图人工去重、安装包、在线发布和 GPU 兼容性，均需在后续阶段完成并分别验证。

## 开发前置（Windows）

建议使用 PowerShell，并安装：

- Git
- Node.js 22 LTS 或更高版本、pnpm 10/11（可用 Corepack）
- Rust stable、Visual Studio Build Tools 的 Desktop development with C++ 工作负载，以及 Tauri v2 所需 WebView2
- Python 3.11
- `uv`
- 如需真实模型验证：可用磁盘空间、可选 NVIDIA 驱动/CUDA，以及已接受访问条件的 Hugging Face 账号

安装 pnpm：

```powershell
corepack enable
corepack prepare pnpm@latest --activate
```

Python Worker 建议使用 uv 管理虚拟环境；模型下载需要单独登录 Hugging Face。不要把 Hugging Face token、模型权重或 Tauri 签名私钥写入项目。

## 本地开发命令

在项目根目录执行（浏览器预览）：

```powershell
pnpm install
pnpm dev
```

启动 Tauri 桌面开发模式：

```powershell
pnpm --filter @anime-pic-manage/desktop tauri:dev
```

若只需要前端单元测试/构建，常用验证命令为：

```powershell
pnpm lint
pnpm test
pnpm build
```

Rust 侧测试使用 Cargo；当前没有独立的 `tauri test` 命令：

```powershell
cargo fmt --manifest-path apps/desktop/src-tauri/Cargo.toml -- --check
cargo check --manifest-path apps/desktop/src-tauri/Cargo.toml --locked
cargo test --manifest-path apps/desktop/src-tauri/Cargo.toml --locked
```

Worker：

```powershell
uv sync --directory apps/ai-worker
'{"schema_version":"1.0","request_id":"health-check","task_id":"health-check","message_type":"health","payload":{}}' | uv run --directory apps/ai-worker python -m ai_worker
```

Tauri 构建：

```powershell
pnpm --filter @anime-pic-manage/desktop tauri:build
```

Windows 便携发布目录可使用 `scripts/package_windows.ps1`（或仓库根目录的 `build.bat`）生成。发布包**自带运行环境，目标机器不需要安装 Python / CUDA / 任何运行库**：

```text
AnimePicManage\
├─ 启动.bat / anime-pic-manage.exe / 使用说明.txt   ← 根目录只留这三项
├─ app\runtime\   内置推理引擎（ai-worker.exe）
├─ app\env\       兼容运行环境（可删，程序仍可用内置引擎）
├─ app\cuda\      显卡加速运行库（应用内一键下载，约 1.4GB，不随包分发）
├─ models\ resources\
├─ data\          数据库与扫描结果（备份这一个即可）
├─ output\        generated / loras / datasets
└─ temp\
```

打包脚本会检查根目录只有主程序与三个入口文件，并对内置推理引擎和兼容环境分别执行健康与运行能力探针。

GPU（NVIDIA CUDA）发布目录使用仓库根目录的两个脚本：

```powershell
build_gpu.bat        # 全量构建：CUDA Worker、便携 Python 环境与桌面程序 -> dist_windows_gpu
build_gpu_fast.bat   # 快速复用：环境未变时只重建程序与 Worker 逻辑 -> dist_windows_gpu
```

`build_gpu.bat` 会为 Worker 准备独立的 `apps/ai-worker/.venv-gpu` 环境并装入 CUDA 版 ONNX Runtime，不影响日常开发用的 `.venv`；`build_gpu_fast.bat` 复用上一次 GPU 构建的运行时与 Python 环境，适合只改了界面或处理逻辑的迭代。目标机器仍需安装匹配的 NVIDIA 驱动与 CUDA 12 / cuDNN 9 运行时，未安装时会自动回退 CPU。两个脚本支持 `--skip-tests` 与 `--no-pause`。

这些脚本以实际 `package.json` 为准；不存在的脚本不应被当成已通过的检查。

## V0.1 验证流程

准备自己拥有的 100～300 张测试图，并按正面、侧面、背面、多人物分组（参考 [基准测试说明](docs/benchmark.md)）：

```text
选择文件夹
→ 检测人物头部
→ 为每个检测框生成 Head / Bust / Large
→ 输出每个人物 Top-5 角色和置信度
→ 应用当前模型的 character 标签与可选作品角色集合交集
→ 导出 JSON / CSV / 可视化结果
```

检测失败和识别失败必须分开统计。没有真实测试图片、模型权重或上游访问权限时，只能验证协议、健康检查和模拟输入，不能宣称识别准确率。

## 模型

默认验证路线使用 Anime Head Detector 与 AnimeTimm `dbv4-full` 角色模型；相似度模块后续再接入 DINOv2、FAISS 与 LPIPS。模型管理、受控下载、标签校验、许可证与锁文件规则见 [模型指南](docs/model-guide.md) 和 [models/README.md](models/README.md)。

模型页和第三方权重的许可证不因本项目采用 MIT 而改变。应用发布前必须重新核对每个具体权重的许可证、访问条件和分发限制。

## 可选安装：AI 生图与 LoRA 训练

生图和训练都是**可选**能力，需要你自己安装对应的程序。不装 ComfyUI / kohya 时，识别、矫正、相似度、分类整理全部照常工作；「AI 生图」页只会提示尚未配置。

本应用**不打包、不修改**这些程序，只驱动你本机已装好的实例（ComfyUI 是 GPL-3.0，本项目是 MIT，因此只调用、不分发）。要准备什么：

| 想做什么 | 自己需要安装 | 在应用里填哪里 |
| --- | --- | --- |
| AI 出图 | ComfyUI（例如秋叶整合包 `ComfyUI-aki-v3`） | 设置 → 生图配置 → ComfyUI 目录 |
| 训练角色 LoRA | kohya_ss 或 sd-scripts，外加一个底模 `.safetensors` | 设置 → 生图配置 → LoRA 训练 |

### 一、联动 ComfyUI 出图

1. **填目录**：设置 → 生图配置 → 选择 ComfyUI 安装目录，指向同时含 `ComfyUI` 与 `python` 两层的文件夹（例如 `E:\freetime\AI_Draw\ComfyUI-aki\ComfyUI-aki-v3`）。端口默认 8188。
2. **启动**：打开「AI 生图」，点「启动 ComfyUI」，首次启动要十几秒；页面顶部会显示运行状态与实际使用的显卡型号。也可以先用整合包自己启动，应用会直接连上。
3. **出图**：选择 checkpoint，填提示词（或用 Pony / Illustrious / SD1.5 预设一键填入），设好尺寸、步数、CFG、batch 后点「生成」，结果图显示在下方，点缩略图可放大，也可打开输出目录。

装了你自己的 LoRA 之后，在同一个页面选择 LoRA 与权重强度即可参与出图。出图用的是本地 HTTP API（`/prompt`、`/history`、`/view`、`/models`、`/object_info`、`/system_stats`、`/interrupt`），不修改你的 ComfyUI。

已有 ComfyUI 工作流（UI 格式 JSON）可以转成本应用用的模板，转换时会直接提交给 ComfyUI 校验：

```powershell
python scripts/comfy_convert_workflow.py "<工作流.json>" --name my-workflow --verify --output resources/comfy-workflows
```

### 二、训练角色 LoRA

思路是：**识别 → 人工矫正 → 导出训练集 → 用本机 kohya 训练 → 装回 ComfyUI 出图**。

1. **矫正**：在「角色识别」里跑一次识别，把认错/漏认的框和角色名人工改对。每个角色去重后 20～40 张最合适。
2. **导出训练集**：「AI 生图 → 训练集导出」选择角色并导出。应用会自动按人物框裁剪、感知哈希去重、用 WD14 打标生成 caption，产出 kohya 能直接读的 `<角色>/NNN.png + NNN.txt` 与 `dataset.toml`。
3. **训练**：设置 → 生图配置 → LoRA 训练，填 kohya_ss / sd-scripts 目录、训练底模和输出目录，点「检测训练器」确认状态；回到「AI 生图 → LoRA 训练」，确认训练集配置、填 LoRA 名称、选预设（SD1.5 或 SDXL · 6GB 显存），点「开始训练」。日志、进度和 loss 会实时显示，可以随时停止，中途产出的 `.safetensors` 会即时列出来。
4. **装回 ComfyUI**：在训练产物上点「安装到 ComfyUI」，LoRA 会被复制到 `models/loras`，随后在出图页选中它，配合触发词使用即可。

几个容易踩的点：

- 底模必须和 LoRA 家族一致：SD1.5 的 LoRA 配 SD1.5 底模，SDXL 配 SDXL。
- **识别模型不一定认识你要的角色**：内建模型只有约 1.2 万个标签，像一些冷门角色就不在其中，这类图的识别结果只会是「未识别」。先用人工矫正 + 应用内「个人模型」把她变成可识别类别，再导出训练集。
- 第一次导出训练集会从 Hugging Face 下载 WD14 打标模型（约 470MB），之后走本地缓存。
- 导出只清理本工具自己写出的训练集：目标角色目录里有不是本工具写入的文件时会直接报错停止，不会动你手工整理的素材。已有数据集（例如 `model\1\<角色>` 那种 `名字NN.jpg + .txt`）可以继续用，格式一致。
- 训练会占满显卡，期间不要同时跑识别/相似度扫描或出图。

完整说明（含参数含义、许可证边界与效果判定方法）见 [角色 LoRA 训练](docs/lora-training.md)。

## 文件安全与隐私

文件移动/复制/重命名策略、跨卷操作、冲突处理、事务记录和撤销要求见 [文件安全](docs/file-safety.md)。隐私边界和日志原则见 [隐私说明](docs/privacy.md)。故障排查见 [troubleshooting](docs/troubleshooting.md)。

## 自动更新与发布安全

`.github/workflows/` 提供基础 CI、发布和更新配置检查骨架。发布签名私钥只能通过 GitHub Actions secret `TAURI_SIGNING_PRIVATE_KEY` 注入，不能创建或提交私钥文件；密码如启用则使用单独 secret。更新公钥、端点和签名配置必须由维护者在仓库变量/应用配置中填入真实值后才可发布。当前工作流不伪造可用的公钥、端点或已发布安装包，详见 [发布说明](docs/release.md)。

## 文档索引

- [架构](docs/architecture.md)
- [模型指南](docs/model-guide.md)
- [角色 LoRA 训练](docs/lora-training.md)
- [文件安全](docs/file-safety.md)
- [基准测试](docs/benchmark.md)
- [隐私](docs/privacy.md)
- [故障排查](docs/troubleshooting.md)
- [发布与自动更新](docs/release.md)
- [风险清单](docs/risk-register.md)
- [架构决策记录](docs/adr/README.md)

## 许可证

本项目原创代码采用 MIT，见 [LICENSE](LICENSE)。第三方依赖、检测器、识别模型和特征模型不自动继承本项目许可证，必须分别遵守其上游许可证。

开发者：[@RoamerFly](https://github.com/RoamerFly) · [项目仓库](https://github.com/RoamerFly/anime-pic-manage) · [提交问题](https://github.com/RoamerFly/anime-pic-manage/issues)
