# 故障排查

## 桌面端无法启动

确认 Windows WebView2、Rust stable、Visual Studio C++ 构建工具和 Node/pnpm 已安装，然后在根目录执行：

```powershell
pnpm install
pnpm --filter @anime-pic-manage/desktop tauri:dev
```

如果只需要浏览器预览，再执行 `pnpm dev`；它不是 Tauri 桌面启动命令。

若根脚本不存在，检查 `apps/desktop/package.json` 的实际脚本。把终端中的首个错误（不要包含 Token 或私人路径）与 commit、Node、Rust 版本一起记录。

## Worker 健康检查失败

首次启动会载入较大的 Python/ONNX 依赖，超过 30 秒时界面会保持“AI 推理环境正在启动”，并在后台等待同一次检查完成，不需要反复点击刷新。如果长时间没有进展，再检查杀毒软件隔离、磁盘占用和下方 Worker 路径。

确认 Python 3.11 与 uv 可用：

```powershell
py --version
uv --version
uv sync --directory apps/ai-worker
'{"schema_version":"1.0","request_id":"health-check","task_id":"health-check","message_type":"health","payload":{}}' | uv run --directory apps/ai-worker python -m ai_worker
```

若 Worker 依赖模型才能启动，先使用 health-only 模式验证 IPC，再检查模型 manifest、文件存在性和 SHA-256。Worker 崩溃时应收集结构化错误并通过 UI 安全重启，不要反复启动损坏模型。

## 模型无法激活

检查：

- 是否从官方仓库取得了完整的模型和元数据；
- AnimeTimm 是否已在 Hugging Face 接受访问条件；
- `selected_tags.csv` 与输出维度是否一致；
- 是否存在 `category == character`；
- `preprocess.json` 的输入尺寸、颜色和归一化是否与模型一致；
- 本地锁文件 SHA-256 是否匹配；
- 当前模型许可证/访问条件是否允许此用途。

不要通过删除标签、忽略校验或猜测标签顺序来绕过错误。

## 识别结果为空或不稳定

先区分“没有检测到人物”和“检测到了人物但无可信角色”。确认图片已正确解码并修正 EXIF，检查 detection 框和 Head/Bust/Large 三种裁剪，再检查作品集合是否与模型实际 character 标签有交集。低 margin、裁剪冲突、侧脸/背面/遮挡应合理进入待确认，不能仅提高 Top-1 分数强行分类。

## GPU 加速不可用

在“设置 → 推理运行环境”确认两件事：运行能力里是否列出 `CUDAExecutionProvider`，以及“推理设备”选择的是“自动”还是“NVIDIA CUDA”。

- 只列出 `CPUExecutionProvider`：当前运行时是 CPU 版 ONNX Runtime，需要用 `-Cuda` 重新打包，或在 Worker 虚拟环境自行安装 `onnxruntime-gpu`。
- 列出 CUDA 但仍回退 CPU：看“基础配置 → 环境状态检查”里的 **CUDA 运行时** 一行。它会把缺失的动态库逐个点名（`cudart64_12.dll`、`cublas64_12.dll`、`cublasLt64_12.dll`、`cudnn64_9.dll`）——`onnxruntime-gpu` 只要求“编译时带 CUDA”，DLL 或 NVIDIA 驱动不可用时仍可能回退 CPU。GPU 包应已在 `app\cuda` 内备齐运行库；先确认该目录没有被杀毒软件隔离，再更新 NVIDIA 驱动。
- CPU 包需要 GPU 加速时，可在“基础配置 → 推理运行环境”里点「一键下载 CUDA 运行时」（约 2.1GB）。应用会从 PyPI 取 NVIDIA 官方的 `nvidia-cuda-runtime-cu12` / `nvidia-cublas-cu12` / `nvidia-cufft-cu12` / `nvidia-cudnn-cu12`，把 DLL 解压到 `app\cuda\` 并自动设为运行时目录，不需要安装系统 CUDA，也不需要改 PATH。
- 也可以用“CUDA 实测”一行确认：它会用已安装模型真正加载一次会话，只有会话真的用了 `CUDAExecutionProvider` 才算通过。
- CUDA 或模型下载失败时，检查“设置 → 基础配置 → 网络代理”。默认值是 `127.0.0.1:7890`；请确认代理程序正在监听该端口，端口不一致就修改，想直连则清空并保存。
- 报错“模型未声明 CUDA 支持”：对应模型的 `metadata.json` 里 `supported_devices` 缺少 `cuda`，补齐后重新检查即可。
- 相似度扫描没有 GPU 选项是预期行为：该流水线是 CPU 感知哈希/直方图算法，优化方向是多核提取与聚类，而不是显卡加速。

## 相似度扫描与动图（GIF/WebP）

相似度扫描支持 PNG/JPEG/WebP/GIF。静态图片只比较一帧；动画 GIF/WebP 会取**第一帧、中间帧、最后一帧**参与比较，取最相似的一帧对作为该图片的相似度，并在结果卡片上标注「动图 N 帧 · 比对首/中/末」。因此一张静图只要与动画中的任意采样帧一致，就会进入同一分组。

注意两点：

- 只采样三帧是速度与召回的折中；如果动画中间某帧与目标图一致但不在这三帧里，仍可能漏判。
- 升级到多帧特征后，旧版缓存里的动图会自动重新提取一次（其他图片继续用缓存），因此第一次扫描会稍慢。

## 路径或分类计划错误

V0.1 不执行文件操作。后续执行前检查源文件指纹、目标目录范围、目标冲突、权限和磁盘空间。跨卷移动必须复制、校验后才删除源文件；发现源文件变化或目标被外部修改时取消该项。永远不要使用覆盖参数。

## CI/发布失败

基础 CI 失败时先查看 Node/pnpm、Rust、Python 版本和 lockfile 是否一致。发布工作流需要仓库 Secret `TAURI_SIGNING_PRIVATE_KEY`（以及启用密码时的独立密码 secret）；私钥不应保存成文件或打印到日志。更新公钥、端点和签名配置是维护者的发布前工作，本仓库不包含伪造值。

## 报告问题

提交 [GitHub Issue](https://github.com/RoamerFly/anime-pic-manage/issues) 时提供复现步骤、应用/Worker 版本、操作系统、模型 ID 和错误码。请脱敏路径，不要上传原图、裁剪图、数据库、Token、私钥或含隐私的日志。
