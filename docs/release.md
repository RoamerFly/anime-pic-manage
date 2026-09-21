# 发布、安装与自动更新

## 一句话概览

推送 `v*.*.*` 标签 → GitHub Actions 在 Windows runner 上构建 **标准 NSIS 安装器 + 便携包 + 更新清单** → 发布到 Release。卸载选项、安装目录记忆、缓存落点都由安装器脚本与应用代码保证，不依赖用户手动清理。

## 两条安装路径

| 路径 | 产物 | 模型 | 适合谁 |
| --- | --- | --- | --- |
| CPU 安装版 | `AnimePicManage-<版本>-windows-x64-setup.exe`（NSIS） | 装完在应用内按需下载 | 体积小、没有 NVIDIA 显卡或接受 CPU 推理 |
| GPU 安装版 | `*-windows-x64-gpu-setup.exe`（NSIS） | GPU Worker；CUDA/cuDNN 与模型按需准备 | NVIDIA 显卡用户，小体积安装后在设置页一键准备 |
| 便携包 | `*-portable.zip` + 独立 runtime/模型/CUDA 分包 | 程序小更新与大型环境分离 | 需要整包带走、放到 U 盘/移动硬盘 |

两条路径的存储结构一致：轻量程序在根目录（另有 `BUILD_FLAVOR.txt` 构建标记），独立 AI 环境在 `app\runtime` / `app\env`，CUDA 在 `app\cuda`，模型与用户数据分别位于 `models\`、`data\`、`output\`。应用覆盖升级只替换程序本体，不删除已安装环境和用户数据。

### 为什么安装版不带环境和模型

AI/Python 环境与模型合计数 GB，塞进安装包会触及 NSIS/GitHub 单文件上限，也会让每次应用升级都重下依赖。安装版因此只带程序本体；AI 环境由“基础配置”按 CPU/GPU 类型独立安装并用 SHA-256 校验，模型由“模型配置”按需下载。发现本机已有缓存时只复制、不删除源目录。

便携发行拆成「轻量程序包 + CPU/GPU runtime 包 + 可选模型包 + GPU CUDA 包」。在线使用可由设置页自动下载、断点续传、校验并原子切换；离线使用将对应分包解压到同一目录。CPU 与 GPU 共用模型包。本地 `dist_windows_gpu` 仍是用于构建验收的完整目录。

## 安装器行为

`apps/desktop/src-tauri/installer/hooks.nsh` 通过 Tauri 的 `bundle.windows.nsis.installerHooks` 注入，负责三件事：

1. **安装后补齐目录**：创建 `data\`、`models\`、`output\{generated,loras,datasets}`、`temp\`，让用户装完就能看到落点。
2. **卸载清理选项**：在「确认卸载」之前插入自定义页，四个复选框——模型缓存 `data\hf-cache`（默认勾选）、识别模型 `models\`（默认勾选）、训练与出图产物 `output\`（默认不勾选）、数据库与人工矫正记录 `data\`（默认不勾选）。
3. **清理逻辑**：`NSIS_HOOK_POSTUNINSTALL` 里按选择删除，并始终清理 `app\`（含一键下载的 CUDA 运行库）与 `temp\`；取消勾选的数据原样保留，`$INSTDIR` 不会被强行删空，四项全选时安装目录会被完整删除。

静默卸载 `uninstall.exe /S` **默认只删程序**，要清理时显式传 ` /DELCACHE /DELMODELS /DELOUTPUT /DELDATA`。

「沿用上次安装目录」由 Tauri 模板自带：安装时把 `$INSTDIR` 写进 `HKCU\Software\<厂商>\<产品名>`，重装时 `RestorePreviousInstallLocation` 读回，因此 `tauri.conf.json` 必须保留 `bundle.publisher`。

> 注意：`hooks.nsh` 含中文，**必须保存为 UTF-8 with BOM**。makensis 对没有 BOM 的脚本按系统 ACP（中文 Windows 上是 GBK）解码，会把中文变成乱码。

## 本地构建

```powershell
build.bat                  # CPU 便携包 -> dist_windows
build_gpu.bat              # CUDA 便携包 -> dist_windows_gpu（全量）
build_gpu_fast.bat         # 复用 GPU 环境，只重建程序与 Worker
build_installer.bat        # 从 dist_windows 生成安装器 -> dist_windows\installer
build_installer.bat --gpu  # 从 dist_windows_gpu 生成安装器
```

`scripts/package_installer_windows.ps1` 只把 `resources` 与 CPU/GPU 构建标记暂存到 `src-tauri\installer-payload`，不再内嵌 AI 环境或 CUDA。随后用 `tauri build --config <临时覆盖>` 声明资源映射；环境作为带版本、SHA-256 的独立 Release 资产由应用管理。

带 `TAURI_SIGNING_PRIVATE_KEY` 时同时产出更新签名（`*-setup.exe.sig`）；没有私钥时只出普通安装器，不会失败。

## CI 发布流程

工作流：`.github/workflows/release.yml`。

- 触发：推送 `v*.*.*` 标签，或手动 `workflow_dispatch`（可指定版本、是否含 GPU、是否真的建 Release）。
- 前置：缺少 `TAURI_SIGNING_PRIVATE_KEY` 直接失败并提示去配置 Secrets。
- 步骤：检查工作区 → 注入统一版本 → 构建完整验收目录 → 生成轻量安装器/程序包 → 生成 CPU/GPU runtime、模型、CUDA 独立分包及 SHA-256 → 汇总 `latest.json` → 创建或追加 Release。
- `latest.json` 由工作流按签名文件生成，字段与应用内更新端点（`releases/latest/download/latest.json`）一致。
- 关闭 `publish` 时只跑构建并把资产上传为 Artifact，可用于发布前干跑。

## 发布前清单

- [ ] `pnpm lint` / `pnpm test` / `cargo test --locked` / `pytest` 全绿
- [ ] 在本机跑一次 `build_installer.bat`，静默安装到临时目录，确认目录结构与 worker 能启动
- [ ] 验证卸载三个选项的行为：勾选项被删除、未勾选项保留、注册表卸载项被清除
- [ ] 确认 Release 资产齐全（安装器 / 便携包 / 模型包 / `latest.json`）且单个文件小于 2GB
- [ ] 确认日志与 Release 正文不含私钥、Token、用户路径或原图信息
- [ ] 用旧版本触发一次「检查更新」，验证版本说明、下载进度、取消与安装
