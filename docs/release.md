# 发布、安装与自动更新

## 一句话概览

推送 `v*.*.*` 标签 → GitHub Actions 在 Windows runner 上构建 **标准 NSIS 安装器 + 便携包 + 更新清单** → 发布到 Release。卸载选项、安装目录记忆、缓存落点都由安装器脚本与应用代码保证，不依赖用户手动清理。

## 两条安装路径

| 路径 | 产物 | 模型 | 适合谁 |
| --- | --- | --- | --- |
| 安装版 | `AnimePicManage-<版本>-windows-x64-setup.exe`（NSIS） | 装完在应用内按需下载 | 想装完就用、走「设置 → 应用」卸载 |
| 便携包 | `*-portable.zip` + `*-models.zip` | 模型包解压到同一目录 | 需要整包带走、放到 U 盘/移动硬盘 |

两条路径的程序目录结构一致：`anime-pic-manage.exe` + `app\`（推理运行时与兼容环境）+ `resources\` + `data\` / `models\` / `output\` / `temp\`。所有运行数据都在软件目录内，删目录即彻底卸载。

### 为什么安装版不带模型

模型合计约 1.1GB，塞进安装包会超过 GitHub Release 单文件 2GB 的上限，也会让每次升级都重下模型。安装版因此只带程序与推理运行时，模型交给「设置 → 模型配置」按需下载；如果本机别处已有缓存（`HF_HOME`、`%USERPROFILE%\.cache\huggingface` 等），同一面板会出现「复制到软件目录」按钮，一键迁移且不删除源目录。

便携包同理拆成「程序包 + 可选模型包」，两个压缩包解压到同一目录即可离线使用；CPU 与 GPU 共用同一个模型包。

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

`scripts/package_installer_windows.ps1` 会把便携包里的 `app\runtime` 与 `resources` 暂存到 `src-tauri\installer-payload`，再用 `tauri build --config <临时覆盖>` 声明资源映射（资源不写进 `tauri.conf.json`，这样 `cargo test`、`tauri dev` 和便携包构建都不依赖暂存目录）。

带 `TAURI_SIGNING_PRIVATE_KEY` 时同时产出更新签名（`*-setup.exe.sig`）；没有私钥时只出普通安装器，不会失败。

## CI 发布流程

工作流：`.github/workflows/release.yml`。

- 触发：推送 `v*.*.*` 标签，或手动 `workflow_dispatch`（可指定版本、是否含 GPU、是否真的建 Release）。
- 前置：缺少 `TAURI_SIGNING_PRIVATE_KEY` 直接失败并提示去配置 Secrets。
- 步骤：检查工作区（`pnpm lint` / `pnpm test`）→ 按标签版本改写 `tauri.conf.json` / `Cargo.toml` / `package.json` 版本 → 构建 CPU 便携包 → 构建 CPU 安装器 → 可选构建 GPU 两件套 → 汇总资产（`setup.exe`、`portable.zip`、`models.zip`、GPU 两份、`latest.json`）→ 创建或追加 Release。
- `latest.json` 由工作流按签名文件生成，字段与应用内更新端点（`releases/latest/download/latest.json`）一致。
- 关闭 `publish` 时只跑构建并把资产上传为 Artifact，可用于发布前干跑。

## 发布前清单

- [ ] `pnpm lint` / `pnpm test` / `cargo test --locked` / `pytest` 全绿
- [ ] 在本机跑一次 `build_installer.bat`，静默安装到临时目录，确认目录结构与 worker 能启动
- [ ] 验证卸载三个选项的行为：勾选项被删除、未勾选项保留、注册表卸载项被清除
- [ ] 确认 Release 资产齐全（安装器 / 便携包 / 模型包 / `latest.json`）且单个文件小于 2GB
- [ ] 确认日志与 Release 正文不含私钥、Token、用户路径或原图信息
- [ ] 用旧版本触发一次「检查更新」，验证版本说明、下载进度、取消与安装
