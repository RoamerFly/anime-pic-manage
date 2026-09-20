# 发布与自动更新安全说明

## 当前状态

仓库中的 workflow 是安全骨架，不代表已经生成可安装包或可用更新端点。真实发布前必须配置应用签名、公钥、更新端点、版本策略、制品存储和 Windows 构建环境，并完成一次离线/测试渠道验证。

## 签名密钥

在 GitHub 仓库设置：

```text
Settings → Secrets and variables → Actions → New repository secret
Name: TAURI_SIGNING_PRIVATE_KEY
```

密码（如果启用）应放入单独的 `TAURI_SIGNING_PRIVATE_KEY_PASSWORD` secret。workflow 只通过环境变量读取 secret，不写入文件、不打印、不提交。绝不在仓库创建私钥、公钥占位文件或把 secret 内容写入构建日志。

## 公钥与端点

Tauri 更新器需要与私钥匹配的公钥以及 HTTPS 更新端点。它们必须由维护者在真实密钥生成后填入应用配置或受控仓库变量。当前仓库不伪造公钥、端点或签名 manifest；空配置时发布工作流应失败并给出配置提示。

## 发布前清单

- 重新核对 Tauri、Rust、Node、Worker 版本与 lockfile；
- 在干净 Windows runner 构建并测试安装包；
- 验证安装包和更新包签名，公钥能校验对应制品；
- 检查更新说明、版本号、HTTPS 端点和回滚方式；
- 检查第三方模型是否未被意外打包，或已附带所有适用许可证；
- 确认 workflow 日志不包含私钥、Token、原图路径或用户数据；
- 先在测试渠道验证检查更新、版本说明、进度、取消、安装和重启。

模型下载和应用更新不是同一个信任边界。模型安装器还必须校验上游 revision、文件大小和 SHA-256。
