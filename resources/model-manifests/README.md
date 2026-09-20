# 模型 manifest 示例

此目录只保存可审计的元数据示例，不包含模型权重。将权重和 sidecar 文件放入本地 `models/` 对应目录后，Worker 才会激活 ONNX 适配器。

每个实际模型目录需要包含 `metadata.json`、`model.onnx`、标签 CSV 和预处理 JSON。`sha256` 为空表示暂不校验哈希；发布前应填写实际权重的 SHA-256。
