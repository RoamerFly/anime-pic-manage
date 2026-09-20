# 本地模型目录

本目录只跟踪模型来源、目录约定和 manifest 示例，不跟踪任何权重、图片或用户数据。`.gitignore` 会排除 ONNX、Safetensors、PyTorch 等常见权重格式。

## 目录约定

```text
models/
├─ detector/anime-head-v2.0-s/
├─ recognizer/animetimm-resnet101-dbv4-full/
├─ recognizer/animetimm-convformer-b36-dbv4-full/
└─ similarity/dinov2-small/
```

建议由模型管理器下载/导入文件，而不是手工把文件复制进 Git。安装成功后在本机应用数据目录维护 `model-lock.json`，记录上游仓库、resolved revision、文件大小、SHA-256、下载时间和许可证。

## 官方来源

- [Anime Head Detection](https://huggingface.co/deepghs/anime_head_detection)
- [AnimeTimm ResNet101 dbv4-full](https://huggingface.co/animetimm/resnet101.dbv4-full)
- [AnimeTimm ConvFormer B36 dbv4-full](https://huggingface.co/animetimm/convformer_b36.dbv4-full)
- [DINOv2 Small](https://huggingface.co/facebook/dinov2-small)
- [LPIPS](https://github.com/richzhang/PerceptualSimilarity)

下载前必须重新检查模型页的许可证、访问条件和文件 revision。详见 [模型指南](../docs/model-guide.md)。
