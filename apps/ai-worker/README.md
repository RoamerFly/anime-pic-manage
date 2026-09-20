# AI Worker

这是一个本地、无网络推理边界的 JSON Lines Worker。标准输出只包含响应 JSONL，结构化日志写到标准错误；Worker 不移动或删除图片。

## 开发运行

在仓库根目录执行：

```powershell
uv run --project apps/ai-worker --with 'Pillow>=10,<13' python apps/ai-worker/run_worker.py
```

模型能力是可选依赖。安装模型运行时后，Worker 才会激活 ONNX 适配器：

```powershell
uv sync --project apps/ai-worker --extra model
```

## 验证

```powershell
uv run --project apps/ai-worker --with pytest pytest apps/ai-worker/tests -q
uv run --project apps/ai-worker --with ruff ruff check apps/ai-worker/src apps/ai-worker/tests
uv run --project apps/ai-worker --with mypy mypy apps/ai-worker/src/ai_worker
```

支持的基础消息包括 `health`、`shutdown`、`model.list`、`model.health`、`images.enumerate`、`crops.create`、`recognition.embedding`、`recognition.image`、`personal_model.train`、`personal_model.evaluate`、`character_set.intersection`、`recognition.fuse` 和 `results.export`。每条消息都包含 `schema_version`、`request_id`、`task_id`、`message_type`、`payload`、`error` 字段。

`recognition.embedding` 只读图片并为一个或多个归一化标注提取 2048 维 L2 归一化向量：

```json
{
  "path": "D:/pictures/example.png",
  "annotations": [
    {"annotation_id": "box-1", "bbox": {"x1": 0.1, "y1": 0.1, "x2": 0.6, "y2": 0.8}}
  ]
}
```

`recognition.image` 可选接收 `personal_prototypes`。每个原型包含 `identity_id`、`display_name`、2048 维 `embedding` 和正整数 `sample_count`。原型会与基础模型候选融合，并在 `top1`/`candidates` 中返回 `source`、`base_score`、`personal_score` 和 `sample_count`。单样本或相似度未超过严格阈值的原型始终返回 `needs_review`；至少 3 个样本且超过阈值后才允许 `high_confidence`。

## 个人模型训练

`personal_model.train` 接收一个版本号和带标签的 2048 维 embedding 样本，生成可持久化的个人模型快照。Worker 只计算并返回 JSON，不保存原图、不联网，也不修改原图：

```json
{
  "version": "personal-v1",
  "samples": [
    {"identity_id": "character-a", "display_name": "角色 A", "embedding": [0.0, 0.0]}
  ],
  "config": {
    "min_samples_per_class": 3,
    "min_classes": 2,
    "min_validation_count": 1,
    "min_accuracy": 0.9
  }
}
```

示例中的 embedding 仅为结构示意；实际数组必须严格为 2048 个有限数字。每个 `identity_id` 可以有多个样本，但其 `display_name` 必须一致。训练返回：

```json
{
  "artifact": {
    "schema_version": "1.0",
    "algorithm": "normalized_centroid_v1",
    "version": "personal-v1",
    "prototypes": [],
    "strict_threshold": 0.86,
    "min_margin": 0.08
  },
  "metrics": {},
  "warnings": [],
  "eligible_for_activation": true
}
```

`normalized_centroid_v1` 是确定性的轻量分类器，不伪称神经网络微调：每类 prototype 是该类样本向量求和后 L2 归一化的 centroid。训练会对每个至少有两个样本的类别执行留一验证，记录准确率、类内相似度、类间相似度和 margin。未显式配置时，`strict_threshold` 取留一正例最低相似度与负例最高相似度的中点（下限 0.50，上限 0.995）；没有足够统计量时回退到 0.86。`min_margin` 取留一 margin 的低分位一半，并限制在 0.02 到 0.50；没有验证样本时回退到 0.08。

默认只有类别数不少于 2、每类至少 3 个样本、留一验证样本数达标且准确率不低于 0.90 时，`eligible_for_activation` 才为 `true`。样本不足仍会生成草稿 artifact，但不能自动激活；warnings 会说明具体原因。

`personal_model.evaluate` 需要 `personal_model`（或 `artifact`）和单独提供的带标签 `samples`，用于评估持久化快照的 holdout 数据。传给 `recognition.image` 时使用 `personal_model` 字段，Worker 会严格校验 schema、算法、版本、原型、阈值和 margin，并在响应顶层及人物结果中返回 `model_version`。旧的 `personal_prototypes` 字段仍然兼容；两者同时提供会返回 `INVALID_PAYLOAD`，避免模型来源不明确。

## Windows 便携构建

从仓库根目录运行 `build.bat --no-pause` 会生成 `dist_windows`：其中的
`ai-worker.exe` 由 PyInstaller 打包，不需要目标电脑安装 Python 或 uv；`env`
目录包含可搬运的 Python 运行时、依赖和 Worker 源码，可在桌面设置中选择“内置 ENV”。
`dist_windows\models` 是随包复制的模型目录，原始图库不会被构建脚本触碰。
