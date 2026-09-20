# 架构说明

## 范围与当前阶段

本文描述阶段 0/V0.1 的目标边界。它是实现约束，不代表相似图、真实分类执行或安装包已经完成。任何未在当前代码中存在的能力都必须在交付记录中标为未实现或未验证。

## 进程边界

```text
React + TypeScript UI
          │  Tauri commands/events
          ▼
Rust/Tauri Core
  ├─ 窗口与任务生命周期
  ├─ Worker 启动、健康检查、重启
  ├─ IPC 边界与路径/权限校验
  └─ SQLite/文件操作的受控入口
          │  本地 JSON Lines（或等价受控协议）
          ▼
Python AI Worker
  ├─ 图片读取与 EXIF 方向修正
  ├─ Anime Head Detector
  ├─ Head/Bust/Large 裁剪
  ├─ AnimeTimm 适配器与结果融合
  └─ V0.1 JSON/CSV/可视化导出
```

UI 不直接载入重量级模型，Worker 崩溃也不应使 UI 进程崩溃。IPC 消息至少携带协议版本、请求 ID、任务 ID、消息类型和结构化错误；路径作为结构化字段传递，禁止拼接命令字符串。

## V0.1 数据流

```text
文件夹选择
  → 递归枚举受支持图片
  → 解码并校正 EXIF
  → 头部检测（零、一个或多个框）
  → 每个 detection_id 生成三种裁剪
  → 识别器输出 Top-K
  → character 标签过滤
  → 作品集合求交集
  → 多裁剪融合、margin/一致性计算
  → high-confidence / needs-review / unrecognized
  → JSON、CSV、可视化导出
```

V0.1 的最后一步只写结果文件、日志和调试产物，不修改源图片。无人图必须与“检测到人物但没有可信角色”分开统计。

## 领域边界

后续阶段建议保持以下单一职责：

- `images` 保存源文件指纹、路径和业务状态。
- `detections` 保存每个检测实例的框、质量和姿态元数据。
- `recognitions` 保存模型、裁剪、Tag、显示名、置信度和候选序号。
- `classification_plans` 保存 dry-run 目标，不等于已执行。
- `file_operations` 保存每次真实文件操作及撤销信息。
- 相似度管线独立使用 SHA-256、感知哈希、向量检索和 LPIPS，不复用角色结论。

稳定 ID 使用模型原始英文 Tag；中文名只用于展示和目录。作品配置与模型标签求交集后才形成当前能力集合。

## 任务与错误

任务状态应支持 `queued → running → paused → running → completed`，并支持 `cancelled`、`failed → retrying`。任务需记录总数、成功、失败、跳过、阶段、时间戳和恢复游标。错误返回可分类为输入文件、模型兼容性、Worker、IPC、权限、输出冲突和未知错误，UI 负责映射为用户可理解的提示。

## 阶段演进

| 阶段 | 交付重点 | 本阶段的安全边界 |
| --- | --- | --- |
| 0 | 可启动桌面端、Worker health check、IPC schema、SQLite 初始迁移 | 不读用户图片也可完成健康检查；无文件写操作 |
| V0.1 | 检测、三尺度裁剪、Top-5、可选作品角色集合过滤、导出、分组基准 | 只读图片并写结果；不移动/复制/改名 |
| V0.2 | 递归扫描、缓存、断点续扫 | 缓存键绑定模型/预处理/策略版本 |
| V0.3 | 待确认、未识别、无人物、分类计划 | 只生成 dry-run 计划 |
| V0.4 | 经确认的移动/复制、冲突处理、事务与撤销 | 每项操作执行前重新校验，绝不覆盖 |
| V0.5 | 分层相似度检索与人工去重 | 删除只能由用户明确触发，优先回收站 |

## 可替换模型

模型加载通过适配器协议完成，而不是在业务流程中写死模型文件名。角色识别器至少具备 `load`、`unload`、`predict`、`get_character_labels`、`get_model_info` 和 `health_check` 语义。模型元数据需声明输入尺寸、预处理、输出激活、标签文件、适配器、许可证和 SHA-256。

详见 [模型指南](model-guide.md)。
