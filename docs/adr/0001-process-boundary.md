# ADR-0001：Tauri/Rust 与 Python Worker 进程隔离

- 状态：已接受（阶段 0 约束）
- 日期：2026-09-02

## 背景

人物检测、裁剪和角色识别依赖 Python/ONNX/PyTorch 生态，不能让 React UI 线程承担重量级推理。桌面 UI 还必须在 Worker 异常时保持可用并显示可恢复错误。

## 决定

Tauri/Rust 负责窗口、任务生命周期、受控 IPC、路径和后续文件安全边界；Python Worker 作为独立进程负责 AI 推理和 V0.1 导出。IPC 使用版本化、带请求 ID/任务 ID/结构化错误的消息。

## 结果

Worker 可独立 health check、重启和替换模型适配器；UI 不直接依赖具体模型库。需要额外维护 sidecar 生命周期、协议兼容和错误恢复。
