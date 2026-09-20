import { Sparkles } from "lucide-react";

interface InitialLoadingProps {
  message?: string;
}

export function InitialLoading({ message = "正在初始化桌面核心与 AI 推理环境…" }: InitialLoadingProps) {
  return (
    <div className="initial-loading-screen" role="status" aria-live="polite">
      <div className="initial-loading-card">
        <div className="initial-loading-brand">
          <div className="brand-mark">
            <Sparkles size={22} />
          </div>
          <div>
            <div className="brand-title">Anime Pic Manage</div>
            <div className="brand-subtitle">本地智能图片工作台</div>
          </div>
        </div>

        <div className="initial-spinner-wrap">
          <div className="initial-spinner" aria-hidden="true" />
          <div className="initial-spinner-glow" aria-hidden="true" />
        </div>

        <p className="initial-loading-text">{message}</p>
        <span className="initial-loading-subtext">本地优先 · 正在准备安全文件沙箱与模型环境</span>
      </div>
    </div>
  );
}
