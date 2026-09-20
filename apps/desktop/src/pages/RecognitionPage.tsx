import { useEffect, useState } from "react";
import { BrainCircuit, FolderOpen, UsersRound } from "lucide-react";
import { CategoryOrganizePanel } from "../components/CategoryOrganizePanel";
import { ClassificationPage } from "./ClassificationPage";
import { PersonalModelPage } from "./PersonalModelPage";
import type { RecognitionPageProps, RecognitionTab } from "../types";

export function RecognitionPage({
  initialTab = "recognition",
  initialFilter = "all",
  autoStart = false,
  status,
  library,
  onChooseLibrary,
  onScanCompleted,
  controller,
}: RecognitionPageProps) {
  const [tab, setTab] = useState<RecognitionTab>(initialTab);
  useEffect(() => {
    setTab(initialTab);
  }, [initialTab]);

  return (
    <div className="page-stack recognition-page">
      <section
        className="workspace-tabs recognition-tabs"
        role="tablist"
        aria-label="角色识别功能"
      >
        <button
          className={`workspace-tab${tab === "recognition" ? " active" : ""}`}
          role="tab"
          aria-selected={tab === "recognition"}
          onClick={() => setTab("recognition")}
        >
          <UsersRound size={16} />
          识别与复核
        </button>
        <button
          className={`workspace-tab${tab === "organize" ? " active" : ""}`}
          role="tab"
          aria-selected={tab === "organize"}
          onClick={() => setTab("organize")}
        >
          <FolderOpen size={16} />
          分类整理
        </button>
        <button
          className={`workspace-tab${tab === "personal" ? " active" : ""}`}
          role="tab"
          aria-selected={tab === "personal"}
          onClick={() => setTab("personal")}
        >
          <BrainCircuit size={16} />
          个人模型
        </button>
      </section>
      {tab === "personal" ? (
        <PersonalModelPage
          onSwitchToScan={(version?: number) => {
            if (version !== undefined) {
              controller.setSelectedModelVersion(version);
            }
            setTab("recognition");
          }}
        />
      ) : tab === "organize" ? (
        <CategoryOrganizePanel
          scanResults={controller.latestScan?.results ?? []}
          libraryPath={library?.path}
          onFilesMoved={controller.updateScanResults}
        />
      ) : (
        <ClassificationPage
          key={`${initialFilter}-${autoStart}`}
          initialFilter={initialFilter}
          autoStart={autoStart}
          library={library}
          worker={status?.worker.status ?? "unavailable"}
          onChooseLibrary={onChooseLibrary}
          onScanCompleted={onScanCompleted}
          controller={controller}
          onOpenPersonalModel={() => setTab("personal")}
        />
      )}
    </div>
  );
}
