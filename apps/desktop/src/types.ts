import type {
  LibraryScanPayload,
  LibraryScanProgress,
  LibrarySelection,
  RuntimeStatus,
} from "@anime-pic-manage/shared-types";
import type { Candidate, PersonStatus, ResultFilter } from "./components/UnifiedInspector";
import type { RecentScanSummary } from "./lib/scan-session";
import type { ScanState } from "./lib/scan-state";

export interface DemoPerson {
  id: string;
  title: string;
  bbox: { left: number; top: number; width: number; height: number };
  confidence: number;
  status: PersonStatus;
  candidates: Candidate[];
  source: { person_index: number };
  margin?: number;
  consistency?: number;
  reason?: string | null;
}

export interface PreviewFile {
  name: string;
  path: string;
  source?: string;
}

export interface ScanController {
  state: ScanState;
  progress: LibraryScanProgress | null;
  error: string | null;
  latestScan: LibraryScanPayload | null;
  selectedModelVersion: number | null;
  setSelectedModelVersion: (version: number | null) => void;
  skipAnnotated: boolean;
  setSkipAnnotated: (skip: boolean) => void;
  start: (
    confirm?: () => boolean,
    overrideModelVersion?: number | null,
    targetDirectory?: string,
  ) => Promise<void>;
  pause: () => Promise<void>;
  resume: () => Promise<void>;
  cancel: () => Promise<void>;
  deleteScan: (directory?: string) => void;
  updateScanResults: (moved: Array<{ source: string; destination: string }>) => void;
}

export interface WorkspaceDataProps {
  status: RuntimeStatus | null;
  loading: boolean;
  library: LibrarySelection | null;
  onChooseLibrary: () => void;
}

export interface WorkspaceProps extends WorkspaceDataProps {
  recentScan: RecentScanSummary | null;
  onNavigate: (to: string) => void;
  controller: ScanController;
}

export type RecognitionTab = "recognition" | "organize" | "personal";

export interface RecognitionPageProps extends WorkspaceDataProps {
  initialTab?: RecognitionTab;
  initialFilter?: ResultFilter;
  autoStart?: boolean;
  onScanCompleted: (payload: LibraryScanPayload) => void;
  controller: ScanController;
}
