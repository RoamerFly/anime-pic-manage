import {
  useCallback,
  useEffect,
  useMemo,
  useRef,
  useState,
  useSyncExternalStore,
} from "react";
import { listen } from "@tauri-apps/api/event";
import { open } from "@tauri-apps/plugin-dialog";
import { NavLink, Route, Routes, useLocation, useNavigate } from "react-router-dom";
import {
  AlertCircle,
  ChevronRight,
  CircleHelp,
  Info,
  LayoutDashboard,
  LoaderCircle,
  ImagePlus,
  RefreshCw,
  Save,
  ScanSearch,
  Settings as SettingsIcon,
  UsersRound,
} from "lucide-react";
import type {
  AppSettings,
  LibraryScanPayload,
  LibraryScanProgress,
  LibraryScanResultEvent,
  LibrarySelection,
  RuntimeStatus,
} from "@anime-pic-manage/shared-types";
import { invokeCore, isTauriRuntime, settleWithin } from "./lib/tauri";
import {
  applyUiFontSize,
  cacheUiFontSize,
  UI_FONT_BASE_PX,
} from "./lib/ui-font";
import {
  getSettingsSaveSnapshot,
  runSettingsSave,
  subscribeSettingsSave,
} from "./lib/settings-bridge";
import {
  deleteStoredScan,
  persistScan,
  readRecentScanSummary,
  readStoredScanForDirectory,
  scanSummary,
  type RecentScanSummary,
} from "./lib/scan-session";
import { isScanActive, scanStateFromPhase, type ScanState } from "./lib/scan-state";
import { InitialLoading } from "./components/InitialLoading";
import {
  WorkspacePage,
  RecognitionPage,
  SimilarityPage,
  GenerationPage,
  SettingsPage,
  AboutUpdateModal,
  developerUrl,
  repositoryUrl,
  issuesUrl,
} from "./pages";
import type { ScanController } from "./types";

const navItems = [
  { to: "/", label: "图片工作台", icon: LayoutDashboard, end: true },
  { to: "/recognition", label: "角色识别", icon: UsersRound },
  { to: "/similarity", label: "相似计算", icon: ScanSearch },
  { to: "/generate", label: "AI 生图", icon: ImagePlus },
  { to: "/settings", label: "设置", icon: SettingsIcon },
];

const workspacePaths = new Set(["/"]);
const recognitionPaths = new Set([
  "/recognition",
  "/classification",
  "/personal-model",
  "/review",
]);
const similarityPaths = new Set(["/similarity"]);

function normalizedScanPath(path: string): string {
  return path.replace(/\\/g, "/").replace(/\/+$/, "").toLowerCase();
}

export function mergeIncrementalScanResult(
  previous: LibraryScanPayload | null,
  event: LibraryScanResultEvent,
): LibraryScanPayload {
  const sameDirectory =
    previous &&
    normalizedScanPath(previous.directory) === normalizedScanPath(event.directory);
  const results = sameDirectory ? [...previous.results] : [];
  const errors = sameDirectory ? [...previous.errors] : [];

  if (event.result) {
    const resultIndex = results.findIndex(
      (item) => normalizedScanPath(item.path) === normalizedScanPath(event.result!.path),
    );
    if (resultIndex >= 0) results[resultIndex] = event.result;
    else results.push(event.result);
  }
  if (event.error) {
    const errorIndex = errors.findIndex(
      (item) =>
        normalizedScanPath(item.path) === normalizedScanPath(event.error!.path) &&
        item.message === event.error!.message,
    );
    if (errorIndex >= 0) errors[errorIndex] = event.error;
    else errors.push(event.error);
  }

  return {
    directory: event.directory,
    total_discovered: event.total_discovered,
    processed: event.processed,
    cancelled: false,
    results,
    errors,
    model_version: event.model_version,
    model_name: event.model_name ?? undefined,
    skip_annotated: event.skip_annotated,
  };
}

function useRuntimeStatus() {
  const [status, setStatus] = useState<RuntimeStatus | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(true);
  const [warming, setWarming] = useState(false);
  const refreshSequence = useRef(0);
  const refresh = useCallback(async () => {
    const sequence = ++refreshSequence.current;
    setLoading(true);
    setWarming(false);
    setError(null);

    const applyResponse = (response: Awaited<ReturnType<typeof invokeCore<RuntimeStatus>>>) => {
      if (sequence !== refreshSequence.current) return;
      if (response.error || !response.payload) {
        setError(response.error?.message ?? "AI 推理环境没有返回可用状态。");
      } else {
        setStatus(response.payload);
        setError(null);
      }
      setWarming(false);
      setLoading(false);
    };

    // A signed GPU package can need more than 30 seconds on the very first
    // launch while antivirus scans the Worker and CUDA DLLs. Keep the original
    // request alive and apply its result automatically instead of reporting a
    // false failure and asking the user to refresh manually.
    const task = invokeCore<RuntimeStatus>("get_runtime_status", "system.health");
    const response = await settleWithin(task, 30_000);
    if (sequence !== refreshSequence.current) return;
    if (response) {
      applyResponse(response);
      return;
    }

    setLoading(false);
    setWarming(true);
    void task.then(applyResponse);
  }, []);
  useEffect(() => {
    void refresh();
  }, [refresh]);
  return { status, error, loading, warming, refresh };
}

function App() {
  const { status, error, loading, warming, refresh } = useRuntimeStatus();
  const [initialLoaded, setInitialLoaded] = useState(false);
  const [library, setLibrary] = useState<LibrarySelection | null>(null);
  const [recentScan, setRecentScan] = useState<RecentScanSummary | null>(() =>
    readRecentScanSummary(),
  );
  const [latestScan, setLatestScan] = useState<LibraryScanPayload | null>(null);
  const [scanProgress, setScanProgress] = useState<LibraryScanProgress | null>(
    null,
  );
  const [scanState, setScanState] = useState<ScanState>("idle");
  const [scanError, setScanError] = useState<string | null>(null);
  const [selectedModelVersion, setSelectedModelVersion] = useState<
    number | null
  >(null);
  const [skipAnnotated, setSkipAnnotated] = useState<boolean>(true);
  const [aboutOpen, setAboutOpen] = useState(false);
  const settingsSave = useSyncExternalStore(
    subscribeSettingsSave,
    getSettingsSaveSnapshot,
  );
  const scanRunRef = useRef(false);
  const activeScanDirectoryRef = useRef<string | null>(null);
  const latestScanRef = useRef<LibraryScanPayload | null>(null);
  const partialPersistTimerRef = useRef<ReturnType<typeof setTimeout> | null>(null);

  const navigate = useNavigate();
  const location = useLocation();

  // Apply the saved recognition defaults once at startup; the switch in the
  // recognition toolbar can still override them for the current session.
  useEffect(() => {
    let active = true;
    void invokeCore<AppSettings>("get_app_settings", "settings.get").then(
      (response) => {
        if (!active || !response.payload) return;
        setSkipAnnotated(response.payload.recognition_skip_annotated);
        const fontSize = response.payload.ui_font_size ?? UI_FONT_BASE_PX;
        applyUiFontSize(fontSize);
        cacheUiFontSize(fontSize);
      },
    ).catch(() => undefined);
    return () => {
      active = false;
    };
  }, []);

  useEffect(() => {
    if (!loading && (status || error || warming)) {
      setInitialLoaded(true);
    }
  }, [loading, status, error, warming]);

  const flushPartialScan = useCallback(() => {
    if (partialPersistTimerRef.current) {
      clearTimeout(partialPersistTimerRef.current);
      partialPersistTimerRef.current = null;
    }
  }, []);

  const schedulePartialScanPersist = useCallback((payload: LibraryScanPayload) => {
    latestScanRef.current = payload;
  }, []);

  useEffect(() => {
    const handlePageHide = () => flushPartialScan();
    const handleVisibilityChange = () => {
      if (document.visibilityState === "hidden") flushPartialScan();
    };
    window.addEventListener("pagehide", handlePageHide);
    document.addEventListener("visibilitychange", handleVisibilityChange);
    return () => {
      window.removeEventListener("pagehide", handlePageHide);
      document.removeEventListener("visibilitychange", handleVisibilityChange);
      flushPartialScan();
    };
  }, [flushPartialScan]);

  const chooseLibrary = useCallback(async () => {
    if (scanRunRef.current) {
      setScanError("扫描进行中，当前图片库已锁定；请先终止任务或等待完成。");
      return;
    }
    if (!isTauriRuntime()) return;
    const selected = await open({
      directory: true,
      multiple: false,
      title: "选择本地图库",
    });
    if (typeof selected !== "string") return;
    const response = await invokeCore<LibrarySelection>(
      "register_library_selection",
      "library.select",
      { path: selected },
    );
    if (!response.error && response.payload) setLibrary(response.payload);
  }, []);

  useEffect(() => {
    let active = true;
    let unlistenProgress: (() => void) | undefined;
    let unlistenResult: (() => void) | undefined;
    void listen<LibraryScanProgress>("scan-progress", (event) => {
      if (!active) return;
      setScanProgress(event.payload);
      setScanState(scanStateFromPhase(event.payload.phase));
      if (["complete", "cancelled"].includes(event.payload.phase)) {
        scanRunRef.current = false;
        const current = latestScanRef.current;
        if (current) {
          const completed = {
            ...current,
            processed: event.payload.current,
            cancelled: event.payload.phase === "cancelled",
          };
          latestScanRef.current = completed;
          setLatestScan(completed);
          flushPartialScan();
        }
      }
    })
      .then((stop) => {
        unlistenProgress = stop;
      })
      .catch(() => undefined);
    void listen<LibraryScanResultEvent>("library://scan-result", (event) => {
      if (!active) return;
      const expectedDirectory = activeScanDirectoryRef.current;
      if (
        expectedDirectory &&
        normalizedScanPath(expectedDirectory) !==
          normalizedScanPath(event.payload.directory)
      ) {
        return;
      }
      activeScanDirectoryRef.current = event.payload.directory;
      setLatestScan((current) => {
        const merged = mergeIncrementalScanResult(current, event.payload);
        schedulePartialScanPersist(merged);
        return merged;
      });
    })
      .then((stop) => {
        unlistenResult = stop;
      })
      .catch(() => undefined);
    return () => {
      active = false;
      unlistenProgress?.();
      unlistenResult?.();
    };
  }, [flushPartialScan, schedulePartialScanPersist]);

  useEffect(() => {
    let active = true;
    const restored = readStoredScanForDirectory(library?.path);
    setLatestScan(restored);
    latestScanRef.current = restored;
    setRecentScan(
      restored
        ? scanSummary(
            restored,
            restored.completed_at ?? new Date(0).toISOString(),
          )
        : null,
    );
    activeScanDirectoryRef.current = library?.path ?? null;
    if (!scanRunRef.current) {
      setScanProgress(null);
      setScanError(null);
      setScanState(
        restored ? (restored.cancelled ? "cancelled" : "completed") : "idle",
      );
    }
    if (library?.path && isTauriRuntime()) {
      void invokeCore<LibraryScanPayload | null>(
        "get_latest_library_scan_result",
        "recognition.get_results",
        { path: library.path },
      ).then((response) => {
        if (!active || response.error || !response.payload) return;
        latestScanRef.current = response.payload;
        setLatestScan(response.payload);
        const completedAt = persistScan(response.payload);
        setRecentScan(scanSummary(response.payload, completedAt));
        if (!scanRunRef.current) {
          setScanState(response.payload.cancelled ? "cancelled" : "completed");
        }
      });
    }
    return () => {
      active = false;
    };
  }, [library?.path]);

  const onScanCompleted = useCallback((payload: LibraryScanPayload) => {
    if (partialPersistTimerRef.current) {
      clearTimeout(partialPersistTimerRef.current);
      partialPersistTimerRef.current = null;
    }
    latestScanRef.current = payload;
    activeScanDirectoryRef.current = payload.directory;
    setLatestScan(payload);
    const completedAt = persistScan(payload);
    setRecentScan(scanSummary(payload, completedAt));
  }, []);

  const updateScanResults = useCallback(
    (moved: Array<{ source: string; destination: string }>) => {
      const map = new Map(moved.map((m) => [m.source, m.destination]));
      setLatestScan((prev) => {
        if (!prev) return null;
        const updatedResults = prev.results.map((r) => {
          if (map.has(r.path)) {
            return { ...r, path: map.get(r.path)! };
          }
          return r;
        });
        const updatedScan = { ...prev, results: updatedResults };
        latestScanRef.current = updatedScan;
        persistScan(updatedScan);
        return updatedScan;
      });
    },
    [],
  );

  const deleteScan = useCallback(
    (directory?: string) => {
      const targetDir = directory || library?.path;
      deleteStoredScan(targetDir);
      if (targetDir && isTauriRuntime()) {
        void invokeCore<boolean>(
          "delete_latest_library_scan_result",
          "recognition.get_results",
          { path: targetDir },
        );
      }
      if (
        !targetDir ||
        (latestScanRef.current &&
          normalizedScanPath(latestScanRef.current.directory) ===
            normalizedScanPath(targetDir))
      ) {
        if (partialPersistTimerRef.current) {
          clearTimeout(partialPersistTimerRef.current);
          partialPersistTimerRef.current = null;
        }
        latestScanRef.current = null;
        setLatestScan(null);
        setScanProgress(null);
        setScanError(null);
        setScanState("idle");
      }
      setRecentScan(readRecentScanSummary());
    },
    [library?.path],
  );

  const startScan = useCallback(
    async (
      confirm?: () => boolean,
      overrideModelVersion?: number | null,
      targetDirectory?: string,
    ) => {
      if (confirm && !confirm()) return;
      const worker = status?.worker.status ?? "unavailable";
      const scanDirectory = targetDirectory?.trim() || library?.path;
      if (!scanDirectory || worker !== "healthy" || scanRunRef.current) return;
      scanRunRef.current = true;
      activeScanDirectoryRef.current = scanDirectory;
      setScanState("running");
      setScanProgress(null);
      setScanError(null);

      const modelToUse =
        overrideModelVersion !== undefined
          ? overrideModelVersion
          : selectedModelVersion;
      const pendingScan: LibraryScanPayload = {
        directory: scanDirectory,
        total_discovered: 0,
        processed: 0,
        cancelled: false,
        results: [],
        errors: [],
        model_version: modelToUse,
        skip_annotated: skipAnnotated,
      };
      latestScanRef.current = pendingScan;
      setLatestScan(pendingScan);
      const response = await invokeCore<LibraryScanPayload>(
        "scan_library_preview",
        "library.scan.start",
        {
          path: scanDirectory,
          limit: null,
          model_version: modelToUse,
          skip_annotated: skipAnnotated,
        },
      );
      scanRunRef.current = false;
      if (response.error || !response.payload) {
        setScanState("failed");
        setScanError(
          response.error?.message ??
            "扫描没有返回结果，请检查 AI Worker 状态。",
        );
        return;
      }
      onScanCompleted(response.payload);
      setScanState(response.payload.cancelled ? "cancelled" : "completed");
      setScanProgress(
        (current) =>
          current ?? {
            phase: response.payload?.cancelled ? "cancelled" : "complete",
            current: response.payload?.processed ?? 0,
            total: response.payload?.total_discovered ?? 0,
            path: response.payload?.directory ?? scanDirectory,
            message: response.payload?.cancelled
              ? "已取消扫描。"
              : "扫描与识别完成。",
          },
      );
    },
    [
      library,
      onScanCompleted,
      selectedModelVersion,
      skipAnnotated,
      status?.worker.status,
    ],
  );

  const pauseScan = useCallback(async () => {
    if (!scanRunRef.current || scanState !== "running") return;
    setScanState("pausing");
    const response = await invokeCore<{ status?: string; message?: string }>(
      "pause_library_scan",
      "library.scan.pause",
    );
    if (response.error) {
      setScanError(response.error.message);
      setScanState("running");
      return;
    }
    if (response.payload?.status === "paused") setScanState("paused");
  }, [scanState]);

  const resumeScan = useCallback(async () => {
    if (!scanRunRef.current || scanState !== "paused") return;
    const response = await invokeCore<{ status?: string; message?: string }>(
      "resume_library_scan",
      "library.scan.resume",
    );
    if (response.error) {
      setScanError(response.error.message);
      setScanState("paused");
      return;
    }
    setScanState(
      response.payload?.status === "cancelling" ? "cancelling" : "running",
    );
  }, [scanState]);

  const cancelScan = useCallback(async () => {
    if (
      !scanRunRef.current ||
      !["running", "pausing", "paused"].includes(scanState)
    )
      return;
    setScanState("cancelling");
    const response = await invokeCore<unknown>(
      "cancel_library_scan",
      "library.scan.cancel",
    );
    if (response.error) {
      setScanError(response.error.message);
      setScanState(scanState === "paused" ? "paused" : "running");
      return;
    }
  }, [scanState]);

  const scanController = useMemo<ScanController>(
    () => ({
      state: scanState,
      progress: scanProgress,
      error: scanError,
      latestScan,
      selectedModelVersion,
      setSelectedModelVersion,
      skipAnnotated,
      setSkipAnnotated,
      start: startScan,
      pause: pauseScan,
      resume: resumeScan,
      cancel: cancelScan,
      deleteScan,
      updateScanResults,
    }),
    [
      cancelScan,
      deleteScan,
      latestScan,
      pauseScan,
      resumeScan,
      scanError,
      scanProgress,
      scanState,
      selectedModelVersion,
      skipAnnotated,
      startScan,
      updateScanResults,
    ],
  );

  useEffect(() => {
    const handleContextMenu = (e: MouseEvent) => {
      e.preventDefault();
    };
    window.addEventListener("contextmenu", handleContextMenu);
    return () => window.removeEventListener("contextmenu", handleContextMenu);
  }, []);

  const pageTitle = useMemo(() => {
    if (workspacePaths.has(location.pathname)) return "图片工作台";
    if (recognitionPaths.has(location.pathname)) return "角色识别";
    if (similarityPaths.has(location.pathname)) return "相似计算";
    return (
      navItems.find((nav) => nav.to === location.pathname)?.label ??
      "图片工作台"
    );
  }, [location.pathname]);

  const runtimeLabel = warming
    ? "AI 推理环境正在启动"
    : status?.database.status === "ready"
      ? "本地核心已连接"
      : isTauriRuntime()
        ? "桌面核心未连接"
        : "浏览器预览模式";

  if (!initialLoaded && loading) {
    return <InitialLoading message="正在初始化本地核心与 AI 推理环境…" />;
  }

  return (
    <div className="app-shell">
      <aside className="sidebar">
        <div className="brand">
          <div>
            <div className="brand-title">Anime Pic Manage</div>
            <div className="brand-subtitle">让图片更有条理</div>
          </div>
        </div>
        <nav className="nav-list" aria-label="主导航">
          {navItems.map(({ to, label, icon: Icon, end }) => (
            <NavLink
              key={to}
              to={to}
              end={end}
              className={({ isActive }) =>
                `nav-item${
                  isActive ||
                  (to === "/" && workspacePaths.has(location.pathname)) ||
                  (to === "/recognition" &&
                    recognitionPaths.has(location.pathname)) ||
                  (to === "/similarity" &&
                    similarityPaths.has(location.pathname))
                    ? " active"
                    : ""
                }`
              }
            >
              <Icon size={18} strokeWidth={1.9} />
              <span>{label}</span>
            </NavLink>
          ))}
        </nav>
        <div className="sidebar-footer">
          <div className="privacy-note">
            <Info size={15} />
            <span>本地管理 · AI 助力</span>
          </div>
          <a
            className="sidebar-link"
            href={issuesUrl}
            target="_blank"
            rel="noreferrer"
          >
            <CircleHelp size={15} />
            问题反馈
            <ChevronRight size={14} />
          </a>
        </div>
      </aside>

      <main className="main-content">
        <header className="topbar">
          <div>
            <div className="eyebrow">工作台 / {pageTitle}</div>
            <h1>{pageTitle}</h1>
            {location.pathname === "/" && (
              <p className="topbar-subtitle">管理本地动漫图片，从这里开始</p>
            )}
            {recognitionPaths.has(location.pathname) && (
              <p className="topbar-subtitle">识别、复核并校正图片中的动漫角色</p>
            )}
          </div>
          <div className="topbar-actions">
            {location.pathname === "/settings" && settingsSave.available && (
              <button
                className="primary-button topbar-save-button"
                onClick={() => void runSettingsSave()}
                disabled={settingsSave.saving || !settingsSave.dirty}
                title={
                  settingsSave.dirty
                    ? "保存全部设置改动"
                    : "当前没有未保存的改动"
                }
              >
                {settingsSave.saving ? (
                  <LoaderCircle size={15} className="spin" />
                ) : (
                  <Save size={15} />
                )}
                保存
              </button>
            )}
            {location.pathname === "/settings" && (
              <button
                className="ghost-button topbar-about-button"
                onClick={() => setAboutOpen(true)}
              >
                <Info size={15} />
                关于与更新
              </button>
            )}
            <span
              className={`runtime-pill ${
                status?.database.status === "ready" ? "ready" : ""
              }`}
            >
              <span className="status-dot" />
              {runtimeLabel}
            </span>
            <button
              className="icon-button"
              onClick={() => void refresh()}
              title="刷新状态"
              aria-label="刷新状态"
            >
              <RefreshCw size={17} className={loading ? "spin" : ""} />
            </button>
          </div>
        </header>

        {error && (
          <div className="alert error">
            <AlertCircle size={17} />
            <span>{error}</span>
            <button onClick={() => void refresh()}>重试</button>
          </div>
        )}

        {warming && !error && (
          <div className="alert info" role="status" aria-live="polite">
            <LoaderCircle size={17} className="spin" />
            <span>
              AI 推理环境正在完成首次初始化；完成后会自动连接，无需手动刷新。
            </span>
          </div>
        )}

        <Routes>
          <Route
            path="/"
            element={
              <WorkspacePage
                status={status}
                loading={loading}
                library={library}
                recentScan={recentScan}
                onChooseLibrary={() => void chooseLibrary()}
                onNavigate={navigate}
                controller={scanController}
              />
            }
          />
          <Route
            path="/recognition"
            element={
              <RecognitionPage
                status={status}
                loading={loading}
                library={library}
                onChooseLibrary={() => void chooseLibrary()}
                onScanCompleted={onScanCompleted}
                controller={scanController}
                autoStart={
                  new URLSearchParams(location.search).get("autostart") === "1"
                }
              />
            }
          />
          <Route
            path="/classification"
            element={
              <RecognitionPage
                initialTab="recognition"
                status={status}
                loading={loading}
                library={library}
                onChooseLibrary={() => void chooseLibrary()}
                onScanCompleted={onScanCompleted}
                controller={scanController}
              />
            }
          />
          <Route
            path="/personal-model"
            element={
              <RecognitionPage
                initialTab="personal"
                status={status}
                loading={loading}
                library={library}
                onChooseLibrary={() => void chooseLibrary()}
                onScanCompleted={onScanCompleted}
                controller={scanController}
              />
            }
          />
          <Route
            path="/review"
            element={
              <RecognitionPage
                initialTab="recognition"
                initialFilter="review"
                status={status}
                loading={loading}
                library={library}
                onChooseLibrary={() => void chooseLibrary()}
                onScanCompleted={onScanCompleted}
                controller={scanController}
              />
            }
          />
          <Route
            path="/similarity"
            element={
              <SimilarityPage
                library={library}
                onChooseLibrary={() => void chooseLibrary()}
                libraryLocked={isScanActive(scanState)}
              />
            }
          />
          <Route
            path="/generate"
            element={<GenerationPage />}
          />
          <Route
            path="/settings"
            element={<SettingsPage status={status} onRefresh={refresh} />}
          />
          <Route
            path="*"
            element={
              <WorkspacePage
                status={status}
                loading={loading}
                library={library}
                recentScan={recentScan}
                onChooseLibrary={() => void chooseLibrary()}
                onNavigate={navigate}
                controller={scanController}
              />
            }
          />
        </Routes>
      </main>
      <AboutUpdateModal
        show={aboutOpen}
        currentVersion={status?.app_version ?? "0.1.0"}
        onClose={() => setAboutOpen(false)}
      />
    </div>
  );
}

export { developerUrl, repositoryUrl, issuesUrl };
export default App;
