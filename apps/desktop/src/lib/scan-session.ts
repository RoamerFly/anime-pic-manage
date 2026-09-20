import type { LibraryScanPayload } from "@anime-pic-manage/shared-types";

export interface RecentScanSummary {
  directory: string;
  completedAt: string;
  totalDiscovered: number;
  processed: number;
  resultCount: number;
  personCount: number;
  errorCount: number;
  cancelled: boolean;
  modelVersion?: number | null;
  modelName?: string;
}

export interface StoredScan {
  payload: LibraryScanPayload;
  completedAt: string;
}

export const SCAN_STORAGE_KEY_LEGACY = "anime-pic-manage.recent-scan.v1";
export const SCAN_STORAGE_KEY_LATEST = "anime-pic-manage.recent-scan.v2";
export const SCAN_STORAGE_DIR_PREFIX = "anime-pic-manage.scan-by-dir.v2:";

function normalizeDirectoryKey(dir: string): string {
  return dir.replace(/\\/g, "/").toLowerCase().trim();
}

const memoryStorage: Record<string, string> = {};

function storageGet(key: string): string | null {
  try {
    if (typeof globalThis.localStorage !== "undefined" && globalThis.localStorage) {
      const val = globalThis.localStorage.getItem(key);
      if (val !== null) return val;
    }
  } catch { /* ignore */ }
  try {
    if (typeof globalThis.sessionStorage !== "undefined" && globalThis.sessionStorage) {
      const val = globalThis.sessionStorage.getItem(key);
      if (val !== null) return val;
    }
  } catch { /* ignore */ }
  return memoryStorage[key] ?? null;
}

function storageSet(key: string, value: string): void {
  memoryStorage[key] = value;
  try {
    globalThis.localStorage?.setItem(key, value);
  } catch {
    try {
      globalThis.sessionStorage?.setItem(key, value);
    } catch {
      /* storage might be full or restricted */
    }
  }
}

function storageRemove(key: string): void {
  delete memoryStorage[key];
  try {
    globalThis.localStorage?.removeItem(key);
  } catch { /* ignore */ }
  try {
    globalThis.sessionStorage?.removeItem(key);
  } catch { /* ignore */ }
}

export function scanSummary(payload: LibraryScanPayload, completedAt: string): RecentScanSummary {
  return {
    directory: payload.directory,
    completedAt,
    totalDiscovered: payload.total_discovered,
    processed: payload.processed,
    resultCount: payload.results.length,
    personCount: payload.results.reduce((total, result) => total + result.people.length, 0),
    errorCount: payload.errors.length,
    cancelled: payload.cancelled,
    modelVersion: payload.model_version,
    modelName: payload.model_name,
  };
}

export function parseStoredScan(value: unknown): StoredScan | null {
  if (!value || typeof value !== "object") return null;
  const raw = value as { payload?: unknown; completedAt?: unknown };
  if (!raw.payload || typeof raw.payload !== "object" || typeof raw.completedAt !== "string") return null;
  const payload = raw.payload as Partial<LibraryScanPayload>;
  if (typeof payload.directory !== "string" || !Array.isArray(payload.results) || !Array.isArray(payload.errors)) return null;
  return { payload: payload as LibraryScanPayload, completedAt: raw.completedAt };
}

export function readStoredScan(): StoredScan | null {
  const v2 = storageGet(SCAN_STORAGE_KEY_LATEST);
  if (v2) {
    try {
      const parsed = parseStoredScan(JSON.parse(v2));
      if (parsed) return parsed;
    } catch { /* ignore */ }
  }
  const v1 = storageGet(SCAN_STORAGE_KEY_LEGACY);
  if (v1) {
    try {
      return parseStoredScan(JSON.parse(v1));
    } catch { /* ignore */ }
  }
  return null;
}

export function readRecentScanSummary(): RecentScanSummary | null {
  const stored = readStoredScan();
  return stored ? scanSummary(stored.payload, stored.completedAt) : null;
}

export function scanForDirectory(stored: StoredScan | null, directory: string | undefined): LibraryScanPayload | null {
  if (!directory) return null;
  const target = normalizeDirectoryKey(directory);
  if (stored && normalizeDirectoryKey(stored.payload.directory) === target) {
    return stored.payload;
  }
  return null;
}

export function readStoredScanForDirectory(directory: string | undefined): LibraryScanPayload | null {
  if (!directory) return null;
  const dirKey = `${SCAN_STORAGE_DIR_PREFIX}${normalizeDirectoryKey(directory)}`;
  const raw = storageGet(dirKey);
  if (raw) {
    try {
      const parsed = parseStoredScan(JSON.parse(raw));
      if (parsed) return parsed.payload;
    } catch { /* ignore */ }
  }
  return scanForDirectory(readStoredScan(), directory);
}

export function persistScan(payload: LibraryScanPayload): string {
  const completedAt = payload.completed_at || new Date().toISOString();
  const record: StoredScan = { payload: { ...payload, completed_at: completedAt }, completedAt };
  const jsonStr = JSON.stringify(record);

  storageSet(SCAN_STORAGE_KEY_LATEST, jsonStr);
  if (payload.directory) {
    const dirKey = `${SCAN_STORAGE_DIR_PREFIX}${normalizeDirectoryKey(payload.directory)}`;
    storageSet(dirKey, jsonStr);
  }
  return completedAt;
}

/**
 * Delete stored scan result for a directory, or current latest scan if not specified.
 */
export function deleteStoredScan(directory?: string): boolean {
  if (directory) {
    const dirKey = `${SCAN_STORAGE_DIR_PREFIX}${normalizeDirectoryKey(directory)}`;
    storageRemove(dirKey);
    const latest = readStoredScan();
    if (latest && normalizeDirectoryKey(latest.payload.directory) === normalizeDirectoryKey(directory)) {
      storageRemove(SCAN_STORAGE_KEY_LATEST);
      storageRemove(SCAN_STORAGE_KEY_LEGACY);
    }
    return true;
  }
  storageRemove(SCAN_STORAGE_KEY_LATEST);
  storageRemove(SCAN_STORAGE_KEY_LEGACY);
  return true;
}

export function clearAllStoredScans(): void {
  storageRemove(SCAN_STORAGE_KEY_LATEST);
  storageRemove(SCAN_STORAGE_KEY_LEGACY);
  for (const key of Object.keys(memoryStorage)) {
    delete memoryStorage[key];
  }
  try {
    const storage = globalThis.localStorage;
    if (storage) {
      const keysToRemove: string[] = [];
      for (let i = 0; i < storage.length; i++) {
        const key = storage.key(i);
        if (key && key.startsWith(SCAN_STORAGE_DIR_PREFIX)) {
          keysToRemove.push(key);
        }
      }
      for (const key of keysToRemove) {
        storage.removeItem(key);
      }
    }
  } catch { /* ignore */ }
}
