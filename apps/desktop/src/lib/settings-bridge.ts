/**
 * Connects the single topbar save button (rendered by App) with the settings
 * page, which owns the pending values. Keeping one save action avoids the
 * previous per-section buttons that could save conflicting partial states.
 */

export interface SettingsSaveSnapshot {
  /** True while the settings page is mounted and has a save handler. */
  available: boolean;
  saving: boolean;
  dirty: boolean;
}

type SaveHandler = () => Promise<void>;

let handler: SaveHandler | null = null;
let saving = false;
let dirty = false;
let snapshot: SettingsSaveSnapshot = {
  available: false,
  saving: false,
  dirty: false,
};
const listeners = new Set<() => void>();

function publish(): void {
  const next: SettingsSaveSnapshot = {
    available: handler !== null,
    saving,
    dirty,
  };
  if (
    next.available === snapshot.available &&
    next.saving === snapshot.saving &&
    next.dirty === snapshot.dirty
  ) {
    return;
  }
  snapshot = next;
  for (const listener of listeners) listener();
}

export function subscribeSettingsSave(listener: () => void): () => void {
  listeners.add(listener);
  return () => {
    listeners.delete(listener);
  };
}

/** Stable snapshot for `useSyncExternalStore`. */
export function getSettingsSaveSnapshot(): SettingsSaveSnapshot {
  return snapshot;
}

export function registerSettingsSave(next: SaveHandler | null): void {
  handler = next;
  if (next === null) {
    saving = false;
    dirty = false;
  }
  publish();
}

export function updateSettingsSaveState(next: {
  saving: boolean;
  dirty: boolean;
}): void {
  saving = next.saving;
  dirty = next.dirty;
  publish();
}

export async function runSettingsSave(): Promise<void> {
  await handler?.();
}
