/**
 * UI font scaling.
 *
 * Every font-size in the stylesheets is written as
 * `calc(N * var(--ui-font-step, 1px))`, so one custom property scales all UI
 * text. The user-facing value is expressed in pixels against a 14px baseline:
 * 14px means the original design size, 15px means roughly 7% larger text.
 */

export const UI_FONT_BASE_PX = 14;
export const UI_FONT_MIN_PX = 12;
export const UI_FONT_MAX_PX = 20;
export const UI_FONT_STEP_PX = 0.5;

const UI_FONT_CACHE_KEY = "anime-pic.ui-font-size";

function roundToStep(value: number): number {
  return Math.round(value * 2) / 2;
}

export function clampUiFontSize(value: number): number {
  if (!Number.isFinite(value)) return UI_FONT_BASE_PX;
  return roundToStep(
    Math.min(UI_FONT_MAX_PX, Math.max(UI_FONT_MIN_PX, value)),
  );
}

export function stepUiFontSize(current: number, direction: -1 | 1): number {
  return clampUiFontSize(clampUiFontSize(current) + direction * UI_FONT_STEP_PX);
}

export function uiFontStepValue(fontSize: number): string {
  return `${(clampUiFontSize(fontSize) / UI_FONT_BASE_PX).toFixed(4)}px`;
}

export function applyUiFontSize(
  fontSize: number,
  root: HTMLElement | null = typeof document === "undefined"
    ? null
    : document.documentElement,
): void {
  if (!root) return;
  root.style.setProperty("--ui-font-step", uiFontStepValue(fontSize));
}

function storage(): Storage | null {
  try {
    return typeof localStorage === "undefined" ? null : localStorage;
  } catch {
    return null;
  }
}

/**
 * Cache the resolved font size so the very first paint already uses it.
 *
 * The persisted value lives in the Rust settings table, which is only readable
 * after an async IPC round trip; without this cache the UI would render at the
 * default size and visibly jump once the settings arrived.
 */
export function cacheUiFontSize(fontSize: number): void {
  const store = storage();
  if (!store) return;
  try {
    store.setItem(UI_FONT_CACHE_KEY, String(clampUiFontSize(fontSize)));
  } catch {
    // Private mode or a full quota must not break the settings page.
  }
}

export function readCachedUiFontSize(): number {
  const store = storage();
  if (!store) return UI_FONT_BASE_PX;
  try {
    const raw = store.getItem(UI_FONT_CACHE_KEY);
    if (raw === null) return UI_FONT_BASE_PX;
    const parsed = Number(raw);
    return Number.isFinite(parsed) ? clampUiFontSize(parsed) : UI_FONT_BASE_PX;
  } catch {
    return UI_FONT_BASE_PX;
  }
}

/** Apply the cached size before the first render to avoid a font flash. */
export function applyCachedUiFontSize(): void {
  applyUiFontSize(readCachedUiFontSize());
}
