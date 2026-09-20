export type ScanState =
  | "idle"
  | "running"
  | "pausing"
  | "paused"
  | "cancelling"
  | "completed"
  | "cancelled"
  | "failed";

/** Map progress events to task state without treating a per-image error as a task failure. */
export function scanStateFromPhase(phase: string): ScanState {
  if (phase === "pausing") return "pausing";
  if (phase === "paused") return "paused";
  if (phase === "cancelling") return "cancelling";
  if (phase === "cancelled") return "cancelled";
  if (phase === "complete" || phase === "completed") return "completed";
  if (phase === "error") return "running";
  return "running";
}

export function isScanActive(state: ScanState): boolean {
  return state === "running" || state === "pausing" || state === "paused" || state === "cancelling";
}
