export interface AnnotationHistoryUpdate<T> {
  snapshots: T[][];
  historyIndex: number;
}

/**
 * Append a committed annotation snapshot while preserving undo/redo branch
 * semantics and bounding memory usage.  Inputs are never mutated; callers can
 * keep ownership of the snapshots they passed in.
 */
export function appendAnnotationHistory<T>(
  history: ReadonlyArray<ReadonlyArray<T>>,
  historyIndex: number,
  nextSnapshot: ReadonlyArray<T>,
  maxEntries: number,
): AnnotationHistoryUpdate<T> {
  if (!Number.isInteger(maxEntries) || maxEntries < 1) {
    throw new RangeError("maxEntries must be a positive integer");
  }

  const branchEnd = Math.min(Math.max(historyIndex + 1, 0), history.length);
  const branched = [...history.slice(0, branchEnd), [...nextSnapshot]];
  const firstSnapshot = Math.max(0, branched.length - maxEntries);
  const snapshots = branched.slice(firstSnapshot).map((snapshot) => [...snapshot]);
  return { snapshots, historyIndex: snapshots.length - 1 };
}
