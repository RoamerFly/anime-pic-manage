import { invoke } from "@tauri-apps/api/core";
import type { IpcEnvelope, MessageType } from "@anime-pic-manage/shared-types";
import { IPC_SCHEMA_VERSION, createRequest } from "@anime-pic-manage/shared-types";

export function requestEnvelope<TPayload>(messageType: MessageType, payload: unknown = {}): IpcEnvelope<unknown> {
  return createRequest(messageType, payload);
}

function toCamelCaseKey(key: string): string {
  return key.replace(/_([a-z0-9])/g, (_, ch: string) => ch.toUpperCase());
}

export function normalizeIpcArgs(args: Record<string, unknown>, requestId: string): Record<string, unknown> {
  const normalized: Record<string, unknown> = {
    requestId,
  };
  for (const [key, val] of Object.entries(args)) {
    normalized[toCamelCaseKey(key)] = val;
  }
  return normalized;
}

export function settleWithin<T>(task: Promise<T>, timeoutMs: number): Promise<T | null> {
  return new Promise((resolve, reject) => {
    const timer = globalThis.setTimeout(() => resolve(null), timeoutMs);
    task.then(
      (value) => {
        globalThis.clearTimeout(timer);
        resolve(value);
      },
      (error) => {
        globalThis.clearTimeout(timer);
        reject(error);
      },
    );
  });
}

export async function invokeCore<TPayload>(
  command: string,
  messageType: MessageType,
  args: Record<string, unknown> = {},
): Promise<IpcEnvelope<TPayload>> {
  const request = requestEnvelope(messageType, args);
  const normalizedArgs = normalizeIpcArgs(args, request.request_id);
  try {
    return await invoke<IpcEnvelope<TPayload>>(command, normalizedArgs);
  } catch (cause) {
    const detail = cause instanceof Error ? cause.message : String(cause);
    const message =
      detail && detail !== "null" && detail !== "undefined" && detail.trim().length > 0
        ? detail
        : "桌面核心暂不可用，请稍后重试。";
    return {
      version: IPC_SCHEMA_VERSION,
      request_id: request.request_id,
      message_type: messageType,
      error: {
        code: "CORE_UNAVAILABLE",
        message,
        detail,
        retryable: true,
        request_id: request.request_id,
      },
    };
  }
}

export function isTauriRuntime(): boolean {
  return "__TAURI_INTERNALS__" in globalThis;
}

export function normalizeDisplayPath(path: string): string {
  if (/^\\\\\?\\UNC\\/i.test(path)) {
    return `\\\\${path.slice(8)}`;
  }
  if (/^\\\\\?\\/.test(path)) {
    return path.slice(4);
  }
  return path;
}

export function normalizePreviewPath(path: string): string {
  return normalizeDisplayPath(path);
}
