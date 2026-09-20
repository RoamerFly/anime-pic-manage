"""Versioned JSON Lines IPC types and validation."""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any
from uuid import uuid4

from .errors import ProtocolError, WorkerError

SCHEMA_VERSION = "1.0"
_REQUIRED_FIELDS = ("schema_version", "request_id", "task_id", "message_type", "payload")


def _string(value: object, field_name: str, *, allow_empty: bool = False) -> str:
    if not isinstance(value, str) or (not allow_empty and not value.strip()):
        raise ProtocolError(f"字段 {field_name} 必须是非空字符串")
    return value


def _payload(value: object) -> dict[str, Any]:
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise ProtocolError("字段 payload 必须是 JSON 对象")
    return dict(value)


@dataclass(frozen=True, slots=True)
class WorkerRequest:
    schema_version: str
    request_id: str
    task_id: str
    message_type: str
    payload: dict[str, Any] = field(default_factory=dict)
    error: dict[str, Any] | None = None

    @classmethod
    def from_dict(cls, value: Mapping[str, object]) -> WorkerRequest:
        missing = [key for key in _REQUIRED_FIELDS if key not in value]
        if missing:
            raise ProtocolError(f"请求缺少字段: {', '.join(missing)}")
        schema_version = value["schema_version"]
        if schema_version not in (SCHEMA_VERSION, 1, "1"):
            raise ProtocolError(f"不支持的 schema_version: {schema_version!r}")
        error = value.get("error")
        if error is not None and not isinstance(error, dict):
            raise ProtocolError("字段 error 必须是对象或 null")
        return cls(
            schema_version=SCHEMA_VERSION,
            request_id=_string(value["request_id"], "request_id"),
            task_id=_string(value["task_id"], "task_id"),
            message_type=_string(value["message_type"], "message_type"),
            payload=_payload(value["payload"]),
            error=dict(error) if isinstance(error, dict) else None,
        )

    @classmethod
    def from_json_line(cls, line: str) -> WorkerRequest:
        try:
            parsed = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ProtocolError("请求不是有效的 JSON", str(exc)) from exc
        if not isinstance(parsed, dict):
            raise ProtocolError("请求根节点必须是 JSON 对象")
        return cls.from_dict(parsed)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "request_id": self.request_id,
            "task_id": self.task_id,
            "message_type": self.message_type,
            "payload": self.payload,
            "error": self.error,
        }


@dataclass(frozen=True, slots=True)
class WorkerResponse:
    schema_version: str
    request_id: str
    task_id: str
    message_type: str
    payload: dict[str, Any] = field(default_factory=dict)
    error: dict[str, Any] | None = None

    @classmethod
    def ok(
        cls,
        request: WorkerRequest,
        payload: Mapping[str, Any] | None = None,
        *,
        message_type: str | None = None,
    ) -> WorkerResponse:
        return cls(
            SCHEMA_VERSION,
            request.request_id,
            request.task_id,
            message_type or f"{request.message_type}.result",
            dict(payload or {}),
            None,
        )

    @classmethod
    def failure(cls, request: WorkerRequest, error: WorkerError) -> WorkerResponse:
        return cls(
            SCHEMA_VERSION,
            request.request_id,
            request.task_id,
            f"{request.message_type}.result",
            {},
            error.as_dict(request.request_id),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "request_id": self.request_id,
            "task_id": self.task_id,
            "message_type": self.message_type,
            "payload": self.payload,
            "error": self.error,
        }

    def to_json_line(self) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False, separators=(",", ":"))


def make_request(
    message_type: str, payload: Mapping[str, Any] | None = None, *, task_id: str | None = None
) -> WorkerRequest:
    """Create a request for tests and embedding callers."""

    return WorkerRequest(
        SCHEMA_VERSION,
        str(uuid4()),
        task_id or str(uuid4()),
        message_type,
        dict(payload or {}),
        None,
    )
