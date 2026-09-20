"""Errors crossing the worker boundary."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(slots=True)
class WorkerError(Exception):
    """A user-facing, serializable worker error."""

    code: str
    message: str
    detail: str | None = None
    retryable: bool = False
    job_id: str | None = None

    def __post_init__(self) -> None:
        Exception.__init__(self, self.message)

    def as_dict(self, request_id: str | None = None) -> dict[str, object]:
        return {
            "code": self.code,
            "message": self.message,
            "detail": self.detail,
            "retryable": self.retryable,
            "request_id": request_id,
            "job_id": self.job_id,
        }


class ProtocolError(WorkerError):
    """Malformed or unsupported IPC message."""

    def __init__(self, message: str, detail: str | None = None) -> None:
        super().__init__("INVALID_REQUEST", message, detail, False)


class ModelError(WorkerError):
    """Model manifest, dependency, or inference error."""


class ModelNotInstalledError(ModelError):
    """The requested adapter cannot run because model assets are absent."""

    def __init__(self, message: str, detail: str | None = None) -> None:
        super().__init__("MODEL_NOT_INSTALLED", message, detail, False)


class ModelManifestError(ModelError):
    """A model manifest or one of its required sidecars is invalid."""

    def __init__(self, message: str, detail: str | None = None) -> None:
        super().__init__("MODEL_MANIFEST_INVALID", message, detail, False)
