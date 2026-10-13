"""Machine-stable, non-silent failure classification; never serialize secrets."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import Any


class ErrorCategory(StrEnum):
    CONFIG = "config"
    CONTRACT = "contract"
    DEPENDENCY = "dependency"
    PROVIDER = "provider"
    DATA = "data"
    RUNTIME = "runtime"
    TRANSIENT = "transient"
    SECURITY = "security"


EXIT_CODES = {
    ErrorCategory.CONFIG: 2,
    ErrorCategory.CONTRACT: 4,
    ErrorCategory.DEPENDENCY: 3,
    ErrorCategory.PROVIDER: 3,
    ErrorCategory.DATA: 4,
    ErrorCategory.RUNTIME: 70,
    ErrorCategory.TRANSIENT: 75,
    ErrorCategory.SECURITY: 78,
}


@dataclass(frozen=True, slots=True)
class ErrorEnvelope:
    category: ErrorCategory
    code: str
    message: str
    retryable: bool
    run_id: str | None = None
    cause_code: str | None = None
    schema_version: str = "0.4.0"

    def __post_init__(self) -> None:
        from .validation import require_hash
        import re
        if type(self.category) is not ErrorCategory:
            raise TypeError("category must be ErrorCategory")
        if type(self.retryable) is not bool:
            raise TypeError("retryable must be bool")
        if self.schema_version != "0.4.0":
            raise ValueError("error schema version mismatch")
        if self.run_id is not None:
            require_hash(self.run_id, "run_id")
        if self.cause_code is not None and (type(self.cause_code) is not str or
                                            re.fullmatch(r"[A-Z][A-Z0-9_]*", self.cause_code) is None):
            raise ValueError("cause_code must be UPPER_SNAKE_CASE")
        if type(self.code) is not str or re.fullmatch(r"[A-Z][A-Z0-9_]*", self.code) is None:
            raise ValueError("error code must be UPPER_SNAKE_CASE")
        if type(self.message) is not str or not self.message or len(self.message) > 1024:
            raise ValueError("error message missing or too long")
        if self.retryable != (self.category == ErrorCategory.TRANSIENT):
            raise ValueError("retryable must match error category")

    @property
    def exit_code(self) -> int:
        return EXIT_CODES[self.category]

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "category": self.category.value,
            "code": self.code,
            "message": self.message,
            "retryable": self.retryable,
            "run_id": self.run_id,
            "cause_code": self.cause_code,
            "exit_code": self.exit_code,
        }


class RLESError(Exception):
    """An explicit stop, with an intentionally redacted public message."""

    def __init__(
        self,
        category: ErrorCategory,
        code: str,
        message: str,
        *,
        run_id: str | None = None,
        cause_code: str | None = None,
    ) -> None:
        self.envelope = ErrorEnvelope(
            category=category,
            code=code,
            message=message,
            retryable=category == ErrorCategory.TRANSIENT,
            run_id=run_id,
            cause_code=cause_code,
        )
        super().__init__(message)

    @property
    def exit_code(self) -> int:
        return self.envelope.exit_code


def error_json(error: RLESError) -> str:
    import json

    return json.dumps(error.envelope.as_dict(), ensure_ascii=True, sort_keys=True, separators=(",", ":"))


def parse_error_json(blob: bytes) -> ErrorEnvelope:
    """Verify an exact canonical CLI error envelope, including derived exit status."""
    import json
    from .serialization import _no_duplicate_pairs

    if type(blob) is not bytes:
        raise TypeError("error JSON must be bytes")
    data = json.loads(blob.decode("ascii"), object_pairs_hook=_no_duplicate_pairs,
                      parse_constant=lambda x: (_ for _ in ()).throw(ValueError(x)))
    expected = {"category", "code", "message", "retryable", "run_id", "cause_code",
                "schema_version", "exit_code"}
    if type(data) is not dict or set(data) != expected:
        raise ValueError("error envelope unknown/missing fields")
    if type(data["category"]) is not str:
        raise TypeError("error category must be string")
    envelope = ErrorEnvelope(category=ErrorCategory(data["category"]), code=data["code"],
                             message=data["message"], retryable=data["retryable"],
                             run_id=data["run_id"], cause_code=data["cause_code"],
                             schema_version=data["schema_version"])
    if type(data["exit_code"]) is not int or data["exit_code"] != envelope.exit_code:
        raise ValueError("error exit code mismatch")
    if json.dumps(data, ensure_ascii=True, sort_keys=True,
                  separators=(",", ":")).encode("ascii") != blob:
        raise ValueError("noncanonical error JSON")
    return envelope
