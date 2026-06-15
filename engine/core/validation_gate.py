"""
engine/core/validation_gate.py — schema enforcement at EVERY inter-agent handoff.

Malformed output halts the job (raises HandoffHalted) and writes a dead-letter;
it is NEVER passed downstream. This is the wall that stops the multi-agent
error cascade.
"""
from __future__ import annotations

import json
from typing import Any, Optional, Type, TypeVar

from pydantic import BaseModel, ValidationError

from engine.core.dead_letter import DeadLetterSink

T = TypeVar("T", bound=BaseModel)


class HandoffHalted(Exception):
    """A handoff failed validation. The job stops here; nothing flows on."""

    def __init__(self, step: str, reason: str, errors: Any = None):
        self.step = step
        self.reason = reason
        self.errors = errors
        super().__init__(f"handoff '{step}' halted: {reason}")


class ValidationGate:
    def __init__(self, dead_letter_sink: Optional[DeadLetterSink] = None):
        self._sink = dead_letter_sink

    async def validate(self, model_cls: Type[T], raw: Any, *,
                       job_id: Optional[str], step: str) -> T:
        # Accept an already-validated model, a dict, or a JSON string (agents
        # commonly emit a JSON string). Malformed JSON halts the job.
        data: Any = raw
        if isinstance(raw, model_cls):
            return raw
        if isinstance(raw, (str, bytes, bytearray)):
            try:
                data = json.loads(raw)
            except (json.JSONDecodeError, ValueError) as exc:
                return await self._halt(job_id, step, f"invalid JSON: {exc}", raw)

        try:
            return model_cls.model_validate(data)
        except ValidationError as exc:
            return await self._halt(job_id, step, "schema validation failed",
                                    exc.errors(), exc)
        except (TypeError, ValueError) as exc:
            return await self._halt(job_id, step, f"invalid payload: {exc}", None, exc)

    async def _halt(self, job_id, step, reason, errors=None, exc=None):
        if self._sink is not None:
            await self._sink.record(
                job_id=job_id, step=step, error=reason,
                context={"errors": _jsonable(errors)} if errors is not None else {},
            )
        raise HandoffHalted(step, reason, errors) from exc


def _jsonable(errors: Any) -> Any:
    try:
        json.dumps(errors)
        return errors
    except (TypeError, ValueError):
        return str(errors)
