from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Literal, Mapping, Sequence


# Importable vocabulary for callers; persisted events accept future kinds too.
KnownEventKind = Literal[
    "user",
    "assistant",
    "tool",
    "usage",
    "approval",
    "reflection",
    "compact",
    "compaction_llm",
    "goal",
]


@dataclass(frozen=True, slots=True)
class DurableEvent:
    """Forward-compatible view over one persisted event."""

    kind: str
    seq: int
    payload: Mapping[str, Any]

    @classmethod
    def parse(
        cls,
        value: Mapping[str, Any],
        *,
        fallback_seq: int,
    ) -> "DurableEvent":
        copied = deepcopy(dict(value))

        raw_seq = copied.get("seq")
        seq = (
            raw_seq
            if isinstance(raw_seq, int) and not isinstance(raw_seq, bool)
            else fallback_seq
        )

        kind = str(copied.get("kind") or "")
        return cls(
            kind=kind,
            seq=seq,
            payload=MappingProxyType(copied),
        )

    def get(self, key: str, default: Any = None) -> Any:
        return self.payload.get(key, default)

    def as_dict(self) -> dict[str, Any]:
        return deepcopy(dict(self.payload))


@dataclass(frozen=True, slots=True)
class UserTurn:
    seq: int
    content: str


@dataclass(frozen=True, slots=True)
class RollbackCheckpoint:
    thread_id: str
    user_seq: int
    git_hash: str | None
    modified_paths: tuple[str, ...]
    memory_snapshot: str


@dataclass(frozen=True, slots=True)
class ToolMutationEvent:
    tool: str
    arguments: Mapping[str, Any]
    result: str
    exit_status: str

    @classmethod
    def parse(cls, event: DurableEvent) -> ToolMutationEvent | None:
        if event.kind != "tool":
            return None
        arguments = event.get("args")
        return cls(
            tool=str(event.get("tool") or ""),
            arguments=(
                MappingProxyType(dict(arguments))
                if isinstance(arguments, Mapping)
                else MappingProxyType({})
            ),
            result=str(event.get("result") or ""),
            exit_status=str(event.get("exit") or ""),
        )


def user_event(
    content: str,
    *,
    images: Sequence[str] = (),
) -> dict[str, Any]:
    event: dict[str, Any] = {"kind": "user", "content": content}
    if images:
        event["images"] = list(images)
    return event


def compact_event(data: Mapping[str, Any]) -> dict[str, Any]:
    reason = (
        data.get("notice_reason")
        or data.get("trigger")
        or data.get("skip_reason")
        or data.get("reason")
        or "unknown"
    )
    info = str(data.get("info") or "").strip()
    forced = " [forced]" if data.get("forced") else ""
    content = f"compaction ({reason}){forced}: {info}".rstrip(": ")
    return {"kind": "compact", "content": content}


def parse_events(rows: Sequence[Mapping[str, Any]]) -> list[DurableEvent]:
    return [
        DurableEvent.parse(row, fallback_seq=index) for index, row in enumerate(rows)
    ]
