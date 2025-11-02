"""Utilities for diffing JSON payloads."""
from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Dict, List


@dataclass
class DiffChange:
    path: str
    old: Any
    new: Any


class JsonDiffer:
    """Compute shallow diffs between two JSON-compatible dictionaries."""

    def __init__(self, old: Dict[str, Any], new: Dict[str, Any]):
        self.old = old or {}
        self.new = new or {}

    def diff(self) -> List[DiffChange]:
        changes: List[DiffChange] = []
        keys = set(self.old) | set(self.new)
        for key in sorted(keys):
            old_value = self.old.get(key)
            new_value = self.new.get(key)
            if old_value == new_value:
                continue
            changes.append(DiffChange(path=key, old=old_value, new=new_value))
        return changes

    def render(self) -> str:
        if not self.diff():
            return "Keine Unterschiede zur letzten gespeicherten Version."
        lines = ["Änderungen:"]
        for change in self.diff():
            lines.append(f"- {change.path}: {json.dumps(change.old, ensure_ascii=False)} → {json.dumps(change.new, ensure_ascii=False)}")
        return "\n".join(lines)


__all__ = ["DiffChange", "JsonDiffer"]
