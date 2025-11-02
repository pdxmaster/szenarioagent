"""Validation utilities for Trainexus scenario designer."""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List


@dataclass
class ValidationResult:
    """Container for schema validation output."""

    is_valid: bool
    errors: List[str]


def load_schema(schema_path: Path) -> dict:
    """Load a JSON schema from disk.

    Parameters
    ----------
    schema_path: Path
        Path to the schema file.
    """
    if not schema_path.exists():
        raise FileNotFoundError(f"Schema not found: {schema_path}")
    with schema_path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def validate_scenario_json(payload: dict, schema_path: Path) -> ValidationResult:
    """Minimal schema validation without external dependencies."""
    schema = load_schema(schema_path)
    errors: List[str] = []

    required = schema.get("required", [])
    for field in required:
        if field not in payload:
            errors.append(f"{field}: missing required field")
        elif isinstance(payload[field], str) and not payload[field].strip():
            errors.append(f"{field}: must not be empty")

    for field, rules in schema.get("properties", {}).items():
        if field not in payload:
            continue
        value = payload[field]
        if rules.get("type") == "string" and not isinstance(value, str):
            errors.append(f"{field}: expected string")
        if isinstance(value, str) and rules.get("minLength", 0) > 0 and len(value.strip()) < rules["minLength"]:
            errors.append(f"{field}: must have at least {rules['minLength']} characters")

    return ValidationResult(is_valid=not errors, errors=errors)


@dataclass
class PromptLintIssue:
    severity: str
    message: str


@dataclass
class PromptLintReport:
    errors: List[PromptLintIssue]
    warnings: List[PromptLintIssue]

    def as_dict(self) -> dict:
        return {
            "errors": [issue.__dict__ for issue in self.errors],
            "warnings": [issue.__dict__ for issue in self.warnings],
        }


def _lint_length(prompt_text: str) -> Iterable[PromptLintIssue]:
    limit = 6000
    if len(prompt_text) > limit:
        yield PromptLintIssue(
            severity="error",
            message=f"Prompt too long ({len(prompt_text)} chars). Target < {limit} chars.",
        )
    elif len(prompt_text) > 4000:
        yield PromptLintIssue(
            severity="warning",
            message=f"Prompt length {len(prompt_text)} chars. Consider trimming for latency.",
        )


def _lint_role(prompt_text: str) -> Iterable[PromptLintIssue]:
    keywords = ["rolle", "verhalten", "format"]
    missing = [kw for kw in keywords if kw.lower() not in prompt_text.lower()]
    if missing:
        yield PromptLintIssue(
            severity="warning",
            message=f"Prompt is missing recommended sections: {', '.join(missing)}.",
        )


def _lint_safety(prompt_text: str) -> Iterable[PromptLintIssue]:
    risky = ["never provide medical", "never provide legal"]
    if any(phrase in prompt_text.lower() for phrase in risky):
        return []  # already contains guardrails
    if "sicher" not in prompt_text.lower() and "respekt" not in prompt_text.lower():
        yield PromptLintIssue(
            severity="warning",
            message="Prompt should mention safety tone (e.g., 'sei respektvoll').",
        )


def prompt_lint(prompt_text: str) -> PromptLintReport:
    """Simple heuristics to help authors craft safe, concise prompts."""
    errors: List[PromptLintIssue] = []
    warnings: List[PromptLintIssue] = []

    for lint_fn in (_lint_length, _lint_role, _lint_safety):
        for issue in lint_fn(prompt_text):
            if issue.severity == "error":
                errors.append(issue)
            else:
                warnings.append(issue)

    return PromptLintReport(errors=errors, warnings=warnings)


__all__ = [
    "ValidationResult",
    "PromptLintIssue",
    "PromptLintReport",
    "load_schema",
    "validate_scenario_json",
    "prompt_lint",
]
