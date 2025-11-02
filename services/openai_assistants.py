"""Wrapper for managing OpenAI assistants."""
from __future__ import annotations

import importlib
import importlib.util
import logging
from dataclasses import dataclass
from typing import Dict, List, Optional

from tenacity import retry, stop_after_attempt, wait_exponential


LOGGER = logging.getLogger(__name__)


@dataclass
class AssistantSpec:
    kind: str
    name: str
    instructions: str
    tools: Optional[List[Dict]] = None


def _client():
    if importlib.util.find_spec("openai") is None:
        raise RuntimeError("openai package is required for assistant operations")
    openai_module = importlib.import_module("openai")
    OpenAI = getattr(openai_module, "OpenAI")
    return OpenAI()


@retry(wait=wait_exponential(multiplier=1, min=2, max=30), stop=stop_after_attempt(5))
def create_or_update_assistant(kind: str, name: str, instructions: str, tools: Optional[List[Dict]] = None) -> str:
    """Create or update an assistant for the given scenario."""
    tools = tools or []
    client = _client()
    metadata = {"kind": kind}
    LOGGER.info("Creating/updating assistant: %s", name)
    assistant = client.beta.assistants.create(
        name=name,
        instructions=instructions,
        tools=tools,
        metadata=metadata,
    )
    return assistant.id


__all__ = ["AssistantSpec", "create_or_update_assistant"]
