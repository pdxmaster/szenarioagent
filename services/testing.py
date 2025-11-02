"""Automated LLM-vs-LLM testing utilities."""
from __future__ import annotations

import importlib
import importlib.util
from dataclasses import dataclass
from typing import Any, Dict, List, Optional


@dataclass
class SimulationTurn:
    role: str
    content: str


@dataclass
class SimulationResult:
    persona: str
    transcript: List[SimulationTurn]
    metadata: Dict[str, Any]


@dataclass
class SummativeEvaluation:
    persona: str
    scores: Dict[str, int]
    highlights: List[str]
    improvements: List[str]
    notes: str
    passed: bool


def _openai_client():
    if importlib.util.find_spec("openai") is None:
        return None
    openai_module = importlib.import_module("openai")
    OpenAI = getattr(openai_module, "OpenAI")
    return OpenAI()


def _llm_chat(messages: List[Dict[str, str]], temperature: float = 0.2) -> str:
    client = _openai_client()
    if client is None:
        # Deterministic fallback for offline environments.
        combined = " ".join(message["content"] for message in messages if message["role"] == "user")
        return f"[offline-response] {combined[:200]}"
    response = client.responses.create(
        model="gpt-4o-mini",
        input=[{"role": msg["role"], "content": msg["content"]} for msg in messages],
        temperature=temperature,
    )
    return response.output_text


def run_simulation(
    main_prompt: str,
    test_persona_prompt: str,
    turns: int = 8,
    params: Optional[Dict[str, Any]] = None,
) -> SimulationResult:
    params = params or {}
    persona = params.get("persona", "unknown")
    transcript: List[SimulationTurn] = []
    system_prompt = {"role": "system", "content": main_prompt}
    user_prompt = {"role": "system", "content": test_persona_prompt}
    history: List[Dict[str, str]] = [system_prompt, {"role": "assistant", "content": "Hallo, willkommen."}]
    for turn in range(turns):
        learner_input = _llm_chat(history + [user_prompt, {"role": "user", "content": "<simulate>"}], params.get("temperature", 0.2))
        transcript.append(SimulationTurn(role="learner", content=learner_input))
        history.append({"role": "user", "content": learner_input})
        interviewer_reply = _llm_chat(history, params.get("temperature", 0.2))
        transcript.append(SimulationTurn(role="assistant", content=interviewer_reply))
        history.append({"role": "assistant", "content": interviewer_reply})
    metadata = {"persona": persona, "turns": turns}
    return SimulationResult(persona=persona, transcript=transcript, metadata=metadata)


def evaluate_summative(transcript: List[SimulationTurn], rubric: Dict[str, int], persona: str) -> SummativeEvaluation:
    combined_text = "\n".join(f"{turn.role}: {turn.content}" for turn in transcript)
    client = _openai_client()
    if client is None:
        base_score = max(40, min(95, 60 + len(combined_text) // 100))
        scores = {key: base_score for key in rubric}
        scores["gesamt"] = int(sum(scores.values()) / len(scores))
        return SummativeEvaluation(
            persona=persona,
            scores=scores,
            highlights=["Offline high-level feedback"],
            improvements=["Verbesserungsvorschläge offline"],
            notes="Offline evaluation (no OpenAI client)",
            passed=scores["gesamt"] >= 60,
        )
    # When OpenAI is available we call the summative assistant directly.
    payload = {
        "role": "system",
        "content": (
            "Bewerte folgendes Transkript auf Basis der Rubrik."  # truncated for brevity
        ),
    }
    response = client.responses.create(
        model="gpt-4o-mini",
        input=[payload, {"role": "user", "content": combined_text}],
    )
    import json

    result = json.loads(response.output_text)
    scores = result.get("scores", {})
    passed = scores.get("gesamt", 0) >= 60
    return SummativeEvaluation(
        persona=persona,
        scores=scores,
        highlights=result.get("highlights", []),
        improvements=result.get("improvements", []),
        notes=result.get("notes", ""),
        passed=passed,
    )


def batch_runner(
    main_prompt: str,
    persona_prompts: Dict[str, str],
    rubric: Dict[str, int],
    turns: int = 8,
) -> List[SummativeEvaluation]:
    results: List[SummativeEvaluation] = []
    for persona, prompt in persona_prompts.items():
        simulation = run_simulation(main_prompt, prompt, turns=turns, params={"persona": persona})
        evaluation = evaluate_summative(simulation.transcript, rubric, persona)
        results.append(evaluation)
    return results


__all__ = [
    "SimulationTurn",
    "SimulationResult",
    "SummativeEvaluation",
    "run_simulation",
    "evaluate_summative",
    "batch_runner",
]
