from services.testing import batch_runner, evaluate_summative, run_simulation


def test_run_simulation_offline_fallback():
    result = run_simulation("system", "persona", turns=1, params={"persona": "best_case"})
    assert result.persona == "best_case"
    assert result.transcript


def test_evaluate_offline_scores():
    simulation = run_simulation("system", "persona", turns=1, params={"persona": "weak"})
    rubric = {"struktur_klarheit": 20}
    evaluation = evaluate_summative(simulation.transcript, rubric, "weak")
    assert "gesamt" in evaluation.scores


def test_batch_runner_multiple_personas():
    persona_prompts = {"best_case": "prompt", "weak": "prompt"}
    rubric = {"struktur_klarheit": 20}
    results = batch_runner("system", persona_prompts, rubric, turns=1)
    assert len(results) == 2
