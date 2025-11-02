from pathlib import Path

from services.validation import ValidationResult, prompt_lint, validate_scenario_json

SCHEMA_PATH = Path(__file__).resolve().parents[1] / "schemas" / "scenario.schema.json"


def test_validate_scenario_success():
    payload = {
        "tag": "bewerbung_junior_data_analyst",
        "desc": "Simulation eines Bewerbungsgesprächs",
        "info": "Trainiert STAR-Antworten",
        "name": "Bewerbungsgespräch – Junior Data Analyst",
        "message": "<p>Hallo</p>",
        "assistant_id": "",
    }
    result = validate_scenario_json(payload, SCHEMA_PATH)
    assert isinstance(result, ValidationResult)
    assert result.is_valid


def test_validate_scenario_failure():
    payload = {
        "tag": "",
        "desc": "",
        "info": "",
        "name": "",
        "message": "",
    }
    result = validate_scenario_json(payload, SCHEMA_PATH)
    assert not result.is_valid
    assert result.errors


def test_prompt_lint_reports_warning():
    report = prompt_lint("Kurzer Prompt ohne Struktur")
    assert report.warnings
