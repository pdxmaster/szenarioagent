from __future__ import annotations

import csv
import json
import os
from datetime import datetime
from pathlib import Path
from typing import Dict, List

import streamlit as st

from services.testing import batch_runner

REPORT_DIR = Path(__file__).resolve().parents[2] / "data" / "ci_reports"
REPORT_DIR.mkdir(parents=True, exist_ok=True)


@st.cache_data(show_spinner=False)
def _load_scenarios(dry_run: bool) -> List[Dict]:
    if dry_run:
        return [
            {
                "tag": "bewerbung_junior_data_analyst",
                "name": "Bewerbungsgespräch – Junior Data Analyst",
                "json_text": json.dumps(
                    {
                        "tag": "bewerbung_junior_data_analyst",
                        "desc": "Simulation eines Bewerbungsgesprächs",
                        "info": "Dummy dry-run Szenario",
                        "name": "Bewerbungsgespräch – Junior Data Analyst",
                        "message": "",
                    }
                ),
            }
        ]
    from services.db_mysql import MySQLClient, MySQLConfig
    mysql_config = MySQLConfig(
        host=os.getenv("MYSQL_HOST", "127.0.0.1"),
        port=int(os.getenv("MYSQL_PORT", "3306")),
        user=os.getenv("MYSQL_USER", "root"),
        password=os.getenv("MYSQL_PASS", ""),
        database=os.getenv("MYSQL_DB", "trainexus"),
    )
    client = MySQLClient(mysql_config)
    groups = client.get_groups(os.getenv("CI_SERVER", "DEV"))
    scenarios: List[Dict] = []
    for group in groups:
        for scenario in client.get_scenarios(group["id"]):
            payload = client.load_scenario_json(scenario["id"]) or {}
            payload["group"] = group["name"]
            scenarios.append(payload)
    return scenarios


def _persona_prompts() -> Dict[str, str]:
    template = (
        "Rolle: Du simulierst Bewerber:innen. Persona: {persona}. Halte dich an die Beschreibung aus der Dokumentation."
    )
    personas = {
        "best_case": "best_case",
        "weak": "weak",
        "zero_knowledge": "zero_knowledge",
    }
    return {key: template.format(persona=value) for key, value in personas.items()}


st.title("CI Regression Checks")
with st.sidebar:
    dry_run = st.checkbox("Dry-Run", value=True)
    turns = st.slider("Turns pro Simulation", 2, 8, 3)
    rubric = {
        "struktur_klarheit": 20,
        "passung_beispiele": 25,
        "fachliche_korrektheit": 20,
        "kommunikation": 20,
        "reflexion": 15,
    }

scenarios = _load_scenarios(dry_run)
if st.button("Regression starten"):
    rows: List[Dict[str, str]] = []
    with st.spinner("Simuliere ..."):
        for scenario in scenarios:
            main_prompt = scenario.get("main_prompt") or ""
            if not main_prompt:
                continue
            persona_prompts = _persona_prompts()
            evaluations = batch_runner(main_prompt, persona_prompts, rubric, turns=turns)
            for eval_result in evaluations:
                rows.append(
                    {
                        "scenario": scenario.get("tag", "unknown"),
                        "persona": eval_result.persona,
                        "score": str(eval_result.scores.get("gesamt", 0)),
                        "passed": "yes" if eval_result.passed else "no",
                    }
                )
    if rows:
        timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        file_path = REPORT_DIR / f"ci_{timestamp}.csv"
        with file_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=["scenario", "persona", "score", "passed"])
            writer.writeheader()
            writer.writerows(rows)
        st.success(f"Bericht gespeichert: {file_path}")
        st.dataframe(rows)
    else:
        st.warning("Keine Szenarien mit Main Prompt gefunden.")

st.caption("Vergleicht Scores mit früheren Läufen, sobald historische Daten vorliegen.")
