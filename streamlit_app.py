from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import streamlit as st
import yaml

from services.diff import JsonDiffer
from services.openai_assistants import create_or_update_assistant
from services.testing import batch_runner
from services.validation import prompt_lint, validate_scenario_json

SCHEMA_PATH = Path(__file__).resolve().parent / "schemas" / "scenario.schema.json"
CONFIG_PATH = Path(__file__).resolve().parent / "config" / "servers.yaml"
WELCOME_SVG = (
    "Herzlich willkommen! In diesem Szenario simulieren Sie ein Bewerbungsgespräch für die Rolle "
    "<b>Junior Data Analyst</b>. "
    "Bitte starten Sie mit einer kurzen Selbstvorstellung und klicken Sie zum Sprechen auf den Mikrofon-Button.\n"
    "<svg xmlns=\"http://www.w3.org/2000/svg\" width=\"20\" height=\"20\" viewBox=\"0 0 200 200\" style=\"vertical-align: middle; margin-right: 4px; display: inline;\">\n"
    "  <rect width=\"200\" height=\"200\" rx=\"24\" fill=\"#00a2ed\"/>\n"
    "  <path d=\"M 0 90 Q 40 90, 70 60 L 140 0 L 160 0 L 90 70 Q 60 100, 90 130 L 160 200 L 140 200 L 70 140 Q 30 100, 0 100 Z\" fill=\"#e5e5e5\"/>\n"
    "</svg>\n"
    "<b>Interview-Modus</b>: Der Avatar stellt Fragen, Sie antworten. \n"
    "Nach dem Gespräch erhalten Sie auf der Auswertungsseite detailliertes Feedback."
)
PROMPT_TEMPLATE_MAIN = """Rolle: Du bist Interviewer:in für ein Bewerbungsgespräch [Rolle: Junior Data Analyst].
Zielgruppe: Bewerber:innen auf Einstiegsniveau.
Didaktisches Ziel: {didactic_goal}
Muss-Vorgaben: {must_rules}
Tabus: {never_rules}

Verhalten:
- Stelle nacheinander kurze, klare Fragen (max. 1 Frage pro Turn).
- Fordere bei generischen Antworten konkrete STAR-Beispiele ein (Situation, Task, Action, Result).
- Passe Nachfragen an die Antwortqualität an.
- Sei respektvoll, ohne Lösungen vorzugeben.
- Keine Fachbegriffe erklären, bevor die Person es versucht hat.

Beurteilungsfokus (nur intern, nicht nennen):
{rubric}

Kontextwissen (RAG Titel:Typ):
{rag_docs}

Format:
- Ausgang nur als Interviewer: Eine einzelne, kurze Frage/Anweisung (1–2 Sätze).
- Wenn Antwort komplett off-topic ist, bringe höflich zurück zum Thema.
"""
PROMPT_TEMPLATE_TEST = """Rolle: Du simulierst Bewerber:innen mit variabler Qualität.
Persona: {persona_hint}
Verhalten:
- Antworte realistisch aus Kandidatensicht.
- Bei 'best_case': nutze STAR, bleib prägnant (<120 Wörter).
- Bei 'weak/zero_knowledge': unpräzise, wenig Beispiele.
- Bei 'off_topic': thematisch abdriften.
- Bei 'trolling': provokant, aber nicht beleidigend.

Sprache: gleiche Sprache wie Interviewer:in.
"""
PROMPT_TEMPLATE_FORMATIVES = """Rolle: Coach nach dem Gespräch.
Eingabe: Gesprächstranskript (max. 12 Turns), didaktisches Ziel, Rubrik.
Aufgabe: Gib 3–5 kurze, konkrete Tipps für die nächste Antwort-Runde. 
- Verweise wenn sinnvoll auf STAR.
- Kein Urteilston, keine Noten, keine Wiederholung ganzer Antworten.
- Max. 120 Wörter.
"""
PROMPT_TEMPLATE_SUMMATIVES = """Rolle: Strenger, fairer Gutachter.
Eingaben:
- Gesprächstranskript
- Didaktisches Ziel
- Rubrik mit Gewichten
- Relevante RAG-Kernaussagen (Kurz-Summary der angehängten Dokumente, falls vorhanden)

Aufgabe:
1) Scoring je Kategorie (0–100) + gewichtetes Gesamtergebnis.
2) 2–3 stärkste Punkte, 2–3 wichtigste Verbesserungen (handlungsorientiert).
3) Max. 1 fachliche Korrektur (falls notwendig, mit Quelle falls im RAG).
4) JSON-Ausgabe strikt:

{
  "scores": {
    "struktur_klarheit": int,
    "passung_beispiele": int,
    "fachliche_korrektheit": int,
    "kommunikation": int,
    "reflexion": int,
    "gesamt": int
  },
  "highlights": [string, string, string],
  "improvements": [string, string, string],
  "notes": string
}
"""


@dataclass
class PersonaDefinition:
    key: str
    label: str
    description: str


PERSONAS = [
    PersonaDefinition("best_case", "Best Case", "Vorbereitet, prägnant, 2–3 STAR-Beispiele"),
    PersonaDefinition("weak", "Schwach", "Ausschweifend, unpräzise, wenig Belege"),
    PersonaDefinition("zero_knowledge", "Zero Knowledge", "Unsicher, kurze Antworten, fachlich dünn"),
    PersonaDefinition("off_topic", "Off Topic", "Driftet zu irrelevanten Hobbys"),
    PersonaDefinition("trolling", "Trolling", "Provoziert, widerspricht unnötig"),
]


def load_servers() -> Dict[str, Dict[str, Any]]:
    with CONFIG_PATH.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def init_state() -> None:
    if "initialized" in st.session_state:
        return
    st.session_state.initialized = True
    st.session_state.current_page = 0
    st.session_state.server_key = "DEV"
    st.session_state.mode = "A"
    st.session_state.dry_run = True
    st.session_state.didactics = {
        "didactic_goal": "",
        "success_criteria": "",
        "must_rules": "",
        "never_rules": "",
        "target_profile": "",
    }
    st.session_state.metadata = {
        "tag": "",
        "desc": "",
        "info": "",
        "name": "",
        "message": WELCOME_SVG,
        "language": "DE",
    }
    st.session_state.rag_uploads: List[Dict[str, Any]] = []
    st.session_state.rag_links: List[str] = []
    st.session_state.prompts = {
        "main": "",
        "tester": "",
        "formative": "",
        "summative": "",
    }
    st.session_state.test_results = []
    st.session_state.last_saved_json: Optional[Dict[str, Any]] = None
    st.session_state.scenario_json: Dict[str, Any] = {}
    st.session_state.assistant_ids = {
        "main": "",
        "formative": "",
        "summative": "",
    }
    st.session_state.selected_group = None
    st.session_state.selected_scenario_id = None


def go_next() -> None:
    st.session_state.current_page = min(st.session_state.current_page + 1, len(WIZARD_PAGES) - 1)


def go_back() -> None:
    st.session_state.current_page = max(st.session_state.current_page - 1, 0)


def render_nav_buttons() -> None:
    cols = st.columns([1, 1, 6])
    with cols[0]:
        if st.button("Zurück", disabled=st.session_state.current_page == 0):
            go_back()
    with cols[1]:
        if st.button("Weiter", disabled=st.session_state.current_page == len(WIZARD_PAGES) - 1):
            go_next()


def render_summary_sidebar() -> None:
    with st.sidebar:
        st.header("Wizard Status")
        st.checkbox("Dry-Run Modus", key="dry_run")
        st.markdown(f"**Server:** {st.session_state.server_key}")
        st.markdown(f"**Modus:** {st.session_state.mode}")
        meta = st.session_state.metadata
        st.markdown("### Metadaten")
        st.write({k: v for k, v in meta.items() if k != "message"})
        st.markdown("### Didaktik")
        st.write(st.session_state.didactics)
        st.markdown("### Prompt-Längen")
        for key, prompt in st.session_state.prompts.items():
            if not prompt:
                continue
            st.write(f"{key}: {len(prompt)} Zeichen")


# Page implementations -----------------------------------------------------

def page_system_mode() -> None:
    st.title("System & Modus")
    st.info("Wähle Server und Arbeitsmodus. Bei Modus C wird die bestehende Version geladen.")
    servers = load_servers()
    server_options = list(servers.keys())
    selected = st.selectbox("Server", server_options, index=server_options.index(st.session_state.server_key))
    st.session_state.server_key = selected
    mode = st.radio(
        "Modus",
        options=["A", "B", "C"],
        format_func=lambda x: {
            "A": "Neues Szenario + neue Gruppe",
            "B": "Neues Szenario in bestehender Gruppe",
            "C": "Bestehendes Szenario bearbeiten",
        }[x],
        index=["A", "B", "C"].index(st.session_state.mode),
    )
    st.session_state.mode = mode
    if mode == "C":
        st.warning("Daten werden nur geladen, wenn nicht im Dry-Run. Verbinde dich bei Bedarf über SSH.")
        if st.session_state.dry_run:
            st.info("Dry-Run aktiv – es werden keine Daten aus MySQL geladen.")
        else:
            st.write("Lade Gruppen und Szenarien ...")
            try:
                from services.db_mysql import MySQLClient, MySQLConfig
            except RuntimeError as exc:
                st.error(str(exc))
                return
            mysql_config = MySQLConfig(
                host=os.getenv("MYSQL_HOST", "127.0.0.1"),
                port=int(os.getenv("MYSQL_PORT", "3306")),
                user=os.getenv("MYSQL_USER", "root"),
                password=os.getenv("MYSQL_PASS", ""),
                database=servers[selected]["mysql_db"],
            )
            client = MySQLClient(mysql_config)
            groups = client.get_groups(selected)
            if not groups:
                st.info("Keine Gruppen gefunden.")
                return
            group_names = {g["name"]: g for g in groups}
            selected_group_name = st.selectbox("Gruppe", list(group_names.keys()))
            group = group_names[selected_group_name]
            st.session_state.selected_group = group
            scenarios = client.get_scenarios(group["id"])
            if not scenarios:
                st.info("Keine Szenarien in dieser Gruppe.")
                return
            scenario_map = {f"{item['tag']} (v{item['version']})": item for item in scenarios}
            selected_scenario = st.selectbox("Szenario", list(scenario_map.keys()))
            scenario = scenario_map[selected_scenario]
            st.session_state.selected_scenario_id = scenario["id"]
            payload = client.load_scenario_json(scenario["id"])
            if payload:
                st.session_state.metadata.update({
                    "tag": payload.get("tag", ""),
                    "desc": payload.get("desc", ""),
                    "info": payload.get("info", ""),
                    "name": payload.get("name", ""),
                    "message": payload.get("message", WELCOME_SVG),
                })
                st.session_state.assistant_ids.update({
                    "main": payload.get("assistant_id", ""),
                    "summative": payload.get("sumfeedback_id", ""),
                    "formative": payload.get("formfeedback_id", ""),
                })
                st.session_state.last_saved_json = payload
                st.success("Szenario geladen. Bitte prüfe die weiteren Seiten.")


def page_didactic_goals() -> None:
    st.title("Didaktische Ziele & Kriterien")
    state = st.session_state.didactics
    state["didactic_goal"] = st.text_area("Didaktisches Ziel", value=state["didactic_goal"], height=120)
    state["success_criteria"] = st.text_area("Erfolgskriterien / Rubrik (mit Gewichten)", value=state["success_criteria"], height=160)
    cols = st.columns(2)
    with cols[0]:
        state["must_rules"] = st.text_area("Muss-Vorgaben", value=state["must_rules"], height=120)
    with cols[1]:
        state["never_rules"] = st.text_area("Tabus", value=state["never_rules"], height=120)
    state["target_profile"] = st.text_input("Zielgruppe / Profil", value=state["target_profile"])
    st.session_state.didactics = state
    st.caption("Diese Angaben werden automatisch in die Prompt-Templates eingebettet.")


def page_metadata() -> None:
    st.title("Szenario-Metadaten")
    meta = st.session_state.metadata
    meta["tag"] = st.text_input("Tag", value=meta["tag"])
    meta["name"] = st.text_input("Name (4–7 Wörter)", value=meta["name"])
    meta["desc"] = st.text_area("Kurzbeschreibung", value=meta["desc"], height=100)
    meta["info"] = st.text_area("Info (2–4 Sätze)", value=meta["info"], height=120)
    meta["language"] = st.selectbox("Sprache", ["DE", "EN", "IT"], index=["DE", "EN", "IT"].index(meta.get("language", "DE")))
    meta["message"] = st.text_area("Welcome Message (HTML)", value=meta["message"], height=200)
    st.session_state.metadata = meta
    st.markdown("### Live-Vorschau")
    st.markdown(f"**{meta['name'] or 'Titel folgt'}**")
    st.markdown(meta["message"], unsafe_allow_html=True)


def page_knowledge() -> None:
    st.title("Wissensbasis (RAG)")
    st.caption("Lade Dokumente hoch, tagge sie und verknüpfe sie mit dem Szenario.")
    uploaded_files = st.file_uploader("Dokumente hochladen", type=["pdf", "txt", "md", "docx"], accept_multiple_files=True)
    doc_type = st.selectbox(
        "Dokumenttyp",
        ["template", "guide", "law/policy", "rubric", "persona-briefing", "background"],
        index=0,
    )
    if uploaded_files:
        for uploaded in uploaded_files:
            content = uploaded.read().decode("utf-8", errors="ignore")
            entry = {
                "title": uploaded.name,
                "doc_type": doc_type,
                "content": content,
                "attached": False,
            }
            st.session_state.rag_uploads.append(entry)
        st.success(f"{len(uploaded_files)} Dokument(e) hinzugefügt.")
    st.markdown("### Bibliothek")
    if st.session_state.rag_uploads:
        for idx, doc in enumerate(st.session_state.rag_uploads):
            cols = st.columns([4, 2, 1])
            with cols[0]:
                st.write(f"**{doc['title']}** ({doc['doc_type']})")
            with cols[1]:
                st.write(f"Länge: {len(doc['content'])} Zeichen")
            with cols[2]:
                toggled = st.checkbox("Link", value=doc.get("attached", False), key=f"rag_link_{idx}")
                doc["attached"] = toggled
                if toggled and doc["title"] not in st.session_state.rag_links:
                    st.session_state.rag_links.append(doc["title"])
                if not toggled and doc["title"] in st.session_state.rag_links:
                    st.session_state.rag_links.remove(doc["title"])
    else:
        st.info("Noch keine Dokumente hochgeladen.")


def _rag_summary() -> str:
    docs = [doc for doc in st.session_state.rag_uploads if doc.get("attached")]
    if not docs:
        return "Keine zusätzlichen Dokumente verknüpft."
    return "\n".join(f"- {doc['title']}: {doc['doc_type']}" for doc in docs)


def _ensure_prompt(key: str, template: str, **context: Any) -> None:
    if not st.session_state.prompts[key]:
        st.session_state.prompts[key] = template.format(**context)


def page_main_prompt() -> None:
    st.title("Hauptdialog Prompt")
    didactic = st.session_state.didactics
    context = {
        "didactic_goal": didactic["didactic_goal"],
        "must_rules": didactic["must_rules"],
        "never_rules": didactic["never_rules"],
        "rubric": didactic["success_criteria"],
        "rag_docs": _rag_summary(),
    }
    _ensure_prompt("main", PROMPT_TEMPLATE_MAIN, **context)
    prompt_text = st.text_area("System Prompt", value=st.session_state.prompts["main"], height=420)
    st.session_state.prompts["main"] = prompt_text
    lint = prompt_lint(prompt_text)
    st.markdown("### Prompt Lint")
    if lint.errors:
        for issue in lint.errors:
            st.error(issue.message)
    if lint.warnings:
        for issue in lint.warnings:
            st.warning(issue.message)


def page_tester_prompt() -> None:
    st.title("Test-Assessor Prompt")
    _ensure_prompt("tester", PROMPT_TEMPLATE_TEST, persona_hint="best_case")
    prompt_text = st.text_area("Persona Prompt", value=st.session_state.prompts["tester"], height=320)
    st.session_state.prompts["tester"] = prompt_text


def page_formative_prompt() -> None:
    st.title("Formatives Feedback")
    _ensure_prompt("formative", PROMPT_TEMPLATE_FORMATIVES)
    prompt_text = st.text_area("Coach Prompt", value=st.session_state.prompts["formative"], height=240)
    st.session_state.prompts["formative"] = prompt_text


def page_summative_prompt() -> None:
    st.title("Summatives Feedback")
    _ensure_prompt("summative", PROMPT_TEMPLATE_SUMMATIVES)
    prompt_text = st.text_area("Gutachter Prompt", value=st.session_state.prompts["summative"], height=360)
    st.session_state.prompts["summative"] = prompt_text


def page_automated_tests() -> None:
    st.title("Automatisierte Testläufe")
    st.write("Simuliere fünf Standard-Personas gegen den Hauptprompt. Ergebnisse basieren auf dem Summativ-Prompt.")
    persona_map = {persona.key: PROMPT_TEMPLATE_TEST.format(persona_hint=persona.key) for persona in PERSONAS}
    rubric_lines = [line.strip() for line in st.session_state.didactics["success_criteria"].splitlines() if line.strip()]
    rubric_weights = {}
    for line in rubric_lines:
        parts = line.split(":")
        if len(parts) == 2:
            rubric_weights[parts[0].strip()] = int("".join(filter(str.isdigit, parts[1]))) if any(ch.isdigit() for ch in parts[1]) else 20
    if "gesamt" not in rubric_weights and rubric_weights:
        rubric_weights["gesamt"] = sum(rubric_weights.values()) // len(rubric_weights)
    if st.button("Tests ausführen"):
        with st.spinner("Simulationen laufen ..."):
            results = batch_runner(st.session_state.prompts["main"], persona_map, rubric_weights or {"struktur_klarheit": 20})
            st.session_state.test_results = results
    if st.session_state.test_results:
        for result in st.session_state.test_results:
            color = "green" if result.passed else "red"
            st.markdown(f"#### Persona: {result.persona}")
            st.markdown(f"**Gesamt:** <span style='color:{color}'>{result.scores.get('gesamt', 'n/a')}</span>", unsafe_allow_html=True)
            st.write("Highlights:", result.highlights)
            st.write("Verbesserungen:", result.improvements)
            st.write("Notizen:", result.notes)


def build_final_json() -> Dict[str, Any]:
    meta = st.session_state.metadata
    payload = {
        "tag": meta["tag"],
        "desc": meta["desc"],
        "info": meta["info"],
        "name": meta["name"],
        "message": meta["message"],
        "assistant_id": st.session_state.assistant_ids["main"],
        "sumfeedback_id": st.session_state.assistant_ids["summative"],
        "formfeedback_id": st.session_state.assistant_ids["formative"],
    }
    st.session_state.scenario_json = payload
    return payload


def page_json_preview() -> None:
    st.title("JSON Preview & Validierung")
    payload = build_final_json()
    st.json(payload)
    validation = validate_scenario_json(payload, SCHEMA_PATH)
    if validation.is_valid:
        st.success("Schema-Validierung erfolgreich.")
    else:
        st.error("Schema-Validierung fehlgeschlagen")
        for error in validation.errors:
            st.error(error)
    if st.session_state.last_saved_json:
        st.markdown("### Unterschiede zur letzten Version")
        differ = JsonDiffer(st.session_state.last_saved_json, payload)
        st.text(differ.render())
    st.markdown("### Welcome-Message Vorschau")
    st.markdown(payload["message"], unsafe_allow_html=True)


def page_deployment() -> None:
    st.title("Deployment")
    payload = st.session_state.scenario_json or build_final_json()
    st.markdown("### Dry-Run Zusammenfassung")
    st.write({
        "assistenten": st.session_state.assistant_ids,
        "mysql_server": st.session_state.server_key,
        "rag_links": st.session_state.rag_links,
    })
    if st.session_state.dry_run:
        st.info("Dry-Run aktiv. Aktionen werden nicht gegen Live-Systeme ausgeführt.")
    else:
        st.warning("Aktionen sind live. Stelle sicher, dass API-Keys gesetzt sind.")
    col1, col2 = st.columns(2)
    with col1:
        if st.button("Assistants erstellen/aktualisieren"):
            if st.session_state.dry_run:
                st.success("Dry-Run: Assistants nicht erstellt.")
            else:
                name = payload["name"]
                main_id = create_or_update_assistant("main", name, st.session_state.prompts["main"])
                form_id = create_or_update_assistant("formative", f"{name} – Formativ", st.session_state.prompts["formative"])
                sum_id = create_or_update_assistant("summative", f"{name} – Summativ", st.session_state.prompts["summative"])
                st.session_state.assistant_ids.update({"main": main_id, "formative": form_id, "summative": sum_id})
                st.success("Assistants aktualisiert.")
    with col2:
        if st.button("Szenario nach MySQL schreiben"):
            if st.session_state.dry_run:
                st.success("Dry-Run: Szenario nicht gespeichert.")
            else:
                try:
                    from services.db_mysql import MySQLClient, MySQLConfig
                except RuntimeError as exc:
                    st.error(str(exc))
                    return
                servers = load_servers()
                mysql_config = MySQLConfig(
                    host=os.getenv("MYSQL_HOST", "127.0.0.1"),
                    port=int(os.getenv("MYSQL_PORT", "3306")),
                    user=os.getenv("MYSQL_USER", "root"),
                    password=os.getenv("MYSQL_PASS", ""),
                    database=servers[st.session_state.server_key]["mysql_db"],
                )
                client = MySQLClient(mysql_config)
                json_text = json.dumps(payload, ensure_ascii=False)
                group_id = st.session_state.selected_group["id"] if st.session_state.selected_group else 0
                client.save_scenario_json(group_id, payload["tag"], json_text, servers[st.session_state.server_key]["owner"])
                st.success("Szenario gespeichert.")


WIZARD_PAGES = [
    ("System & Modus", page_system_mode),
    ("Didaktik", page_didactic_goals),
    ("Metadaten", page_metadata),
    ("Wissensbasis", page_knowledge),
    ("Dialog Prompt", page_main_prompt),
    ("Tester Prompt", page_tester_prompt),
    ("Formatives Feedback", page_formative_prompt),
    ("Summatives Feedback", page_summative_prompt),
    ("Tests", page_automated_tests),
    ("JSON & Diff", page_json_preview),
    ("Deployment", page_deployment),
]


def main() -> None:
    init_state()
    render_summary_sidebar()
    title, page_fn = WIZARD_PAGES[st.session_state.current_page]
    page_fn()
    render_nav_buttons()


if __name__ == "__main__":
    main()
