from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass
from typing import List

from services.db_mysql import MySQLClient, MySQLConfig
from services.db_pgvector import Embeddings, PgClient, PgConfig


@dataclass
class SeedDocument:
    title: str
    doc_type: str
    content: str


SCENARIO_PAYLOAD = {
    "tag": "bewerbung_junior_data_analyst",
    "desc": "Simulation eines strukturierten Bewerbungsgesprächs (Einstiegsrolle).",
    "info": "Sie trainieren prägnante Antworten, STAR-Beispiele und professionelles Auftreten. Nach dem Gespräch erhalten Sie formatives und summatives Feedback.",
    "name": "Bewerbungsgespräch – Junior Data Analyst",
    "message": (
        "Herzlich willkommen! In diesem Szenario simulieren Sie ein Bewerbungsgespräch für die Rolle <b>Junior Data Analyst</b>. "
        "Bitte starten Sie mit einer kurzen Selbstvorstellung und klicken Sie zum Sprechen auf den Mikrofon-Button."
        "<svg xmlns=\"http://www.w3.org/2000/svg\" width=\"20\" height=\"20\" viewBox=\"0 0 200 200\" style=\"vertical-align: middle; margin-right: 4px; display: inline;\">"
        "  <rect width=\"200\" height=\"200\" rx=\"24\" fill=\"#00a2ed\"/>"
        "  <path d=\"M 0 90 Q 40 90, 70 60 L 140 0 L 160 0 L 90 70 Q 60 100, 90 130 L 160 200 L 140 200 L 70 140 Q 30 100, 0 100 Z\" fill=\"#e5e5e5\"/>"
        "</svg>"
        "<b>Interview-Modus</b>: Der Avatar stellt Fragen, Sie antworten. "
        "Nach dem Gespräch erhalten Sie auf der Auswertungsseite detailliertes Feedback."
    ),
    "assistant_id": "",
    "sumfeedback_id": "",
    "formfeedback_id": "",
}

DOCUMENTS: List[SeedDocument] = [
    SeedDocument(
        title="Interview-Leitfaden Junior Data Analyst",
        doc_type="template",
        content="""Leitfaden für Interviewer:innen mit Fragen zu Motivation, STAR-Beispielen und analytischen Grundlagen.""",
    ),
    SeedDocument(
        title="Interview-Rubrik STAR-Bewertung",
        doc_type="rubric",
        content="""Bewertungsskala Struktur & Klarheit, Passung & Beispiele, Fachliche Korrektheit, Kommunikation, Reflexion.""",
    ),
]


def mysql_client() -> MySQLClient:
    return MySQLClient(
        MySQLConfig(
            host=os.getenv("MYSQL_HOST", "127.0.0.1"),
            port=int(os.getenv("MYSQL_PORT", "3306")),
            user=os.getenv("MYSQL_USER", "root"),
            password=os.getenv("MYSQL_PASS", ""),
            database=os.getenv("MYSQL_DB", "trainexus"),
        )
    )


def pg_client() -> PgClient:
    return PgClient(
        PgConfig(
            host=os.getenv("PG_HOST", "127.0.0.1"),
            port=int(os.getenv("PG_PORT", "5432")),
            database=os.getenv("PG_DB", "trainexus_rag"),
            user=os.getenv("PG_USER", "rag_user"),
            password=os.getenv("PG_PASS", ""),
        )
    )


def run(args: argparse.Namespace) -> None:
    if args.mysql:
        if args.dry_run:
            print("[Dry-Run] Würde Szenario nach MySQL schreiben:")
            print(json.dumps(SCENARIO_PAYLOAD, indent=2, ensure_ascii=False))
        else:
            client = mysql_client()
            json_text = json.dumps(SCENARIO_PAYLOAD, ensure_ascii=False)
            client.save_scenario_json(args.group_id, SCENARIO_PAYLOAD["tag"], json_text, os.getenv("SEED_OWNER", "seed"))
            print("MySQL-Szenario gespeichert.")
    if args.postgres:
        client = None
        if not args.dry_run:
            client = pg_client()
            embedding_model = os.getenv("EMBEDDING_MODEL", "text-embedding-3-large")
            embedder = Embeddings(embedding_model)
        for doc in DOCUMENTS:
            if args.dry_run:
                print(f"[Dry-Run] Dokument: {doc.title} ({doc.doc_type})")
                continue
            doc_id = client.upsert_document(
                title=doc.title,
                doc_type=doc.doc_type,
                server=os.getenv("SEED_SERVER", "DEV"),
                owner=os.getenv("SEED_OWNER", "seed"),
                content=doc.content,
            )
            embedding = embedder.create_embedding(doc.content)
            client.upsert_embedding(doc_id, embedding)
            client.link_to_scenario(SCENARIO_PAYLOAD["tag"], doc_id)
            print(f"Dokument {doc.title} gespeichert und verknüpft.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Seed Trainexus Szenario")
    parser.add_argument("--mysql", action="store_true", help="Szenario nach MySQL schreiben")
    parser.add_argument("--postgres", action="store_true", help="Dokumente nach Postgres schreiben")
    parser.add_argument("--dry-run", action="store_true", help="Keine Schreiboperationen ausführen")
    parser.add_argument("--group-id", type=int, default=0, help="Zielgruppe für MySQL-Inserts")
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
