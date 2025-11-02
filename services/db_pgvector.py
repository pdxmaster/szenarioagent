"""Postgres + pgvector integration."""
from __future__ import annotations

import contextlib
import hashlib
import importlib
import importlib.util
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple


@dataclass
class PgConfig:
    host: str
    port: int
    database: str
    user: str
    password: str


class PgClient:
    """Small helper client around psycopg2."""

    def __init__(self, config: PgConfig):
        self.config = config

    def _connect(self):
        if importlib.util.find_spec("psycopg2") is None:
            raise RuntimeError("psycopg2 is required for Postgres connectivity")
        psycopg2 = importlib.import_module("psycopg2")
        return psycopg2.connect(
            host=self.config.host,
            port=self.config.port,
            dbname=self.config.database,
            user=self.config.user,
            password=self.config.password,
        )

    def _execute(self, query: str, params: Optional[Sequence[Any]] = None) -> None:
        with contextlib.closing(self._connect()) as conn:
            with conn.cursor() as cursor:
                cursor.execute(query, params)
                conn.commit()

    def _fetchall(self, query: str, params: Optional[Sequence[Any]] = None) -> List[Tuple[Any, ...]]:
        with contextlib.closing(self._connect()) as conn:
            with conn.cursor() as cursor:
                cursor.execute(query, params)
                return cursor.fetchall()

    # Document management -------------------------------------------------
    def upsert_document(
        self,
        title: str,
        doc_type: str,
        server: str,
        owner: str,
        content: str,
        checksum: Optional[str] = None,
        token_count: Optional[int] = None,
    ) -> str:
        import uuid

        checksum = checksum or hashlib.sha256(content.encode("utf-8")).hexdigest()
        document_id = str(uuid.uuid4())
        query = (
            "INSERT INTO documents (id, title, doc_type, server, owner, content, checksum, token_count, created_at)"
            " VALUES (%s, %s, %s, %s, %s, %s, %s, %s, NOW())"
            " ON CONFLICT (checksum) DO UPDATE SET title=EXCLUDED.title, doc_type=EXCLUDED.doc_type, content=EXCLUDED.content,"
            " owner=EXCLUDED.owner, token_count=EXCLUDED.token_count RETURNING id"
        )
        params = (
            document_id,
            title,
            doc_type,
            server,
            owner,
            content,
            checksum,
            token_count,
        )
        rows = self._fetchall(query, params)
        return rows[0][0]

    def list_documents(self, server: Optional[str] = None) -> List[Dict[str, Any]]:
        base = "SELECT id, title, doc_type, server, owner, checksum, token_count, created_at FROM documents"
        params: Sequence[Any] = []
        if server:
            base += " WHERE server=%s"
            params = [server]
        base += " ORDER BY created_at DESC"
        rows = self._fetchall(base, params)
        return [
            {
                "id": row[0],
                "title": row[1],
                "doc_type": row[2],
                "server": row[3],
                "owner": row[4],
                "checksum": row[5],
                "token_count": row[6],
                "created_at": row[7],
            }
            for row in rows
        ]

    def link_to_scenario(self, scenario_tag: str, document_id: str) -> None:
        query = (
            "INSERT INTO scenario_documents (scenario_tag, document_id) VALUES (%s, %s)"
            " ON CONFLICT DO NOTHING"
        )
        self._execute(query, (scenario_tag, document_id))

    def unlink_from_scenario(self, scenario_tag: str, document_id: str) -> None:
        query = "DELETE FROM scenario_documents WHERE scenario_tag=%s AND document_id=%s"
        self._execute(query, (scenario_tag, document_id))

    def list_scenario_documents(self, scenario_tag: str) -> List[str]:
        rows = self._fetchall(
            "SELECT document_id FROM scenario_documents WHERE scenario_tag=%s",
            (scenario_tag,),
        )
        return [row[0] for row in rows]

    def search_by_text(self, text: str, limit: int = 10) -> List[Dict[str, Any]]:
        query = (
            "SELECT id, title, doc_type FROM documents WHERE to_tsvector('simple', content) @@ plainto_tsquery(%s)"
            " ORDER BY created_at DESC LIMIT %s"
        )
        rows = self._fetchall(query, (text, limit))
        return [
            {"id": row[0], "title": row[1], "doc_type": row[2]}
            for row in rows
        ]

    def upsert_embedding(self, document_id: str, embedding: Sequence[float]) -> None:
        query = (
            "INSERT INTO document_embeddings (id, document_id, embedding, created_at)"
            " VALUES (%s, %s, %s, NOW()) ON CONFLICT (document_id) DO UPDATE SET embedding=EXCLUDED.embedding"
        )
        import uuid

        embedding_vector = list(embedding)
        params = (str(uuid.uuid4()), document_id, embedding_vector)
        self._execute(query, params)


class Embeddings:
    def __init__(self, model: str):
        self.model = model

    def create_embedding(self, text: str) -> List[float]:
        if importlib.util.find_spec("openai") is None:
            raise RuntimeError("openai package is required for embeddings")
        openai_module = importlib.import_module("openai")
        OpenAI = getattr(openai_module, "OpenAI")
        client = OpenAI()
        response = client.embeddings.create(model=self.model, input=text)
        return list(response.data[0].embedding)


__all__ = ["PgConfig", "PgClient", "Embeddings"]
