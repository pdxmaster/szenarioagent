"""MySQL + SSH tunnel helpers for Trainexus."""
from __future__ import annotations

import contextlib
import importlib
import importlib.util
import socket
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional


def _load_paramiko():
    if importlib.util.find_spec("paramiko") is None:
        raise RuntimeError("paramiko is required for SSH tunnelling. Install with `pip install paramiko`.")
    return importlib.import_module("paramiko")


@dataclass
class SSHConfig:
    host: str
    port: int
    user: str
    key_path: Optional[str] = None
    password: Optional[str] = None
    remote_host: str = "127.0.0.1"
    remote_port: int = 3306
    local_bind_port: int = 13306


class SSHManager:
    """Manage a local SSH tunnel for MySQL connections."""

    def __init__(self, config: SSHConfig):
        self.config = config
        paramiko_module = _load_paramiko()
        self._paramiko = paramiko_module
        self._client: Optional[Any] = None
        self._forwarder = None

    def start(self) -> None:
        if self._client:
            return
        client = self._paramiko.SSHClient()
        client.set_missing_host_key_policy(self._paramiko.AutoAddPolicy())
        client.connect(
            hostname=self.config.host,
            port=self.config.port,
            username=self.config.user,
            key_filename=self.config.key_path,
            password=self.config.password,
            look_for_keys=False,
        )
        transport = client.get_transport()
        if transport is None:
            raise RuntimeError("SSH transport could not be established")
        self._forwarder = self._paramiko.forward.forward_local_port(
            ("127.0.0.1", self.config.local_bind_port),
            self.config.remote_host,
            self.config.remote_port,
            transport,
        )
        self._client = client

    def stop(self) -> None:
        if self._forwarder:
            self._forwarder.close()
            self._forwarder = None
        if self._client:
            self._client.close()
            self._client = None

    def is_running(self) -> bool:
        if not self._client:
            return False
        transport = self._client.get_transport()
        return bool(transport and transport.is_active())

    def check_local_port(self) -> bool:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(0.2)
        try:
            sock.connect(("127.0.0.1", self.config.local_bind_port))
            return True
        except OSError:
            return False
        finally:
            sock.close()


@dataclass
class MySQLConfig:
    host: str
    port: int
    user: str
    password: str
    database: str
    use_ssh: bool = False


class MySQLClient:
    """Thin wrapper around PyMySQL with domain-specific helpers."""

    def __init__(self, config: MySQLConfig):
        self.config = config

    def _connect(self):
        if importlib.util.find_spec("pymysql") is None:
            raise RuntimeError("pymysql is required for MySQL connectivity")
        pymysql = importlib.import_module("pymysql")
        return pymysql.connect(
            host=self.config.host,
            port=self.config.port,
            user=self.config.user,
            password=self.config.password,
            database=self.config.database,
            cursorclass=pymysql.cursors.DictCursor,
            autocommit=True,
            charset="utf8mb4",
        )

    def _fetchall(self, query: str, params: Optional[Iterable[Any]] = None) -> List[Dict[str, Any]]:
        with contextlib.closing(self._connect()) as connection:
            with connection.cursor() as cursor:
                cursor.execute(query, params)
                return list(cursor.fetchall())

    def _execute(self, query: str, params: Optional[Iterable[Any]] = None) -> None:
        with contextlib.closing(self._connect()) as connection:
            with connection.cursor() as cursor:
                cursor.execute(query, params)

    # Public API -----------------------------------------------------------
    def get_groups(self, server: str) -> List[Dict[str, Any]]:
        sql = "SELECT id, name, server, owner FROM groups WHERE server=%s AND is_active=1 ORDER BY name"
        return self._fetchall(sql, (server,))

    def get_scenarios(self, group_id: int) -> List[Dict[str, Any]]:
        sql = "SELECT id, tag, version, updated_at FROM scenarios WHERE group_id=%s ORDER BY updated_at DESC"
        return self._fetchall(sql, (group_id,))

    def load_scenario_json(self, scenario_id: int) -> Optional[Dict[str, Any]]:
        sql = "SELECT json_text FROM scenarios WHERE id=%s"
        rows = self._fetchall(sql, (scenario_id,))
        if not rows:
            return None
        import json

        return json.loads(rows[0]["json_text"])

    def save_scenario_json(self, group_id: int, tag: str, json_text: str, owner: str) -> None:
        sql = (
            "INSERT INTO scenarios (group_id, tag, json_text, version, owner, created_at, updated_at) "
            "VALUES (%s, %s, %s, 1, %s, NOW(), NOW()) "
            "ON DUPLICATE KEY UPDATE json_text=VALUES(json_text), version=version+1, updated_at=NOW(), owner=VALUES(owner)"
        )
        self._execute(sql, (group_id, tag, json_text, owner))

    def save_version(self, scenario_id: int, json_text: str, version: int, changelog: str = "") -> None:
        sql = (
            "INSERT INTO scenario_versions (scenario_id, version, json_text, changelog, created_at) "
            "VALUES (%s, %s, %s, %s, NOW())"
        )
        self._execute(sql, (scenario_id, version, json_text, changelog))


__all__ = ["SSHConfig", "SSHManager", "MySQLConfig", "MySQLClient"]
