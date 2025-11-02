# Trainexus Scenario Designer

Streamlit-Wizard zum Erstellen, Testen und Ausrollen von Dialog-Szenarien für Trainexus/ExamSim. Der Wizard führt Expert:innen durch elf Seiten – von Didaktik über Prompting, automatisierte Tests bis zur Veröffentlichung bei OpenAI und MySQL.

## Features

- Multipage-Streamlit-App mit persistentem `session_state`
- Drei Arbeitsmodi (neue Gruppe, bestehende Gruppe, bestehendes Szenario bearbeiten)
- Upload und Verwaltung von Wissensdokumenten mit pgvector-Anbindung
- Prompt-Linting und Vorlagen für Hauptdialog, Tester:in sowie formative und summative Feedbacks
- Automatisierte LLM-vs-LLM Simulationen inklusive Bewertungslogik
- JSON-Vorschau, Schema-Validierung und Diff zur letzten Version
- Deployment-Step für OpenAI Assistants und MySQL Persistenz
- CI-Dashboard für Regressionstests aller Szenarien

## Setup

### Voraussetzungen

- Python 3.11+
- Installierte Projektabhängigkeiten (siehe `requirements.txt`)
- Zugangsdaten in `.env`

### Installation

```bash
pip install -r requirements.txt
streamlit run streamlit_app.py
```

### `.env` Variablen

```
OPENAI_API_KEY=...
SSH_HOST=...
SSH_PORT=22
SSH_USER=...
SSH_KEY_PATH=...
MYSQL_HOST=127.0.0.1
MYSQL_PORT=3306
MYSQL_USER=trainexus
MYSQL_PASS=...
MYSQL_DB=trainexus
PG_HOST=127.0.0.1
PG_PORT=5432
PG_DB=trainexus_rag
PG_USER=rag_user
PG_PASS=...
EMBEDDING_MODEL=text-embedding-3-large
```

### Server-Profile

`config/servers.yaml` enthält vorkonfigurierte Profile (UBT, Trainexus, KNOLL, DEV). Erweiterungen erfolgen durch Ergänzen neuer Sektionen.

### SSH Tunnel

Der `SSHManager` in `services/db_mysql.py` nutzt Paramiko für lokale Tunnel. Im Streamlit-Wizard wird der Tunnel nicht automatisch gestartet; verwende ein externes Skript oder erweitere die App, falls nötig.

### Datenbanken

**MySQL**

```sql
CREATE TABLE groups (
  id INT AUTO_INCREMENT PRIMARY KEY,
  name VARCHAR(255) NOT NULL,
  server VARCHAR(32) NOT NULL,
  created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
  owner VARCHAR(128),
  is_active TINYINT DEFAULT 1
);

CREATE TABLE scenarios (
  id INT AUTO_INCREMENT PRIMARY KEY,
  group_id INT,
  tag VARCHAR(255) UNIQUE,
  json_text LONGTEXT,
  version INT DEFAULT 1,
  created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
  updated_at DATETIME DEFAULT CURRENT_TIMESTAMP,
  owner VARCHAR(128),
  FOREIGN KEY (group_id) REFERENCES groups(id)
);

CREATE TABLE scenario_versions (
  id INT AUTO_INCREMENT PRIMARY KEY,
  scenario_id INT,
  version INT,
  json_text LONGTEXT,
  changelog TEXT,
  created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
  FOREIGN KEY (scenario_id) REFERENCES scenarios(id)
);
```

**Postgres / pgvector**

```sql
CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE documents (
  id UUID PRIMARY KEY,
  title TEXT,
  doc_type TEXT,
  server TEXT,
  owner TEXT,
  content TEXT,
  checksum TEXT UNIQUE,
  token_count INT,
  created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE document_embeddings (
  id UUID PRIMARY KEY,
  document_id UUID REFERENCES documents(id),
  embedding VECTOR(1536),
  created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE scenario_documents (
  scenario_tag TEXT,
  document_id UUID REFERENCES documents(id),
  PRIMARY KEY (scenario_tag, document_id)
);

CREATE INDEX IF NOT EXISTS document_embeddings_ivf ON document_embeddings USING ivfflat (embedding vector_l2_ops);
CREATE INDEX IF NOT EXISTS documents_server_doc_type ON documents(server, doc_type);
```

### Seed Script

`scripts/seed.py` lädt ein Beispielszenario „Bewerbungsgespräch – Junior Data Analyst“ sowie zwei Dokumente in Postgres und verknüpft sie.

```bash
python scripts/seed.py --mysql --postgres
```

Die Skript-Parameter erlauben `--dry-run`, um Einträge nur zu visualisieren.

### Tests

```
pytest
```

## CI Dashboard

Über `streamlit run streamlit_app.py` und Seitenleiste „CI Regression Checks“ lassen sich Regressionstests starten. Ergebnisse werden als CSV nach `data/ci_reports` exportiert.

## Anpassungen

- Prompt-Templates anpassen in `streamlit_app.py`
- Neue Personas in `PERSONAS` hinzufügen
- Zusätzliche Checks im CI-Report ergänzen

Viel Erfolg beim Design deiner Szenarien!
