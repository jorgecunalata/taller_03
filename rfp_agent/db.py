"""Carga rfp.xlsx → rfp_items (R-001…R-174) y anota el golden de tres ítems."""
from __future__ import annotations

import sqlite3
from pathlib import Path

from openpyxl import load_workbook

from rfp_agent.config import GOLDEN_IDS, RFP_DB, RFP_XLSX, RUNS_DB

SCHEMA_RFP = """
CREATE TABLE IF NOT EXISTS rfp_items (
    id TEXT PRIMARY KEY,
    texto TEXT NOT NULL,
    excel_row INTEGER NOT NULL,
    in_golden INTEGER NOT NULL DEFAULT 0
);
CREATE TABLE IF NOT EXISTS golden (
    id TEXT PRIMARY KEY,
    respondible INTEGER NOT NULL,
    sql_verificacion TEXT NOT NULL,
    notas TEXT,
    FOREIGN KEY (id) REFERENCES rfp_items(id)
);
CREATE TABLE IF NOT EXISTS facts (
    id TEXT PRIMARY KEY,
    req_id TEXT NOT NULL,
    statement TEXT NOT NULL,
    source_doc TEXT,
    locator TEXT,
    kb_status TEXT,
    FOREIGN KEY (req_id) REFERENCES rfp_items(id)
);
"""

SCHEMA_RUNS = """
CREATE TABLE IF NOT EXISTS evidence_log (
    requirement_id TEXT NOT NULL,
    chunk_key TEXT NOT NULL,
    cita TEXT NOT NULL,
    created_at TEXT NOT NULL DEFAULT (datetime('now')),
    UNIQUE(requirement_id, chunk_key, cita)
);
CREATE TABLE IF NOT EXISTS run_trace (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    requirement_id TEXT NOT NULL,
    status TEXT,
    answer TEXT,
    payload TEXT,
    created_at TEXT NOT NULL DEFAULT (datetime('now'))
);
"""

GOLDEN_ROWS = [
    {
        "id": "R-001",
        "respondible": 1,
        "sql_verificacion": "SELECT statement, source_doc, locator FROM facts WHERE req_id = 'R-001'",
        "notas": "Frase del PDF funcional, no del xlsx.",
    },
    {
        "id": "R-038",
        "respondible": 1,
        "sql_verificacion": "SELECT statement, source_doc, locator FROM facts WHERE req_id = 'R-038'",
        "notas": "Alineado con D02 de Taller 2: 24/7 y 24/7/365.",
    },
    {
        "id": "R-057",
        "respondible": 0,
        "sql_verificacion": "SELECT statement, source_doc, locator FROM facts WHERE req_id = 'R-057'",
        "notas": "El PDF no da TPS ni un máximo numérico. Abstenerse.",
    },
]

FACTS_ROWS = [
    {
        "id": "F-001",
        "req_id": "R-001",
        "statement": "Liquidación de fondos final e irrevocable en tiempo real",
        "source_doc": "descripcion_funcional.pdf",
        "locator": "p.11",
        "kb_status": "VALIDADO",
    },
    {
        "id": "F-038a",
        "req_id": "R-038",
        "statement": "El sistema RTGS de Montran se ha mejorado con capacidades para funcionar 24/7",
        "source_doc": "descripcion_funcional.pdf",
        "locator": "p.13",
        "kb_status": "VALIDADO",
    },
    {
        "id": "F-038b",
        "req_id": "R-038",
        "statement": "El RTGS de Montran puede funcionar y procesar pagos de forma 24/7/365",
        "source_doc": "descripcion_funcional.pdf",
        "locator": "p.15",
        "kb_status": "VALIDADO",
    },
]


def connect_rfp(path: Path | None = None) -> sqlite3.Connection:
    db = path or RFP_DB
    db.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(db))
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    return conn


def connect_runs(path: Path | None = None) -> sqlite3.Connection:
    db = path or RUNS_DB
    db.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(db))
    conn.row_factory = sqlite3.Row
    return conn


def _leer_xlsx(xlsx: Path) -> list[tuple[str, int, str]]:
    wb = load_workbook(xlsx, data_only=True)
    ws = wb[wb.sheetnames[0]]
    filas = []
    for excel_row, row in enumerate(ws.iter_rows(min_row=2, values_only=True), start=2):
        texto = (row[0] or "").strip() if row else ""
        if not texto:
            continue
        rid = f"R-{len(filas) + 1:03d}"
        filas.append((rid, excel_row, texto))
    return filas


def load_rfp(xlsx: Path | None = None, db_path: Path | None = None) -> dict:
    xlsx = xlsx or RFP_XLSX
    if not xlsx.exists():
        raise FileNotFoundError(f"No está el workbook RFP: {xlsx}")
    filas = _leer_xlsx(xlsx)
    conn = connect_rfp(db_path)
    try:
        conn.executescript(SCHEMA_RFP)
        conn.execute("DELETE FROM facts")
        conn.execute("DELETE FROM golden")
        conn.execute("DELETE FROM rfp_items")
        for rid, excel_row, texto in filas:
            conn.execute(
                "INSERT INTO rfp_items (id, texto, excel_row, in_golden) VALUES (?, ?, ?, ?)",
                (rid, texto, excel_row, int(rid in GOLDEN_IDS)),
            )
        for g in GOLDEN_ROWS:
            conn.execute(
                "INSERT INTO golden (id, respondible, sql_verificacion, notas) VALUES (?, ?, ?, ?)",
                (g["id"], g["respondible"], g["sql_verificacion"], g["notas"]),
            )
        for f in FACTS_ROWS:
            conn.execute(
                "INSERT INTO facts (id, req_id, statement, source_doc, locator, kb_status) VALUES (?, ?, ?, ?, ?, ?)",
                (f["id"], f["req_id"], f["statement"], f["source_doc"], f["locator"], f["kb_status"]),
            )
        conn.commit()
    finally:
        conn.close()
    return {
        "rfp_items": len(filas),
        "golden": len(GOLDEN_ROWS),
        "facts": len(FACTS_ROWS),
        "db": str(db_path or RFP_DB),
    }


def ensure_loaded() -> None:
    conn = connect_rfp()
    try:
        n = conn.execute(
            "SELECT COUNT(*) FROM sqlite_master WHERE type='table' AND name='rfp_items'"
        ).fetchone()[0]
        if n == 0:
            load_rfp()
            return
        count = conn.execute("SELECT COUNT(*) FROM rfp_items").fetchone()[0]
        if count != 174:
            load_rfp()
    except sqlite3.Error:
        load_rfp()
    finally:
        conn.close()
    conn = connect_runs()
    try:
        conn.executescript(SCHEMA_RUNS)
        conn.commit()
    finally:
        conn.close()


def get_item(req_id: str) -> dict | None:
    ensure_loaded()
    conn = connect_rfp()
    try:
        row = conn.execute("SELECT * FROM rfp_items WHERE id = ?", (req_id,)).fetchone()
        return dict(row) if row else None
    finally:
        conn.close()


def list_golden() -> list[dict]:
    ensure_loaded()
    conn = connect_rfp()
    try:
        rows = conn.execute(
            "SELECT g.*, i.texto FROM golden g JOIN rfp_items i ON i.id = g.id ORDER BY g.id"
        ).fetchall()
        return [dict(r) for r in rows]
    finally:
        conn.close()
