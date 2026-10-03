"""Servidor MCP simulado (sesión 13): tools/list + tools/call. El grafo no hardcodea N=4."""
from __future__ import annotations

import json
import re
import sqlite3
from typing import Any, Callable

from rfp_agent.config import CODE_ROOTS, DEMO_ATS, QUERY_FACTS_ROW_LIMIT
from rfp_agent.db import SCHEMA_RUNS, connect_rfp, connect_runs, ensure_loaded, get_item
from rfp_agent.retriever import qdrant_status, retrieve_knowledge

_WRITE = re.compile(
    r"\b(insert|update|delete|drop|alter|create|attach|detach|pragma|replace|vacuum)\b",
    re.I,
)


class ServidorMCP:
    def __init__(self, nombre: str):
        self.nombre = nombre
        self._impl: dict[str, Callable[..., Any]] = {}
        self._catalogo: list[dict] = []

    def publicar(self, name: str, description: str, input_schema: dict, fn: Callable[..., Any]) -> None:
        self._catalogo.append(
            {"name": name, "description": description, "inputSchema": input_schema}
        )
        self._impl[name] = fn

    def manejar(self, metodo: str, params: dict | None = None) -> dict:
        params = params or {}
        if metodo == "tools/list":
            return {"tools": list(self._catalogo)}
        if metodo == "tools/call":
            nombre = params.get("name")
            if nombre not in self._impl:
                return {"isError": True, "content": f"tool desconocida: {nombre}"}
            try:
                args = params.get("arguments") or {}
                if isinstance(args, str):
                    args = json.loads(args or "{}")
                return {"isError": False, "content": self._impl[nombre](**args)}
            except TypeError as e:
                return {"isError": True, "content": f"argumentos inválidos: {e}"}
            except Exception as e:
                return {"isError": True, "content": f"{type(e).__name__}: {e}"}
        return {"isError": True, "content": f"método no soportado: {metodo}"}


class ClienteMCP:
    def __init__(self, servidor: ServidorMCP):
        self.servidor = servidor
        self.catalogo: list[dict] = []

    def descubrir(self) -> list[str]:
        self.catalogo = self.servidor.manejar("tools/list")["tools"]
        return [t["name"] for t in self.catalogo]

    def esquemas_para_openai(self, names: list[str] | None = None) -> list[dict]:
        """Adaptador MCP inputSchema → tools OpenAI. Vive en el cliente, no en el agente."""
        if not self.catalogo:
            self.descubrir()
        tools = self.catalogo
        if names is not None:
            allowed = set(names)
            tools = [t for t in tools if t["name"] in allowed]
        return [
            {
                "type": "function",
                "function": {
                    "name": t["name"],
                    "description": t["description"],
                    "parameters": t["inputSchema"],
                },
            }
            for t in tools
        ]

    def invocar(self, nombre: str, argumentos: dict | None = None) -> Any:
        r = self.servidor.manejar(
            "tools/call", {"name": nombre, "arguments": argumentos or {}}
        )
        return {"error": r["content"]} if r["isError"] else r["content"]


def _get_requirement(requirement_id: str) -> dict:
    item = get_item(requirement_id)
    if not item:
        raise ValueError(f"requisito desconocido: {requirement_id}")
    conn = connect_rfp()
    try:
        g = conn.execute("SELECT * FROM golden WHERE id = ?", (requirement_id,)).fetchone()
    finally:
        conn.close()
    return {
        "id": item["id"],
        "texto": item["texto"],
        "excel_row": item["excel_row"],
        "in_golden": bool(item["in_golden"]),
        "sql_verificacion": dict(g)["sql_verificacion"] if g else None,
        "respondible": int(g["respondible"]) if g else None,
    }


def _list_sources() -> dict:
    qd = qdrant_status()
    roots = []
    for name in CODE_ROOTS:
        p = DEMO_ATS / name
        roots.append({"name": name, "path": str(p), "exists": p.exists()})
    return {
        "qdrant": qd,
        "demo_ats": {"path": str(DEMO_ATS), "exists": DEMO_ATS.exists(), "code_roots": roots},
        "sqlite_facts": True,
        "stub_docs": True,
    }


def _query_facts(sql: str) -> dict:
    ensure_loaded()
    texto = (sql or "").strip().rstrip(";")
    if not texto:
        raise ValueError("sql vacío")
    if ";" in texto:
        raise ValueError("una sola sentencia")
    if not re.match(r"^select\b", texto, re.I):
        raise ValueError("solo SELECT")
    if _WRITE.search(texto):
        raise ValueError("escritura prohibida")
    conn = connect_rfp()
    try:
        cur = conn.execute(texto)
        cols = [d[0] for d in cur.description] if cur.description else []
        rows = [dict(zip(cols, r)) for r in cur.fetchmany(QUERY_FACTS_ROW_LIMIT)]
        return {"columns": cols, "rows": rows, "truncated": cur.fetchone() is not None}
    except sqlite3.Error as e:
        raise ValueError(f"sql inválido: {e}") from e
    finally:
        conn.close()


def _record_evidence(requirement_id: str, chunk_key: str, cita: str) -> dict:
    ensure_loaded()
    conn = connect_runs()
    try:
        conn.executescript(SCHEMA_RUNS)
        conn.execute(
            "INSERT OR IGNORE INTO evidence_log (requirement_id, chunk_key, cita) VALUES (?, ?, ?)",
            (requirement_id, chunk_key, cita),
        )
        conn.commit()
        n = conn.execute(
            "SELECT COUNT(*) FROM evidence_log WHERE requirement_id = ?",
            (requirement_id,),
        ).fetchone()[0]
        return {"ok": True, "requirement_id": requirement_id, "stored": n}
    finally:
        conn.close()


def _retrieve(query: str, scope: str = "both", k: int = 5) -> dict:
    hits = retrieve_knowledge(query=query, scope=scope, k=k)
    return {"hits": hits, "k": len(hits)}


def construir_servidor() -> ServidorMCP:
    s = ServidorMCP("rfp-taller3")
    s.publicar(
        "retrieve_knowledge",
        "Única tool RAG/CAG. Recupera fragmentos de documentos y/o código. "
        "Parámetros: query, scope (docs|code|both), k. Cada hit trae kb_status.",
        {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Consulta de recuperación"},
                "scope": {
                    "type": "string",
                    "enum": ["docs", "code", "both"],
                    "description": "docs, code o both",
                },
                "k": {"type": "integer", "description": "Número de fragmentos"},
            },
            "required": ["query"],
        },
        _retrieve,
    )
    s.publicar(
        "get_requirement",
        "Lee un ítem de las 174 filas RFP (id y texto). No exige que esté en el golden.",
        {
            "type": "object",
            "properties": {
                "requirement_id": {"type": "string", "description": "Id tipo R-001"},
            },
            "required": ["requirement_id"],
        },
        _get_requirement,
    )
    s.publicar(
        "query_facts",
        "SELECT de solo lectura sobre SQLite de hechos/golden. El límite de filas lo aplica el servidor.",
        {
            "type": "object",
            "properties": {
                "sql": {"type": "string", "description": "Una sentencia SELECT"},
            },
            "required": ["sql"],
        },
        _query_facts,
    )
    s.publicar(
        "record_evidence",
        "Ancla evidencia usada (requirement_id, chunk_key, cita) en SQLite de corrida. Idempotente.",
        {
            "type": "object",
            "properties": {
                "requirement_id": {"type": "string"},
                "chunk_key": {"type": "string"},
                "cita": {"type": "string"},
            },
            "required": ["requirement_id", "chunk_key", "cita"],
        },
        _record_evidence,
    )
    s.publicar(
        "list_sources",
        "Lista fuentes disponibles (Qdrant, stub, demo_ats/CODE_ROOTS). No recupera texto.",
        {"type": "object", "properties": {}},
        _list_sources,
    )
    return s
