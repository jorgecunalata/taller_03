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
    from rfp_agent.config import CORPUS, resolve_code_root
    from rfp_agent.docs_pipeline import iter_doc_paths

    qd = qdrant_status()
    roots = []
    for name in CODE_ROOTS:
        p = resolve_code_root(name)
        roots.append(
            {
                "name": name,
                "path": str(p) if p else str(DEMO_ATS / name),
                "exists": bool(p and p.exists()),
            }
        )
    doc_paths, dirs_info = iter_doc_paths()
    return {
        "qdrant": qd,
        "demo_ats": {"path": str(DEMO_ATS), "exists": DEMO_ATS.exists(), "code_roots": roots},
        "corpus": {
            "path": str(CORPUS),
            "exists": CORPUS.exists(),
            "dirs": dirs_info,
            "files": [p.name for p in doc_paths],
            "collection": qd.get("documentos"),
        },
        "sqlite_facts": True,
        "stub_docs": not qd.get("documentos"),
    }


def reescribir_sql_facts(sql: str) -> str:
    """H200 a menudo usa requirement_id; en `facts` la columna es req_id.

    `evidence_log` (runs.sqlite) sí usa requirement_id, pero query_facts solo lee rfp.sqlite.
    """
    texto = (sql or "").strip()
    # Si menciona evidence_log, no reescribir (tabla no está aquí de todos modos).
    if re.search(r"\bevidence_log\b", texto, re.I):
        return texto
    # facts / golden / rfp_items: requirement_id → req_id (salvo rfp_items.id).
    if re.search(r"\bfacts\b", texto, re.I) or re.search(r"\brequirement_id\b", texto, re.I):
        texto = re.sub(r"\brequirement_id\b", "req_id", texto, flags=re.I)
    return texto


def _query_facts(sql: str) -> dict:
    ensure_loaded()
    texto = (sql or "").strip().rstrip(";")
    if not texto:
        return {"error": "sql vacío", "columns": [], "rows": []}
    if ";" in texto:
        return {"error": "una sola sentencia", "columns": [], "rows": []}
    if not re.match(r"^select\b", texto, re.I):
        return {"error": "solo SELECT", "columns": [], "rows": []}
    if _WRITE.search(texto):
        return {"error": "escritura prohibida", "columns": [], "rows": []}
    if re.search(r"\bevidence_log\b", texto, re.I):
        return {
            "error": (
                "evidence_log vive en runs.sqlite; usa record_evidence. "
                "Para hechos: SELECT … FROM facts WHERE req_id = 'R-001'"
            ),
            "columns": [],
            "rows": [],
        }
    texto = reescribir_sql_facts(texto)
    conn = connect_rfp()
    try:
        cur = conn.execute(texto)
        cols = [d[0] for d in cur.description] if cur.description else []
        rows = [dict(zip(cols, r)) for r in cur.fetchmany(QUERY_FACTS_ROW_LIMIT)]
        return {"columns": cols, "rows": rows, "truncated": cur.fetchone() is not None}
    except sqlite3.Error as e:
        # No raise: el harness devuelve error estructurado; no ensuciar el draft.
        hint = ""
        if "no such column" in str(e).lower() and "requirement" in str(e).lower():
            hint = " (en facts la columna es req_id, no requirement_id)"
        return {
            "error": f"sql inválido: {e}{hint}",
            "columns": [],
            "rows": [],
            "hint": "SELECT statement, source_doc, locator FROM facts WHERE req_id = 'R-001'",
        }
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
        return {"ok": True, "requirement_id": requirement_id, "chunk_key": chunk_key, "stored": n}
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
        "SELECT de solo lectura sobre SQLite de hechos/golden (rfp.sqlite). "
        "En la tabla facts la columna es req_id (NO requirement_id). "
        "Ejemplo: SELECT statement, source_doc, locator FROM facts WHERE req_id = 'R-001'. "
        "El límite de filas lo aplica el servidor.",
        {
            "type": "object",
            "properties": {
                "sql": {
                    "type": "string",
                    "description": "Una sentencia SELECT. facts.req_id, no requirement_id.",
                },
            },
            "required": ["sql"],
        },
        _query_facts,
    )
    s.publicar(
        "record_evidence",
        "Ancla evidencia usada (requirement_id, chunk_key, cita) en runs.sqlite. Idempotente. "
        "No es SQL: pasa argumentos JSON. evidence_log.requirement_id es el id R-nnn.",
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
