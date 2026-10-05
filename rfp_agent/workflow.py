"""Fachada: run(requirement_id) -> {answer, citations, trace, status, sql_used}."""
from __future__ import annotations

import json

from rfp_agent.config import MAX_PASOS
from rfp_agent.db import SCHEMA_RUNS, connect_runs, ensure_loaded, get_item
from rfp_agent.graph import compile_graph
from rfp_agent.llm import backend_usado
from rfp_agent.mcp import ClienteMCP, construir_servidor


class RfpWorkflow:
    def __init__(self):
        ensure_loaded()
        self.servidor = construir_servidor()
        self.cliente = ClienteMCP(self.servidor)
        self.cliente.descubrir()
        self.app, self.recursion_limit = compile_graph(self.cliente)

    def run(self, requirement_id: str) -> dict:
        item = get_item(requirement_id)
        if not item:
            raise ValueError(f"requisito desconocido: {requirement_id}")
        inicial = {
            "requirement_id": requirement_id,
            "requirement_text": item["texto"],
            "plan": [],
            "sql_checks": [],
            "ruta": "",
            "evidence": [],
            "draft": "",
            "citations": [],
            "verdict": "",
            "critique": "",
            "sql_used": [],
            "answer": "",
            "status": "",
            "mensajes": [],
            "pasos": 0,
            "replan": 0,
            "rol": "planner",
            "tokens_usados": 0,
            "tool_rounds": 0,
            "tool_sigs": [],
            "_traza": [],
        }
        salida = self.app.invoke(
            inicial,
            {"recursion_limit": self.recursion_limit},
        )
        paquete = {
            "requirement_id": requirement_id,
            "answer": salida.get("answer") or salida.get("draft") or "",
            "citations": salida.get("citations") or [],
            "trace": salida.get("_traza") or [],
            "status": salida.get("status") or "completed",
            "sql_used": salida.get("sql_used") or [],
            "evidence": salida.get("evidence") or [],
            "verdict": salida.get("verdict"),
            "pasos": salida.get("pasos"),
            "llm": backend_usado(),
            "max_pasos": MAX_PASOS,
        }
        _persistir(paquete)
        return paquete


def _persistir(paquete: dict) -> None:
    conn = connect_runs()
    try:
        conn.executescript(SCHEMA_RUNS)
        conn.execute(
            "INSERT INTO run_trace (requirement_id, status, answer, payload) VALUES (?, ?, ?, ?)",
            (
                paquete["requirement_id"],
                paquete["status"],
                paquete["answer"],
                json.dumps(
                    {k: paquete[k] for k in ("status", "pasos", "llm", "citations") if k in paquete},
                    ensure_ascii=False,
                    default=str,
                ),
            ),
        )
        conn.commit()
    finally:
        conn.close()
