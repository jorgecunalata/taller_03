"""Evaluador RFP: SQL de suelo, sin LLM-as-judge. Solo ítems in_golden=1."""
from __future__ import annotations

import json
import re

from rfp_agent.config import ABSTENCION, OUT
from rfp_agent.db import connect_rfp, ensure_loaded, list_golden
from rfp_agent.textutil import normalizar, se_abstuvo
from rfp_agent.workflow import RfpWorkflow

_PROHIBIDO = ("[[rellenar]]", "ejemplo_no_validado")
_TPS = re.compile(r"\b(\d+)\s*(tps|transacciones por segundo)\b", re.I)


def _cubre(answer: str, citations: list, statement: str) -> bool:
    blob = normalizar(answer + " " + json.dumps(citations, ensure_ascii=False))
    st = normalizar(statement)
    if st and st in blob:
        return True
    tokens = [t for t in st.split() if len(t) > 4]
    if not tokens:
        return bool(st) and st in blob
    return all(t in blob for t in tokens[:4])


def _citas_pdf(citations: list, evidence: list, answer: str = "") -> bool:
    """Acepta ancla PDF en citations/evidence o [n] + fuente PDF en el answer."""
    blob = (
        json.dumps(citations, ensure_ascii=False).lower()
        + json.dumps(evidence, ensure_ascii=False).lower()
        + (answer or "").lower()
    )
    if (
        "descripcion_funcional" in blob
        or "p.11" in blob
        or "p.13" in blob
        or "p.15" in blob
    ):
        return True
    # [1]/[2] en el texto + al menos una citation/evidence con source/locator
    if re.search(r"\[\d+\]", answer or ""):
        for c in citations or []:
            src = f"{c.get('source') or ''} {c.get('locator') or ''}".lower()
            if "descripcion_funcional" in src or re.search(r"p\.\d+", src):
                return True
        for e in evidence or []:
            src = f"{e.get('source') or ''} {e.get('locator') or ''}".lower()
            if "descripcion_funcional" in src or re.search(r"p\.\d+", src):
                return True
    return False


def evaluar_paquete(item: dict, paquete: dict, facts_rows: list[dict]) -> dict:
    answer = paquete.get("answer") or ""
    citations = paquete.get("citations") or []
    evidence = paquete.get("evidence") or []
    blob = normalizar(answer + json.dumps(citations, ensure_ascii=False))
    fallos = []
    if any(p in blob for p in _PROHIBIDO):
        fallos.append("cita_plantilla")
    if "sql inválido" in answer.lower() or "valueerror" in answer.lower():
        fallos.append("ruido_tool_en_answer")
    respondible = bool(item.get("respondible"))
    if not respondible:
        if not se_abstuvo(answer, ABSTENCION):
            fallos.append("falta_abstencion")
        if _TPS.search(answer):
            fallos.append("inventa_tps")
    else:
        for row in facts_rows:
            if not _cubre(answer, citations, row.get("statement") or ""):
                fallos.append(f"falta_hecho:{row.get('statement')}")
        if not _citas_pdf(citations, evidence, answer):
            fallos.append("falta_cita_pdf")
    ok = not fallos
    return {
        "id": item["id"],
        "pass": ok,
        "fallos": fallos,
        "status": paquete.get("status"),
        "sql_rows": len(facts_rows),
        "answer": answer,
    }


def eval_rfp(ids: list[str] | None = None, workflow: RfpWorkflow | None = None) -> dict:
    ensure_loaded()
    from rfp_agent.config import FORBIDDEN_GOLDEN_IDS, GOLDEN_IDS

    golden = list_golden()
    if ids:
        wanted = {i for i in ids if i not in FORBIDDEN_GOLDEN_IDS}
        golden = [g for g in golden if g["id"] in wanted]
    else:
        golden = [g for g in golden if g["id"] in GOLDEN_IDS]
    # Nunca evaluar R-018 / fuera del slice.
    golden = [g for g in golden if g["id"] in GOLDEN_IDS and g["id"] not in FORBIDDEN_GOLDEN_IDS]
    wf = workflow or RfpWorkflow()
    conn = connect_rfp()
    resultados = []
    try:
        for item in golden:
            paquete = wf.run(item["id"])
            cur = conn.execute(item["sql_verificacion"])
            cols = [d[0] for d in cur.description] if cur.description else []
            facts_rows = [dict(zip(cols, r)) for r in cur.fetchall()]
            fila = evaluar_paquete(item, paquete, facts_rows)
            fila["trace_nodos"] = [t.get("nodo") for t in paquete.get("trace") or []]
            fila["llm"] = paquete.get("llm")
            resultados.append(fila)
    finally:
        conn.close()
    resumen = {
        "n": len(resultados),
        "passed": sum(1 for r in resultados if r["pass"]),
        "failed": sum(1 for r in resultados if not r["pass"]),
        "items": resultados,
    }
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / "eval_rfp.json"
    path.write_text(json.dumps(resumen, ensure_ascii=False, indent=2), encoding="utf-8")
    resumen["path"] = str(path)
    return resumen
