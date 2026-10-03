"""Grafo LangGraph 1.2.12: Planner → Reader → Writer → Verifier ⇄ Planner → Synthesizer."""
from __future__ import annotations

import json
import operator
import time
from typing import Annotated, Any, TypedDict

from langgraph.graph import END, START, StateGraph

from rfp_agent import prompts
from rfp_agent.config import ABSTENCION, MAX_PASOS, MAX_REPLAN, TOKEN_BUDGET
from rfp_agent.llm import TOOLS_POR_ROL, get_llm
from rfp_agent.mcp import ClienteMCP


class Estado(TypedDict, total=False):
    requirement_id: str
    requirement_text: str
    plan: list
    sql_checks: list
    ruta: str
    evidence: Annotated[list, operator.add]
    draft: str
    citations: list
    verdict: str
    critique: str
    sql_used: list
    answer: str
    status: str
    mensajes: list
    pasos: int
    replan: int
    rol: str
    tokens_usados: int
    _traza: Annotated[list, operator.add]


PROMPTS = {
    "planner": prompts.PLANNER,
    "reader": prompts.READER,
    "writer": prompts.WRITER,
    "verifier": prompts.VERIFIER,
    "synthesizer": prompts.SYNTHESIZER,
}

SIGUIENTE = {
    "planner": "reader",
    "reader": "writer",
    "writer": "verifier",
    "verifier": "synthesizer",
    "synthesizer": None,
}


def _paso(nodo: str, r: dict | None = None, **extra: Any) -> dict:
    uso = (r or {}).get("uso", {})
    return {
        "nodo": nodo,
        "tokens_entrada": uso.get("prompt_tokens", 0),
        "tokens_salida": uso.get("completion_tokens", 0),
        "latencia_s": (r or {}).get("latencia_s", 0.0),
        **extra,
    }


def _parse_json(content: str) -> dict:
    texto = (content or "").strip()
    if texto.startswith("```"):
        texto = texto.strip("`")
        if texto.lower().startswith("json"):
            texto = texto[4:].strip()
    try:
        data = json.loads(texto)
        return data if isinstance(data, dict) else {"value": data}
    except json.JSONDecodeError:
        start, end = texto.find("{"), texto.rfind("}")
        if start >= 0 and end > start:
            try:
                return json.loads(texto[start : end + 1])
            except json.JSONDecodeError:
                pass
        return {"raw": texto}


def _user_de(estado: Estado) -> str:
    rid = estado.get("requirement_id", "")
    rol = estado.get("rol", "planner")
    if rol == "planner":
        partes = [
            f"requirement_id={rid}",
            f"texto: {estado.get('requirement_text', '')}",
            f"replan={estado.get('replan', 0)}",
        ]
        if estado.get("critique"):
            partes.append(f"critique_previa={estado['critique']}")
        return "\n".join(partes)
    if rol == "reader":
        return (
            f"requirement_id={rid}\ntexto: {estado.get('requirement_text','')}\n"
            f"plan={json.dumps(estado.get('plan') or [], ensure_ascii=False)}\n"
            f"ruta={estado.get('ruta') or 'docs'}"
        )
    if rol == "writer":
        return (
            f"requirement_id={rid}\ntexto: {estado.get('requirement_text','')}\n"
            f"evidence={json.dumps(estado.get('evidence') or [], ensure_ascii=False)}\n---"
        )
    if rol == "verifier":
        draft = estado.get("draft") or ""
        return (
            f"requirement_id={rid}\ntexto: {estado.get('requirement_text','')}\n"
            f"draft={draft}\n---\n"
            f"citations={json.dumps(estado.get('citations') or [], ensure_ascii=False)}"
        )
    return (
        f"requirement_id={rid}\nverdict={estado.get('verdict')}\n"
        f"draft={json.dumps({'draft': estado.get('draft'), 'citations': estado.get('citations')}, ensure_ascii=False)}\n---"
    )


def _hits_desde_mensajes(mensajes: list[dict]) -> list[dict]:
    extra = []
    for m in mensajes:
        if m.get("role") != "tool":
            continue
        try:
            body = json.loads(m.get("content") or "{}")
        except json.JSONDecodeError:
            continue
        if isinstance(body, dict) and body.get("error"):
            extra.append(
                {
                    "chunk_key": "tool-error",
                    "text": str(body["error"]),
                    "source": "mcp",
                    "locator": "",
                    "kb_status": "ERROR",
                    "score": 0.0,
                }
            )
            continue
        for h in (body.get("hits") or []) if isinstance(body, dict) else []:
            extra.append(
                {
                    "chunk_key": h.get("chunk_key"),
                    "text": h.get("text"),
                    "source": h.get("source"),
                    "locator": h.get("locator"),
                    "kb_status": h.get("kb_status"),
                    "score": h.get("score"),
                }
            )
    return extra


def construir_grafo(cliente: ClienteMCP):
    llm = get_llm()

    def nodo_agente(estado: Estado) -> dict:
        rol = estado.get("rol") or "planner"
        mensajes = estado.get("mensajes") or []
        if not mensajes or mensajes[0].get("role") != "system":
            mensajes = [
                {"role": "system", "content": PROMPTS[rol]},
                {"role": "user", "content": _user_de(estado)},
            ]
        tools = cliente.esquemas_para_openai(TOOLS_POR_ROL.get(rol) or [])
        r = llm.chat(mensajes, tools=tools or None, role=rol)
        m = r["mensaje"]
        nuevo = {
            "role": "assistant",
            "content": m.get("content") or "",
        }
        if m.get("tool_calls"):
            nuevo["tool_calls"] = m["tool_calls"]
        pedidas = [tc["function"]["name"] for tc in m.get("tool_calls") or []]
        tokens = int(estado.get("tokens_usados") or 0) + int(
            (r.get("uso") or {}).get("total_tokens") or 0
        )
        out: dict[str, Any] = {
            "mensajes": mensajes + [nuevo],
            "pasos": int(estado.get("pasos") or 0) + 1,
            "tokens_usados": tokens,
            "rol": rol,
            "_traza": [_paso(rol, r, pide=pedidas or "respuesta final")],
        }
        if not m.get("tool_calls"):
            datos = _parse_json(m.get("content") or "")
            out.update(_aplicar_salida(rol, estado, datos, mensajes + [nuevo]))
        return out

    def nodo_herramientas(estado: Estado) -> dict:
        ultimo = (estado.get("mensajes") or [{}])[-1]
        respuestas = []
        llamadas = []
        for tc in ultimo.get("tool_calls") or []:
            raw_args = tc.get("function", {}).get("arguments") or "{}"
            try:
                args = json.loads(raw_args) if isinstance(raw_args, str) else (raw_args or {})
            except json.JSONDecodeError:
                args = {}
            nombre = tc["function"]["name"]
            obs = cliente.invocar(nombre, args)
            llamadas.append((nombre, args))
            respuestas.append(
                {
                    "role": "tool",
                    "tool_call_id": tc.get("id"),
                    "content": json.dumps(obs, ensure_ascii=False, default=str),
                }
            )
        extra_ev = _hits_desde_mensajes(respuestas)
        out: dict[str, Any] = {
            "mensajes": (estado.get("mensajes") or []) + respuestas,
            "_traza": [_paso("herramientas", llamadas=llamadas, rol=estado.get("rol"))],
        }
        if extra_ev:
            out["evidence"] = extra_ev
        sql_rows = []
        for r in respuestas:
            try:
                body = json.loads(r["content"])
            except json.JSONDecodeError:
                continue
            if isinstance(body, dict) and "rows" in body:
                sql_rows.extend(body["rows"])
        if sql_rows:
            out["sql_used"] = (estado.get("sql_used") or []) + sql_rows
        return out

    def nodo_tope(estado: Estado) -> dict:
        answer = estado.get("answer") or estado.get("draft") or ABSTENCION
        return {
            "status": "max_steps_reached",
            "answer": answer,
            "citations": estado.get("citations") or [],
            "_traza": [_paso("tope", razon="MAX_PASOS o presupuesto")],
        }

    def ruta_agente(estado: Estado) -> str:
        pasos = int(estado.get("pasos") or 0)
        tokens = int(estado.get("tokens_usados") or 0)
        agotado = pasos >= MAX_PASOS or tokens >= TOKEN_BUDGET
        mensajes = estado.get("mensajes") or []
        ultimo = mensajes[-1] if mensajes else {}
        if ultimo.get("tool_calls"):
            return "tope" if agotado else "pide_herramienta"
        rol = estado.get("rol") or "planner"
        if rol == "synthesizer":
            return "fin"
        if agotado:
            return "tope"
        return "siguiente_rol"

    def nodo_siguiente(estado: Estado) -> dict:
        actual = estado.get("rol") or "planner"
        if actual == "verifier" and estado.get("verdict") == "fail" and int(estado.get("replan") or 0) < MAX_REPLAN:
            return {
                "mensajes": [],
                "rol": "planner",
                "replan": int(estado.get("replan") or 0) + 1,
                "_traza": [_paso("handoff", desde="verifier", hacia="planner", replan=True)],
            }
        nxt = SIGUIENTE.get(actual) or "synthesizer"
        return {
            "mensajes": [],
            "rol": nxt,
            "_traza": [_paso("handoff", desde=actual, hacia=nxt)],
        }

    g = StateGraph(Estado)
    g.add_node("agente", nodo_agente)
    g.add_node("herramientas", nodo_herramientas)
    g.add_node("tope", nodo_tope)
    g.add_node("siguiente", nodo_siguiente)
    g.add_edge(START, "agente")
    g.add_conditional_edges(
        "agente",
        ruta_agente,
        {
            "pide_herramienta": "herramientas",
            "siguiente_rol": "siguiente",
            "tope": "tope",
            "fin": END,
        },
    )
    g.add_edge("herramientas", "agente")
    g.add_edge("tope", END)
    g.add_edge("siguiente", "agente")
    return g


def _aplicar_salida(rol: str, estado: Estado, datos: dict, mensajes: list[dict]) -> dict:
    out: dict[str, Any] = {}
    if rol == "planner":
        out["plan"] = datos.get("plan") or []
        out["sql_checks"] = datos.get("sql_checks") or []
        out["ruta"] = datos.get("ruta") or "docs"
    elif rol == "reader":
        extra = _hits_desde_mensajes(mensajes)
        if extra:
            # evadir duplicar si herramientas ya añadió
            existentes = {(e.get("chunk_key"), e.get("text")) for e in (estado.get("evidence") or [])}
            nuevos = [e for e in extra if (e.get("chunk_key"), e.get("text")) not in existentes]
            if nuevos:
                out["evidence"] = nuevos
    elif rol == "writer":
        out["draft"] = datos.get("draft") or datos.get("raw") or ""
        out["citations"] = datos.get("citations") or []
        if datos.get("abstain") and not out["draft"]:
            out["draft"] = ABSTENCION
    elif rol == "verifier":
        out["verdict"] = datos.get("verdict") or "fail"
        out["critique"] = datos.get("critique") or ""
        if datos.get("sql_used"):
            out["sql_used"] = datos["sql_used"]
    elif rol == "synthesizer":
        out["answer"] = datos.get("answer") or estado.get("draft") or ABSTENCION
        out["citations"] = datos.get("citations") or estado.get("citations") or []
        st = datos.get("status") or "completed"
        if estado.get("verdict") == "fail":
            st = "rejected_insufficient_evidence"
        out["status"] = st
    return out


def compile_graph(cliente: ClienteMCP):
    from rfp_agent.config import MAX_PASOS as MP

    g = construir_grafo(cliente)
    # El router es el freno; recursion_limit es red de seguridad (sesión 14).
    return g.compile(), max(4 * MP, 32)
