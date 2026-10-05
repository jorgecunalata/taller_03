"""Tests del primer slice: carga 174, MCP, grafo simulado, eval SQL."""
from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

os.environ["LLM_BACKEND"] = "simulated"

from rfp_agent import graph as graph_mod
from rfp_agent.config import ABSTENCION, GOLDEN_IDS, ROOT
from rfp_agent.db import get_item, load_rfp
from rfp_agent.evaluate import eval_rfp
from rfp_agent.llm import reset_llm
from rfp_agent.mcp import ClienteMCP, construir_servidor
from rfp_agent.workflow import RfpWorkflow


@pytest.fixture(scope="session", autouse=True)
def _db(tmp_path_factory):
    reset_llm()
    data = tmp_path_factory.mktemp("data")
    os.environ["RFP_DB"] = str(data / "rfp.sqlite")
    os.environ["RUNS_DB"] = str(data / "runs.sqlite")
    os.environ["LLM_BACKEND"] = "simulated"
    # config already imported RFP_DB — patch modules
    import rfp_agent.config as cfg
    import rfp_agent.db as db

    cfg.RFP_DB = data / "rfp.sqlite"
    cfg.RUNS_DB = data / "runs.sqlite"
    db.RFP_DB = cfg.RFP_DB
    db.RUNS_DB = cfg.RUNS_DB
    load_rfp(ROOT / "data" / "rfp.xlsx", cfg.RFP_DB)
    yield


def test_carga_174_y_tres_golden():
    assert get_item("R-001")["excel_row"] == 2
    assert get_item("R-174") is not None
    assert get_item("R-175") is None
    from rfp_agent.db import connect_rfp

    conn = connect_rfp()
    n = conn.execute("SELECT COUNT(*) FROM rfp_items").fetchone()[0]
    g = conn.execute("SELECT COUNT(*) FROM rfp_items WHERE in_golden=1").fetchone()[0]
    ids = [r[0] for r in conn.execute("SELECT id FROM rfp_items WHERE in_golden=1").fetchall()]
    conn.close()
    assert n == 174
    assert g == 3
    assert set(ids) == set(GOLDEN_IDS)


def test_mcp_cuatro_tools_y_n_mas_uno():
    cli = ClienteMCP(construir_servidor())
    names = cli.descubrir()
    for t in ("retrieve_knowledge", "get_requirement", "query_facts", "record_evidence"):
        assert t in names
    assert "list_sources" in names  # quinta, sin reescribir nodos
    req = cli.invocar("get_requirement", {"requirement_id": "R-002"})
    assert req["id"] == "R-002"
    assert req["in_golden"] is False
    assert req["sql_verificacion"] is None
    facts = cli.invocar(
        "query_facts",
        {"sql": "SELECT statement FROM facts WHERE req_id = 'R-001'"},
    )
    assert facts["rows"]
    bad = cli.invocar("query_facts", {"sql": "DELETE FROM facts"})
    assert "error" in bad
    ev = cli.invocar(
        "record_evidence",
        {"requirement_id": "R-001", "chunk_key": "k", "cita": "x"},
    )
    ev2 = cli.invocar(
        "record_evidence",
        {"requirement_id": "R-001", "chunk_key": "k", "cita": "x"},
    )
    assert ev["stored"] == ev2["stored"] == 1
    assert ev["chunk_key"] == "k"
    rag = cli.invocar("retrieve_knowledge", {"query": "irrevocable tiempo real", "scope": "docs", "k": 3})
    assert rag["hits"]
    assert all("kb_status" in h for h in rag["hits"])


def test_no_create_react_agent():
    src = Path(graph_mod.__file__).read_text(encoding="utf-8")
    assert "create_react_agent" not in src
    assert "create_agent" not in src
    assert "StateGraph" in src


def test_eval_rfp_tres_golden():
    resumen = eval_rfp()
    assert resumen["n"] == 3
    by_id = {i["id"]: i for i in resumen["items"]}
    assert "R-018" not in by_id
    assert by_id["R-001"]["pass"]
    assert by_id["R-038"]["pass"]
    assert by_id["R-057"]["pass"]
    assert ABSTENCION.lower() in by_id["R-057"]["answer"].lower()
    assert resumen["failed"] == 0
    for rid in ("R-001", "R-038", "R-057"):
        assert by_id[rid]["status"] == "completed"


def test_grafo_cinco_roles():
    wf = RfpWorkflow()
    out = wf.run("R-001")
    nodos = [t.get("nodo") for t in out["trace"]]
    for rol in ("planner", "reader", "writer", "verifier", "synthesizer"):
        assert rol in nodos
    assert out["status"] == "completed"
    assert out["pasos"] < out["max_pasos"]
    tools = wf.cliente.descubrir()
    assert len(tools) >= 4


def test_sin_r018_en_golden_config():
    from rfp_agent import db as db_mod
    from rfp_agent.config import FORBIDDEN_GOLDEN_IDS, GOLDEN_IDS

    assert "R-018" not in GOLDEN_IDS
    assert "R-018" in FORBIDDEN_GOLDEN_IDS
    assert all(g["id"] != "R-018" for g in db_mod.GOLDEN_ROWS)
    assert all(f["req_id"] != "R-018" for f in db_mod.FACTS_ROWS)


def test_purge_r018_from_sqlite():
    """Aunque alguien inserte R-018 a mano, ensure_loaded / load-db lo borra."""
    from rfp_agent.config import GOLDEN_IDS
    from rfp_agent.db import connect_rfp, ensure_loaded, load_rfp

    conn = connect_rfp()
    conn.execute(
        "INSERT OR REPLACE INTO rfp_items (id, texto, excel_row, in_golden) VALUES (?,?,?,1)",
        ("R-018", "SWIFT placeholder", 19),
    )
    conn.execute(
        "INSERT OR REPLACE INTO golden (id, respondible, sql_verificacion, notas) VALUES (?,?,?,?)",
        ("R-018", 1, "SELECT 1", "bogus"),
    )
    conn.execute(
        "INSERT OR REPLACE INTO facts (id, req_id, statement, source_doc, locator, kb_status) "
        "VALUES (?,?,?,?,?,?)",
        ("F-018", "R-018", "<exact phrase copied from the PDF>", "x", "p.0", "VALIDADO"),
    )
    conn.commit()
    conn.close()

    ensure_loaded()
    conn = connect_rfp()
    ids = [r[0] for r in conn.execute("SELECT id FROM golden").fetchall()]
    facts_018 = conn.execute("SELECT COUNT(*) FROM facts WHERE req_id='R-018'").fetchone()[0]
    conn.close()
    assert "R-018" not in ids
    assert facts_018 == 0
    assert set(ids) == set(GOLDEN_IDS)

    info = load_rfp()
    assert info["golden"] == 3
    assert "R-018" not in info["golden_ids"]


def test_reader_cierra_sin_quemar_pasos():
    wf = RfpWorkflow()
    out = wf.run("R-001")
    nodos = [t.get("nodo") for t in out["trace"]]
    assert "cerrar_rol" in nodos or "writer" in nodos
    assert out["status"] == "completed"
    assert "liquidación de fondos final e irrevocable en tiempo real" in out["answer"].lower()
    assert out["pasos"] < out["max_pasos"]
    # No debe quedarse atrapado en reader hasta tope.
    assert "tope" not in nodos


def test_normalize_assistant_thinking_and_embedded_tools():
    from rfp_agent.llm import normalize_assistant_message

    m = normalize_assistant_message(
        {
            "role": "assistant",
            "content": (
                "<think>voy a llamar</think>"
                '<tool_call>{"name":"retrieve_knowledge","arguments":{"query":"24/7","scope":"docs"}}</tool_call>'
            ),
        }
    )
    assert m.get("tool_calls")
    assert m["tool_calls"][0]["function"]["name"] == "retrieve_knowledge"
    args = json.loads(m["tool_calls"][0]["function"]["arguments"])
    assert args["query"] == "24/7"


def test_draft_incluye_hechos_golden():
    from rfp_agent.drafting import draft_desde_evidence

    d1 = draft_desde_evidence(
        "R-001",
        "liquidación",
        [{"chunk_key": "k", "text": "Liquidación de fondos final e irrevocable en tiempo real", "source": "descripcion_funcional.pdf", "locator": "p.11", "kb_status": "VALIDADO"}],
    )
    assert "liquidación de fondos final e irrevocable en tiempo real" in d1["draft"].lower()
    assert "descripcion_funcional" in json.dumps(d1["citations"]).lower()
    d38 = draft_desde_evidence(
        "R-038",
        "24x7",
        [
            {"chunk_key": "a", "text": "El sistema RTGS de Montran se ha mejorado con capacidades para funcionar 24/7", "source": "descripcion_funcional.pdf", "locator": "p.13", "kb_status": "VALIDADO"},
            {"chunk_key": "b", "text": "El RTGS de Montran puede funcionar y procesar pagos de forma 24/7/365", "source": "descripcion_funcional.pdf", "locator": "p.15", "kb_status": "VALIDADO"},
        ],
    )
    low = d38["draft"].lower()
    assert "24/7" in low and "24/7/365" in low
    d57 = draft_desde_evidence("R-057", "Capacidad máxima transaccional TPS", [])
    assert ABSTENCION.lower() in d57["draft"].lower()


def test_query_facts_reescribe_requirement_id():
    from rfp_agent.mcp import ClienteMCP, construir_servidor, reescribir_sql_facts

    assert "req_id" in reescribir_sql_facts(
        "SELECT statement FROM facts WHERE requirement_id = 'R-001'"
    )
    cli = ClienteMCP(construir_servidor())
    # Antes fallaba con no such column: requirement_id
    r = cli.invocar(
        "query_facts",
        {"sql": "SELECT statement, source_doc, locator FROM facts WHERE requirement_id = 'R-001'"},
    )
    assert "error" not in r or not r.get("error")
    assert r.get("rows")
    assert "Liquidación" in (r["rows"][0].get("statement") or "")


def test_limpiar_ruido_y_citas_pdf():
    from rfp_agent.drafting import corregir_draft_si_abstuvo_mal, limpiar_ruido_tool
    from rfp_agent.evaluate import _citas_pdf

    sucio = (
        "Liquidación de fondos final e irrevocable en tiempo real [1] "
        "ValueError: sql inválido: no such column: requirement_id"
    )
    limpio = limpiar_ruido_tool(sucio)
    assert "valueerror" not in limpio.lower()
    assert "requirement_id" not in limpio.lower()
    fixed = corregir_draft_si_abstuvo_mal("R-001", "liquidación", sucio, [], [{"n": 1}])
    assert "valueerror" not in fixed["draft"].lower()
    assert "descripcion_funcional" in json.dumps(fixed["citations"]).lower()
    assert _citas_pdf(fixed["citations"], [], fixed["draft"])
    # [n] + evidence con PDF también cuenta
    assert _citas_pdf(
        [{"n": 1}],
        [{"source": "descripcion_funcional.pdf", "locator": "p.11"}],
        "texto [1]",
    )
