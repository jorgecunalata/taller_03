"""Tests del primer slice: carga 174, MCP, grafo simulado, eval SQL."""
from __future__ import annotations

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
    assert by_id["R-001"]["pass"]
    assert by_id["R-038"]["pass"]
    assert by_id["R-057"]["pass"]
    assert ABSTENCION.lower() in by_id["R-057"]["answer"].lower()
    assert resumen["failed"] == 0


def test_grafo_cinco_roles():
    wf = RfpWorkflow()
    out = wf.run("R-001")
    nodos = [t.get("nodo") for t in out["trace"]]
    for rol in ("planner", "reader", "writer", "verifier", "synthesizer"):
        assert rol in nodos
    assert out["status"] in {"completed", "max_steps_reached", "rejected_insufficient_evidence"}
    tools = wf.cliente.descubrir()
    assert len(tools) >= 4
