"""Tests del CAG index-code: resolución case-insensitive, chunking, skip missing."""
from __future__ import annotations

import json
import os
from pathlib import Path
from unittest import mock

import numpy as np
import pytest

os.environ.setdefault("LLM_BACKEND", "simulated")

from rfp_agent.config import CODE_ROOTS, EMBED_DIM, resolve_child
from rfp_agent.code_pipeline import construir_catalogo, construir_chunks, indexar_codigo
from rfp_agent.evidence_filter import es_ruido_retrieve


@pytest.fixture()
def mini_demo(tmp_path: Path) -> Path:
    """Árbol lowercase como en el Mac; sin core_ms."""
    demo = tmp_path / "demo_ats"
    (demo / "core" / "src").mkdir(parents=True)
    (demo / "core_rtgs" / "src").mkdir(parents=True)
    (demo / "core_parser").mkdir()
    # Extra que no debe indexarse aunque exista
    (demo / "demo_ats_release").mkdir()
    (demo / "core_runtime").mkdir()

    java = demo / "core_rtgs" / "src" / "SettlementService.java"
    java.write_text(
        """
package com.montran.rtgs;

/**
 * Liquidación bruta en tiempo real.
 */
public class SettlementService {
    public void settle(String paymentId) {
        // irrevocable finality
        applyFinality(paymentId);
    }

    private void applyFinality(String paymentId) {
        System.out.println(paymentId);
    }
}
""".strip(),
        encoding="utf-8",
    )
    (demo / "core" / "src" / "app.properties").write_text(
        "app.name=montran-core\nfeature.rtgs=true\n" + ("x=1\n" * 20),
        encoding="utf-8",
    )
    (demo / "core_parser" / "messages.xml").write_text(
        '<?xml version="1.0"?>\n<root><msg type="pacs.008"/></root>\n' + ("<!-- pad -->\n" * 10),
        encoding="utf-8",
    )
    return demo


def test_resolve_child_case_insensitive(tmp_path: Path):
    parent = tmp_path / "demo"
    (parent / "core_rtgs").mkdir(parents=True)
    found = resolve_child(parent, "CORE_RTGS")
    assert found is not None
    assert found.name == "core_rtgs"
    assert resolve_child(parent, "CORE_MS") is None


def test_construir_chunks_skips_missing_and_extras(mini_demo: Path):
    chunks, summary = construir_chunks(demo=mini_demo)
    roots = summary["roots"]
    assert roots["CORE_MS"]["skipped"] is True
    assert roots["CORE_RTGS"]["exists"] is True
    assert roots["CORE_RTGS"]["files"] >= 1
    assert roots["CORE"]["files"] >= 1
    assert summary["files_read"] >= 2
    assert any(c["role"] == "clase" for c in chunks)
    assert any(c.get("class") == "SettlementService" for c in chunks)
    # Extras no están en CODE_ROOTS → no aparecen como raíces indexadas
    assert "demo_ats_release" not in roots
    assert "core_runtime" not in {k.lower() for k in roots}
    catalogo = construir_catalogo(chunks)
    assert "core_rtgs:" in catalogo


def test_es_ruido_permite_codigo_en_scope_code():
    texto = "package com.x;\n" + ("public class Foo { void bar() {} }\n" * 80)
    assert es_ruido_retrieve(texto, "CODIGO", "Foo.java", scope="code") is False
    assert es_ruido_retrieve(texto, "CODIGO", "Foo.java", scope="docs") is True
    assert es_ruido_retrieve("[[RELLENAR]] plantilla", "EJEMPLO_NO_VALIDADO", "x.md", scope="docs")


def test_indexar_codigo_memory_qdrant(mini_demo: Path, tmp_path: Path, monkeypatch):
    from qdrant_client import QdrantClient

    # Cliente en memoria compartido vía monkeypatch de store.cliente
    mem = QdrantClient(location=":memory:")

    def _cliente(url=None, timeout=120):
        return mem

    import rfp_agent.store as store_mod

    monkeypatch.setattr(store_mod, "cliente", _cliente)

    fake = np.random.default_rng(0).standard_normal((1, EMBED_DIM)).astype("float32")
    fake /= np.linalg.norm(fake, axis=1, keepdims=True)

    def _fake_embed(textos, batch_size=64):
        n = len(textos)
        arr = np.repeat(fake, n, axis=0).copy()
        # variación mínima por fila para no colapsar todo
        arr += np.linspace(0, 0.01, n * EMBED_DIM, dtype="float32").reshape(n, EMBED_DIM)
        normas = np.linalg.norm(arr, axis=1, keepdims=True)
        return arr / np.clip(normas, 1e-9, None)

    monkeypatch.setattr("rfp_agent.code_pipeline.embed", _fake_embed)
    monkeypatch.setattr("rfp_agent.code_pipeline.backend_embeddings", lambda: "fake")

    out = tmp_path / "outputs"
    out.mkdir()
    monkeypatch.setattr("rfp_agent.code_pipeline.OUT", out)
    monkeypatch.setattr("rfp_agent.code_pipeline.CHUNKS_PATH", out / "code_chunks.jsonl")
    monkeypatch.setattr("rfp_agent.code_pipeline.CATALOG_PATH", out / "catalogo_cag.txt")
    monkeypatch.setattr("rfp_agent.code_pipeline.INGESTA_PATH", out / "ingesta_codigo.json")

    summary = indexar_codigo(demo=mini_demo, qdrant_url="memory", recreate=True)
    assert summary["files_read"] >= 2
    assert summary["points_upserted"] >= summary["chunks"]
    assert mem.collection_exists("montran_code")
    assert (out / "ingesta_codigo.json").exists()
    data = json.loads((out / "ingesta_codigo.json").read_text(encoding="utf-8"))
    assert data["roots"]["CORE_MS"]["skipped"] is True


def test_cli_index_code_help():
    from rfp_agent.cli import main

    with pytest.raises(SystemExit) as ei:
        main(["index-code", "-h"])
    assert ei.value.code == 0


def test_cli_lists_index_code():
    from rfp_agent.cli import main
    import io
    from contextlib import redirect_stdout

    buf = io.StringIO()
    with redirect_stdout(buf):
        with pytest.raises(SystemExit) as ei:
            main(["-h"])
    assert ei.value.code == 0
    assert "index-code" in buf.getvalue()
