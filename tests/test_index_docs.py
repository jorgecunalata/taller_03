"""Tests del RAG index-docs: PDF/DOCX/MD, kb_status, chunking, skip dirs."""
from __future__ import annotations

import io
import json
import os
from contextlib import redirect_stdout
from pathlib import Path
from unittest import mock

import numpy as np
import pytest

os.environ.setdefault("LLM_BACKEND", "simulated")

from rfp_agent.config import CHUNK_TOKENS, EMBED_DIM, OVERLAP_TOKENS
from rfp_agent.docs_pipeline import (
    construir_chunks,
    estado_kb,
    fragmentar,
    indexar_documentos,
    iter_doc_paths,
)
from rfp_agent.evidence_filter import es_plantilla_invalida
from rfp_agent.retriever import _sanear_hit


@pytest.fixture()
def mini_corpus(tmp_path: Path) -> Path:
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    (corpus / "docs").mkdir()
    # PDF real del repo (si existe) o texto plano como proxy
    real_pdf = Path(__file__).resolve().parents[1] / "corpus" / "descripcion_funcional.pdf"
    if real_pdf.exists():
        (corpus / "descripcion_funcional.pdf").write_bytes(real_pdf.read_bytes())
    else:
        (corpus / "descripcion_funcional.pdf").write_bytes(b"%PDF-1.4 fake")

    (corpus / "docs" / "plantilla_capacidad.md").write_text(
        "# Capacidad\nEJEMPLO_NO_VALIDADO\n[[RELLENAR]] TPS máximo\n" + ("x " * 120),
        encoding="utf-8",
    )
    # docx mínimo vía python-docx
    from docx import Document

    doc = Document()
    doc.add_paragraph(
        "Documento corporativo de prueba con texto suficiente para superar el piso. "
        "Describe liquidación RTGS y controles internos. " + ("palabra " * 80)
    )
    doc.save(corpus / "docs" / "nota_interna.docx")
    return corpus


def test_estado_kb_no_promueve_plantillas():
    assert estado_kb("Liquidación final e irrevocable", "descripcion_funcional.pdf") == "VALIDADO"
    assert estado_kb("hola [[RELLENAR]]", "guia.pdf") == "EJEMPLO_NO_VALIDADO"
    assert estado_kb("sin marcadores", "capacidad.md") == "EJEMPLO_NO_VALIDADO"
    assert estado_kb("EJEMPLO_NO_VALIDADO plantilla", "x.txt") == "EJEMPLO_NO_VALIDADO"


def test_fragmentar_overlap_512_102():
    tokens = [f"w{i}" for i in range(900)]
    paginas = [{"page": 3, "text": " ".join(tokens)}]
    chunks = fragmentar(paginas, "doc.pdf", "VALIDADO")
    assert chunks
    assert all(c["tokens"] <= CHUNK_TOKENS for c in chunks)
    assert chunks[0]["locator"].startswith("p.3")
    assert chunks[0]["kb_status"] == "VALIDADO"
    # Con solape, más de un chunk y segundo arranca antes del fin del primero
    if len(chunks) >= 2:
        assert OVERLAP_TOKENS > 0
        assert chunks[1]["tokens"] <= CHUNK_TOKENS


def test_iter_doc_paths_skips_missing_docs_dir(tmp_path: Path):
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    (corpus / "solo.pdf").write_bytes(b"%PDF-1.4\n")
    paths, dirs = iter_doc_paths(corpus)
    assert dirs["corpus"]["exists"] is True
    assert dirs["corpus/docs"]["skipped"] is True
    assert any(p.name == "solo.pdf" for p in paths)


def test_construir_chunks_pdf_y_plantilla(mini_corpus: Path):
    chunks, summary = construir_chunks(corpus=mini_corpus)
    assert summary["dirs"]["corpus"]["exists"] is True
    assert summary["files_seen"] >= 2
    # PDF funcional → VALIDADO; md → EJEMPLO_NO_VALIDADO
    statuses = {c["doc_id"]: c["kb_status"] for c in chunks}
    if "descripcion_funcional.pdf" in statuses:
        assert statuses["descripcion_funcional.pdf"] == "VALIDADO"
    assert statuses.get("plantilla_capacidad.md") == "EJEMPLO_NO_VALIDADO"
    # Plantilla indexada pero filtrable en retrieve
    plantilla = [c for c in chunks if c["doc_id"] == "plantilla_capacidad.md"]
    assert plantilla
    assert es_plantilla_invalida(
        plantilla[0]["text"], plantilla[0]["kb_status"], plantilla[0]["doc_id"]
    )
    assert (
        _sanear_hit(
            {
                "text": plantilla[0]["text"],
                "kb_status": plantilla[0]["kb_status"],
                "source": plantilla[0]["doc_id"],
            }
        )
        is None
    )


def test_indexar_documentos_memory_qdrant(mini_corpus: Path, tmp_path: Path, monkeypatch):
    from qdrant_client import QdrantClient

    mem = QdrantClient(location=":memory:")

    def _cliente(url=None, timeout=120):
        return mem

    import rfp_agent.store as store_mod

    monkeypatch.setattr(store_mod, "cliente", _cliente)

    fake = np.random.default_rng(1).standard_normal((1, EMBED_DIM)).astype("float32")
    fake /= np.linalg.norm(fake, axis=1, keepdims=True)

    def _fake_embed(textos, batch_size=64):
        n = len(textos)
        arr = np.repeat(fake, n, axis=0).copy()
        arr += np.linspace(0, 0.01, n * EMBED_DIM, dtype="float32").reshape(n, EMBED_DIM)
        normas = np.linalg.norm(arr, axis=1, keepdims=True)
        return arr / np.clip(normas, 1e-9, None)

    monkeypatch.setattr("rfp_agent.docs_pipeline.embed", _fake_embed)
    monkeypatch.setattr("rfp_agent.docs_pipeline.backend_embeddings", lambda: "fake")

    out = tmp_path / "outputs"
    out.mkdir()
    monkeypatch.setattr("rfp_agent.docs_pipeline.OUT", out)
    monkeypatch.setattr("rfp_agent.docs_pipeline.CHUNKS_PATH", out / "doc_chunks.jsonl")
    monkeypatch.setattr("rfp_agent.docs_pipeline.INGESTA_PATH", out / "ingesta_documentos.json")

    summary = indexar_documentos(corpus=mini_corpus, qdrant_url="memory", recreate=True)
    assert summary["files_indexed"] >= 1
    assert summary["chunks"] >= 1
    assert summary["points_upserted"] == summary["chunks"]
    assert mem.collection_exists("documentos")
    data = json.loads((out / "ingesta_documentos.json").read_text(encoding="utf-8"))
    assert data["collection"] == "documentos"
    # Plantilla no marcada VALIDADO
    for fila in data["archivos"]:
        if str(fila.get("nombre", "")).endswith(".md"):
            assert fila.get("kb_status") == "EJEMPLO_NO_VALIDADO"


def test_cli_index_docs_help_and_listing():
    from rfp_agent.cli import main

    with pytest.raises(SystemExit) as ei:
        main(["index-docs", "-h"])
    assert ei.value.code == 0

    buf = io.StringIO()
    with redirect_stdout(buf):
        with pytest.raises(SystemExit) as ei2:
            main(["-h"])
    assert ei2.value.code == 0
    out = buf.getvalue()
    assert "index-docs" in out
    assert "index-code" in out


def test_indexar_documentos_corpus_ausente(tmp_path: Path, monkeypatch):
    missing = tmp_path / "no_corpus"
    out = tmp_path / "outputs"
    out.mkdir()
    monkeypatch.setattr("rfp_agent.docs_pipeline.OUT", out)
    monkeypatch.setattr("rfp_agent.docs_pipeline.CHUNKS_PATH", out / "doc_chunks.jsonl")
    monkeypatch.setattr("rfp_agent.docs_pipeline.INGESTA_PATH", out / "ingesta_documentos.json")
    summary = indexar_documentos(corpus=missing, qdrant_url="memory", recreate=True)
    assert summary["files_seen"] == 0
    assert summary["dirs"]["corpus"]["skipped"] is True
    assert "warning" in summary
