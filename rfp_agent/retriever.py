"""Recuperación RAG/CAG: Qdrant si hay índice; si no, stub léxico sobre facts + citas del PDF."""
from __future__ import annotations

from rfp_agent.config import (
    COLLECTION_CODE,
    COLLECTION_DOCS,
    DEMO_ATS,
    QDRANT_URL,
)
from rfp_agent.db import connect_rfp, ensure_loaded
from rfp_agent.evidence_filter import es_plantilla_invalida, es_ruido_retrieve
from rfp_agent.textutil import normalizar

STUB_CHUNKS = [
    {
        "chunk_key": "doc:descripcion_funcional.pdf:p11:irrevocable",
        "text": (
            "Liquidación de fondos final e irrevocable en tiempo real entre los miembros "
            "del servicio RTGS, de forma continua."
        ),
        "source": "descripcion_funcional.pdf",
        "locator": "p.11",
        "kb_status": "VALIDADO",
        "scope": "docs",
    },
    {
        "chunk_key": "doc:descripcion_funcional.pdf:p13:24x7",
        "text": (
            "El sistema RTGS de Montran se ha mejorado con capacidades para funcionar 24/7 "
            "en las distintas capas del producto: núcleo, servicios de valor agregado y monitoreo. "
            "La plataforma actual funciona 24/7."
        ),
        "source": "descripcion_funcional.pdf",
        "locator": "p.13",
        "kb_status": "VALIDADO",
        "scope": "docs",
    },
    {
        "chunk_key": "doc:descripcion_funcional.pdf:p15:24x7-365",
        "text": (
            "El RTGS de Montran puede funcionar y procesar pagos de forma 24/7/365 con muy poco "
            "tiempo de inactividad del procesamiento de pagos para realizar las operaciones de Fin de Día."
        ),
        "source": "descripcion_funcional.pdf",
        "locator": "p.15",
        "kb_status": "VALIDADO",
        "scope": "docs",
    },
    {
        "chunk_key": "doc:descripcion_funcional.pdf:p10:proposito",
        "text": (
            "El propósito del documento es enumerar y describir los módulos, funciones y "
            "características del sistema de Liquidación Bruta en Tiempo Real (RTGS) de Montran."
        ),
        "source": "descripcion_funcional.pdf",
        "locator": "p.10",
        "kb_status": "VALIDADO",
        "scope": "docs",
    },
]


def _qdrant_disponible() -> bool:
    try:
        from qdrant_client import QdrantClient

        cli = QdrantClient(url=QDRANT_URL, timeout=2, check_compatibility=False)
        return cli.collection_exists(COLLECTION_DOCS) or cli.collection_exists(COLLECTION_CODE)
    except Exception:
        return False


def qdrant_status() -> dict:
    docs = False
    code = False
    try:
        from qdrant_client import QdrantClient

        cli = QdrantClient(url=QDRANT_URL, timeout=2, check_compatibility=False)
        docs = bool(cli.collection_exists(COLLECTION_DOCS))
        code = bool(cli.collection_exists(COLLECTION_CODE))
    except Exception:
        pass
    return {
        "url": QDRANT_URL,
        COLLECTION_DOCS: docs,
        COLLECTION_CODE: code,
        "usable": docs or code,
    }


def _sanear_hit(h: dict) -> dict | None:
    """Descarta plantillas; marca kb_status si el texto trae [[RELLENAR]]."""
    texto = h.get("text") or ""
    status = h.get("kb_status") or "DESCONOCIDO"
    source = h.get("source") or ""
    scope = h.get("scope") or ""
    if es_plantilla_invalida(texto, status, source):
        return None
    if es_ruido_retrieve(texto, status, source, scope):
        return None
    # Si el payload no traía status pero el texto lo delata, ya se filtró arriba.
    return h


def _embed_vector(query: str):
    """Vector de consulta: cliente LLM H200, o embedder CAG (mismo bge-m3)."""
    from rfp_agent.llm import embed_query

    vec = embed_query(query)
    if vec is not None:
        return vec
    try:
        from rfp_agent.embedder import embed

        return embed([query], batch_size=1)[0].tolist()
    except Exception:
        return None


def _buscar_qdrant(query: str, scope: str, k: int) -> list[dict] | None:
    if not _qdrant_disponible():
        return None
    try:
        from qdrant_client import QdrantClient

        vec = _embed_vector(query)
        if vec is None:
            return None
        cli = QdrantClient(url=QDRANT_URL, timeout=30, check_compatibility=False)
        colecciones = []
        if scope in ("docs", "both") and cli.collection_exists(COLLECTION_DOCS):
            colecciones.append(COLLECTION_DOCS)
        if scope in ("code", "both") and cli.collection_exists(COLLECTION_CODE):
            colecciones.append(COLLECTION_CODE)
        if not colecciones:
            return None
        # Pedir de más: tras filtrar plantillas (docs) puede quedar poco.
        fetch_k = max(k * 4, 12)
        hits: list[dict] = []
        for nombre in colecciones:
            resp = cli.query_points(
                collection_name=nombre, query=vec, limit=fetch_k, with_payload=True
            )
            hit_scope = "code" if nombre == COLLECTION_CODE else "docs"
            for p in resp.points:
                payload = p.payload or {}
                page = payload.get("page")
                locator = payload.get("locator")
                if not locator:
                    if page is not None and page != "":
                        locator = f"p.{page}" if not str(page).startswith("p.") else str(page)
                    else:
                        locator = payload.get("path") or ""
                raw = {
                    "chunk_key": payload.get("chunk_key") or str(p.id),
                    "text": payload.get("text") or "",
                    "source": payload.get("doc_id") or payload.get("path") or nombre,
                    "locator": locator,
                    "kb_status": payload.get("kb_status") or "DESCONOCIDO",
                    "score": float(p.score),
                    "scope": hit_scope,
                    "project": payload.get("project") or "",
                    "class": payload.get("class") or "",
                }
                # Filtros de plantilla solo aplican a docs; en code se conservan chunks.
                if hit_scope == "docs":
                    limpio = _sanear_hit(raw)
                else:
                    if es_plantilla_invalida(
                        raw.get("text") or "",
                        raw.get("kb_status") or "",
                        raw.get("source") or "",
                    ):
                        limpio = None
                    elif (raw.get("kb_status") or "").upper() in {"ERROR", "AUSENTE"}:
                        limpio = None
                    else:
                        limpio = raw
                if limpio:
                    hits.append(limpio)
        hits.sort(key=lambda h: h["score"], reverse=True)
        return hits[:k]
    except Exception:
        return None


def _buscar_stub(query: str, scope: str, k: int) -> list[dict]:
    ensure_loaded()
    q = normalizar(query)
    tokens = [t for t in q.split() if len(t) > 3]
    candidatos = [c for c in STUB_CHUNKS if scope in ("docs", "both") or c["scope"] == scope]
    conn = connect_rfp()
    try:
        for row in conn.execute("SELECT * FROM facts"):
            candidatos.append(
                {
                    "chunk_key": f"fact:{row['id']}",
                    "text": row["statement"],
                    "source": row["source_doc"],
                    "locator": row["locator"],
                    "kb_status": row["kb_status"],
                    "scope": "docs",
                }
            )
    finally:
        conn.close()

    puntuados = []
    for c in candidatos:
        if es_plantilla_invalida(c.get("text") or "", c.get("kb_status") or "", c.get("source") or ""):
            continue
        blob = normalizar(c["text"] + " " + (c.get("source") or ""))
        score = sum(1 for t in tokens if t in blob)
        if "24/7" in query or "24x7" in q or "24 x 7" in q:
            if "24/7" in c["text"]:
                score += 3
        if "irrevoc" in q and "irrevoc" in blob:
            score += 3
        if "tps" in q or "capacidad maxima" in q or "transaccional" in q:
            score += 0
        puntuados.append((score, c))
    puntuados.sort(key=lambda x: x[0], reverse=True)
    out = []
    vistos = set()
    for score, c in puntuados:
        if c["chunk_key"] in vistos:
            continue
        vistos.add(c["chunk_key"])
        item = dict(c)
        item["score"] = float(score)
        out.append(item)
        if len(out) >= k:
            break
    return out


def retrieve_knowledge(query: str, scope: str = "both", k: int = 5) -> list[dict]:
    scope = scope if scope in ("docs", "code", "both") else "both"
    k = max(1, min(int(k or 5), 12))
    hits = _buscar_qdrant(query, scope, k)
    if hits is not None:
        # Si Qdrant solo devolvió basura filtrada, caer al stub documental.
        if not hits and scope in ("docs", "both"):
            return _buscar_stub(query, "docs", k)
        return hits
    if scope == "code":
        return [
            {
                "chunk_key": "code:unavailable",
                "text": (
                    "El índice de código (demo_ats / Qdrant montran_code) no está disponible en este "
                    f"entorno. DEMO_ATS={DEMO_ATS} exists={DEMO_ATS.exists()}."
                ),
                "source": "stub",
                "locator": "",
                "kb_status": "AUSENTE",
                "score": 0.0,
                "scope": "code",
            }
        ]
    return _buscar_stub(query, scope, k)
