"""Embeddings bge-m3: H200 Ollama (:11434) o fallback local sentence-transformers.

EMBEDDING_BACKEND:
  auto  — intenta H200; si falla y hay sentence-transformers, local; si no, error claro
  h200  — solo H200 (VPN GlobalProtect)
  local — solo sentence-transformers BAAI/bge-m3 (CPU/MPS)
"""
from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from functools import lru_cache

import numpy as np

from rfp_agent.config import (
    EMBED_DIM,
    EMBED_MAX_TOKENS,
    EMBED_MODEL,
    EMBED_OLLAMA_NAME,
    EMBEDDING_BACKEND,
    H200_API_KEY,
    H200_EMBED_PORT,
    H200_HOST,
)

_BACKEND_USED: str | None = None


class EmbeddingsUnavailable(RuntimeError):
    """Sin H200 ni fallback local usable."""


def _mensaje_sin_embeddings() -> str:
    return (
        "No hay embeddings disponibles para index-code / index-docs.\n"
        f"  H200 Ollama bge-m3: http://{H200_HOST}:{H200_EMBED_PORT} "
        "(GlobalProtect + EMBEDDING_BACKEND=h200|auto).\n"
        "  Sin VPN: pip install 'sentence-transformers' torch && "
        "EMBEDDING_BACKEND=local (mismo modelo BAAI/bge-m3; reindexa si cambias de ruta)."
    )


def contar_tokens(texto: str) -> int:
    """Aprox. sin transformers: ~4 chars/token (solo presupuesto de catálogo)."""
    return max(1, len(texto or "") // 4)


def _recortar_chars(texto: str, max_tokens: int = EMBED_MAX_TOKENS) -> str:
    # bge-m3 8192 tokens ≈ ~32k chars; recorte duro por seguridad.
    limite = max_tokens * 4
    t = texto or ""
    return t if len(t) <= limite else t[:limite]


def _h200_ollama(lote: list[str], timeout: float = 300) -> list[list[float]]:
    """POST /api/embed (contrato Taller 2) con fallback a /v1/embeddings."""
    url_api = f"http://{H200_HOST}:{H200_EMBED_PORT}/api/embed"
    cuerpo = json.dumps({"model": EMBED_OLLAMA_NAME, "input": lote}).encode()
    req = urllib.request.Request(
        url_api,
        data=cuerpo,
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {H200_API_KEY}",
        },
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            data = json.loads(resp.read())
        emb = data.get("embeddings")
        if emb:
            return emb
    except (urllib.error.URLError, TimeoutError, json.JSONDecodeError, KeyError):
        pass

    url_v1 = f"http://{H200_HOST}:{H200_EMBED_PORT}/v1/embeddings"
    cuerpo_v1 = json.dumps({"model": EMBED_OLLAMA_NAME, "input": lote}).encode()
    req_v1 = urllib.request.Request(
        url_v1,
        data=cuerpo_v1,
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {H200_API_KEY}",
        },
    )
    with urllib.request.urlopen(req_v1, timeout=timeout) as resp:
        data = json.loads(resp.read())
    ordenados = sorted(data["data"], key=lambda d: d["index"])
    return [d["embedding"] for d in ordenados]


@lru_cache(maxsize=1)
def _local_model():
    import torch
    from sentence_transformers import SentenceTransformer

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    return SentenceTransformer(EMBED_MODEL, device=device)


def _local(lote: list[str]) -> list[list[float]]:
    arr = _local_model().encode(lote, batch_size=min(16, len(lote) or 1))
    return np.asarray(arr, dtype=np.float32).tolist()


def probe_h200(timeout: float = 3.0) -> bool:
    try:
        _h200_ollama(["ping"], timeout=timeout)
        return True
    except Exception:
        return False


def probe_local() -> bool:
    try:
        import sentence_transformers  # noqa: F401
        import torch  # noqa: F401

        return True
    except Exception:
        return False


def backend_embeddings() -> str:
    global _BACKEND_USED
    if _BACKEND_USED:
        return _BACKEND_USED
    # Fuerza resolución.
    embed(["ok"], batch_size=1)
    return _BACKEND_USED or "unknown"


def embed(textos: list[str], batch_size: int = 64) -> np.ndarray:
    """Devuelve matriz (n, 1024) L2-normalizada."""
    global _BACKEND_USED
    if not textos:
        return np.zeros((0, EMBED_DIM), dtype=np.float32)

    textos = [_recortar_chars(t) for t in textos]
    backend = (os.environ.get("EMBEDDING_BACKEND") or EMBEDDING_BACKEND or "auto").lower()
    vectores: list[list[float]] = []

    def _run_h200() -> None:
        nonlocal vectores
        for i in range(0, len(textos), batch_size):
            vectores.extend(_h200_ollama(textos[i : i + batch_size]))

    def _run_local() -> None:
        nonlocal vectores
        for i in range(0, len(textos), min(batch_size, 16)):
            vectores.extend(_local(textos[i : i + min(batch_size, 16)]))

    if backend == "h200":
        try:
            _run_h200()
            _BACKEND_USED = "h200"
        except Exception as exc:
            raise EmbeddingsUnavailable(_mensaje_sin_embeddings()) from exc
    elif backend == "local":
        try:
            _run_local()
            _BACKEND_USED = "local"
        except Exception as exc:
            raise EmbeddingsUnavailable(_mensaje_sin_embeddings()) from exc
    else:
        # auto
        try:
            _run_h200()
            _BACKEND_USED = "h200"
        except Exception:
            try:
                _run_local()
                _BACKEND_USED = "local"
            except Exception as exc:
                raise EmbeddingsUnavailable(_mensaje_sin_embeddings()) from exc

    arr = np.asarray(vectores, dtype=np.float32)
    if arr.ndim != 2 or arr.shape[1] != EMBED_DIM:
        raise EmbeddingsUnavailable(
            f"Dimensión de embedding inesperada: {arr.shape}; se esperaba (*, {EMBED_DIM})."
        )
    normas = np.linalg.norm(arr, axis=1, keepdims=True)
    return arr / np.clip(normas, 1e-9, None)


def reset_embedder() -> None:
    global _BACKEND_USED
    _BACKEND_USED = None
    _local_model.cache_clear()
