"""Cliente Qdrant para colecciones densas (CAG código / docs)."""
from __future__ import annotations

import uuid
from typing import Any

from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams

from rfp_agent.config import EMBED_DIM, QDRANT_URL


def cliente(url: str | None = None, timeout: float = 120) -> QdrantClient:
    return QdrantClient(
        url=url or QDRANT_URL,
        timeout=timeout,
        check_compatibility=False,
    )


def recrear(nombre: str, url: str | None = None) -> None:
    cli = cliente(url=url)
    if cli.collection_exists(nombre):
        cli.delete_collection(nombre)
    cli.create_collection(
        collection_name=nombre,
        vectors_config=VectorParams(size=EMBED_DIM, distance=Distance.COSINE),
    )


def upsert(
    nombre: str,
    ids: list[str],
    vectores: Any,
    payloads: list[dict],
    lote: int = 64,
    url: str | None = None,
) -> None:
    cli = cliente(url=url)
    puntos: list[PointStruct] = []
    for clave, vec, payload in zip(ids, vectores, payloads):
        cuerpo = dict(payload)
        cuerpo["chunk_key"] = clave
        if hasattr(vec, "tolist"):
            vector = vec.tolist()
        else:
            vector = list(vec)
        puntos.append(
            PointStruct(
                id=str(uuid.uuid5(uuid.NAMESPACE_URL, clave)),
                vector=vector,
                payload=cuerpo,
            )
        )
        if len(puntos) >= lote:
            cli.upsert(collection_name=nombre, points=puntos)
            puntos = []
    if puntos:
        cli.upsert(collection_name=nombre, points=puntos)
