"""Índice CAG de código Montran (ideas Taller 2: chunking Java/XML + catálogo).

No vende el zip de taller02: reimplementa lo necesario para `index-code`.
"""
from __future__ import annotations

import json
import re
from collections import defaultdict
from pathlib import Path

from rfp_agent.config import (
    CATALOG_TOKEN_BUDGET,
    CODE_EMBED_BATCH,
    CODE_EXTS,
    CODE_ROOTS,
    COLLECTION_CODE,
    DEMO_ATS,
    MAX_CODE_CHUNKS,
    OUT,
    PROJECT_PURPOSE,
    QDRANT_URL,
    RTGS_PROJECTS,
    SKIP_DIR_NAMES,
    resolve_code_root,
)
from rfp_agent.embedder import EmbeddingsUnavailable, backend_embeddings, contar_tokens, embed
from rfp_agent.store import recrear, upsert

CHUNKS_PATH = OUT / "code_chunks.jsonl"
CATALOG_PATH = OUT / "catalogo_cag.txt"
INGESTA_PATH = OUT / "ingesta_codigo.json"


def _proyecto_key(root_name: str) -> str:
    return root_name.lower()


def _caminar_root(root: Path, root_name: str):
    for path in root.rglob("*"):
        if not path.is_file() or path.suffix.lower() not in CODE_EXTS:
            continue
        if any(p in SKIP_DIR_NAMES for p in path.parts):
            continue
        try:
            if path.stat().st_size > 400_000:
                continue
        except OSError:
            continue
        yield path, root_name


def _javadoc_antes(texto: str, pos: int) -> str:
    previo = texto[max(0, pos - 800) : pos]
    m = re.search(
        r"/\*\*((?:(?!\*/).)*)\*/\s*(?:@\w+(?:\([^)]*\))?\s*)*"
        r"(?:(?:public|protected|private|abstract|final|static)\s+)*$",
        previo,
        re.S,
    )
    if not m:
        return ""
    linea = re.sub(r"\s*\*\s*", " ", m.group(1))
    linea = re.sub(r"\s+", " ", linea).strip()
    return linea[:240]


def _chunk_java(path: Path, texto: str, proyecto: str, rel: str, con_metodos: bool) -> list[dict]:
    pkg = ""
    m = re.search(r"^\s*package\s+([\w.]+)\s*;", texto, re.M)
    if m:
        pkg = m.group(1)
    clase = path.stem
    pos_clase = texto.find(f"class {clase}")
    if pos_clase < 0:
        pos_clase = texto.find(f"interface {clase}")
    if pos_clase < 0:
        pos_clase = texto.find(f"enum {clase}")
    doc = _javadoc_antes(texto, pos_clase) if pos_clase >= 0 else ""
    firmas = []
    for linea in texto.splitlines():
        s = linea.strip()
        if s.startswith(("public ", "protected ", "private ")) and "(" in s and not s.endswith(";"):
            firmas.append(s[:180])
            if len(firmas) >= 25:
                break
    proposito = PROJECT_PURPOSE.get(proyecto, "")
    cabecera = (
        f"{proyecto} › {pkg}.{clase}\n{proposito}\n{doc}\n" + "\n".join(firmas)
    ).strip()
    chunks = [
        {
            "chunk_key": f"{rel}::clase",
            "doc_id": path.name,
            "path": rel,
            "project": proyecto,
            "package": pkg,
            "class": clase,
            "method": "",
            "role": "clase",
            "text": cabecera[:4000],
            "source_type": "codigo",
            "kb_status": "CODIGO",
            "language": "en",
        }
    ]
    if not con_metodos:
        return chunks
    for mth in re.finditer(
        r"(?m)^[ \t]*((?:public|protected|private)[^\n{;]{0,160}\([^;\n]*\)[^\n{;]*)\{",
        texto,
    ):
        nombre = mth.group(1).strip().split("(")[0].split()[-1]
        inicio = mth.end()
        profundidad = 1
        i = inicio
        while i < len(texto) and profundidad:
            if texto[i] == "{":
                profundidad += 1
            elif texto[i] == "}":
                profundidad -= 1
            i += 1
        cuerpo = texto[mth.start() : min(i, mth.start() + 1500)]
        chunks.append(
            {
                "chunk_key": f"{rel}::{nombre}:{mth.start()}",
                "doc_id": path.name,
                "path": rel,
                "project": proyecto,
                "package": pkg,
                "class": clase,
                "method": nombre,
                "role": "metodo",
                "text": f"{proyecto} › {pkg}.{clase}#{nombre}\n{cuerpo}",
                "source_type": "codigo",
                "kb_status": "CODIGO",
                "language": "en",
            }
        )
        if len(chunks) > 40:
            break
    return chunks


def _chunk_otro(path: Path, texto: str, proyecto: str, rel: str) -> dict:
    recorte = texto[:2500]
    return {
        "chunk_key": f"{rel}::archivo",
        "doc_id": path.name,
        "path": rel,
        "project": proyecto,
        "package": "",
        "class": "",
        "method": "",
        "role": path.suffix.lower().lstrip(".") or "archivo",
        "text": f"{proyecto} › {rel}\n{PROJECT_PURPOSE.get(proyecto, '')}\n{recorte}",
        "source_type": "codigo",
        "kb_status": "CODIGO",
        "language": "en",
    }


def construir_chunks(demo: Path | None = None) -> tuple[list[dict], dict]:
    """Recorre CODE_ROOTS bajo DEMO_ATS; omite raíces ausentes."""
    base = demo or DEMO_ATS
    por_raiz: dict[str, dict] = {}
    clase_chunks: list[dict] = []
    metodo_chunks: list[dict] = []
    omitidos_global = 0

    for root_name in CODE_ROOTS:
        # Permitir override de DEMO_ATS en tests vía `demo`.
        if demo is not None:
            root = None
            direct = demo / root_name
            if direct.exists():
                root = direct
            else:
                target = root_name.lower()
                try:
                    for child in demo.iterdir():
                        if child.name.lower() == target:
                            root = child
                            break
                except OSError:
                    root = None
        else:
            root = resolve_code_root(root_name)

        entry = {
            "canonical": root_name,
            "resolved": str(root) if root else None,
            "exists": bool(root and root.is_dir()),
            "files": 0,
            "chunks": 0,
            "skipped": False,
        }
        if not root or not root.is_dir():
            entry["skipped"] = True
            entry["reason"] = "missing"
            por_raiz[root_name] = entry
            continue

        vistos = omitidos = 0
        chunks_raiz = 0
        proyecto = _proyecto_key(root.name)
        for path, _ in _caminar_root(root, root_name):
            try:
                texto = path.read_text(encoding="utf-8", errors="replace")
            except OSError:
                omitidos += 1
                continue
            if len(texto.strip()) < 40:
                omitidos += 1
                continue
            vistos += 1
            try:
                rel = str(path.relative_to(base))
            except ValueError:
                rel = str(path)
            if path.suffix.lower() == ".java":
                piezas = _chunk_java(path, texto, proyecto, rel, con_metodos=True)
                clase_chunks.append(piezas[0])
                metodo_chunks.extend(piezas[1:])
                chunks_raiz += len(piezas)
            else:
                clase_chunks.append(_chunk_otro(path, texto, proyecto, rel))
                chunks_raiz += 1
        entry["files"] = vistos
        entry["chunks"] = chunks_raiz
        entry["files_omitted"] = omitidos
        por_raiz[root_name] = entry
        omitidos_global += omitidos

    todos = clase_chunks + metodo_chunks
    truncado = False
    if len(todos) > MAX_CODE_CHUNKS:
        truncado = True
        metodo_chunks = [c for c in metodo_chunks if c["project"] in RTGS_PROJECTS]
        todos = clase_chunks + metodo_chunks
        if len(todos) > MAX_CODE_CHUNKS:
            todos = todos[:MAX_CODE_CHUNKS]

    # Recalcular chunks por raíz tras posible recorte.
    conteo = defaultdict(int)
    for c in todos:
        # map project → canonical display: find matching root
        for name in CODE_ROOTS:
            if name.lower() == c["project"]:
                conteo[name] += 1
                break
    for name, entry in por_raiz.items():
        if entry.get("exists"):
            entry["chunks_indexed"] = conteo.get(name, 0)

    summary = {
        "demo_ats": str(base),
        "demo_ats_exists": base.exists(),
        "roots": por_raiz,
        "files_read": sum(e.get("files", 0) for e in por_raiz.values()),
        "files_omitted": omitidos_global,
        "chunks": len(todos),
        "chunks_clase_o_archivo": len(clase_chunks),
        "chunks_metodo": sum(1 for c in todos if c.get("role") == "metodo"),
        "tope": MAX_CODE_CHUNKS,
        "truncated": truncado,
    }
    return todos, summary


def construir_catalogo(chunks: list[dict]) -> str:
    lineas = []
    for proyecto, frase in PROJECT_PURPOSE.items():
        if frase:
            lineas.append(f"{proyecto}: {frase}")
        else:
            lineas.append(
                f"{proyecto}: propósito no documentado; el índice sí puede contener sus archivos."
            )
    paquetes: dict[str, int] = {}
    for c in chunks:
        if c.get("project") == "core_rtgs" and c.get("package"):
            paquetes[c["package"]] = paquetes.get(c["package"], 0) + 1
    for pkg, n in sorted(paquetes.items(), key=lambda kv: -kv[1]):
        lineas.append(f"paquete {pkg}: {n} fragmentos indexados en core_rtgs.")
    texto = "\n".join(lineas)
    while contar_tokens(texto) > CATALOG_TOKEN_BUDGET and "\n" in texto:
        texto = texto.rsplit("\n", 1)[0]
    CATALOG_PATH.write_text(texto + "\n", encoding="utf-8")
    return texto


def _puntos_catalogo(catalogo: str) -> list[dict]:
    puntos = []
    for linea in catalogo.splitlines():
        if ":" not in linea:
            continue
        nombre = linea.split(":", 1)[0].replace("paquete ", "").strip()
        proyecto = nombre if nombre in PROJECT_PURPOSE else "core_rtgs"
        puntos.append(
            {
                "chunk_key": f"catalogo::{linea[:80]}",
                "doc_id": f"catalogo:{nombre}",
                "path": f"catalogo/{nombre}",
                "project": proyecto if nombre in PROJECT_PURPOSE else "core_rtgs",
                "package": "" if nombre in PROJECT_PURPOSE else nombre,
                "class": "",
                "method": "",
                "role": "catalogo",
                "text": linea,
                "source_type": "catalogo",
                "kb_status": "CATALOGO",
                "language": "es",
            }
        )
    return puntos


def indexar_codigo(
    demo: Path | None = None,
    qdrant_url: str | None = None,
    recreate: bool = True,
) -> dict:
    """Construye chunks + catálogo CAG y los sube a Qdrant COLLECTION_CODE."""
    OUT.mkdir(parents=True, exist_ok=True)
    chunks, summary = construir_chunks(demo=demo)
    if summary["files_read"] == 0:
        summary["warning"] = (
            f"Ningún archivo indexable bajo {summary['demo_ats']}. "
            "Revisa DEMO_ATS (p. ej. ../demo_ats) y que existan core*, CORE*."
        )
        INGESTA_PATH.write_text(
            json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        return summary

    catalogo = construir_catalogo(chunks)
    puntos = chunks + _puntos_catalogo(catalogo)

    with CHUNKS_PATH.open("w", encoding="utf-8") as fh:
        for c in puntos:
            fh.write(json.dumps(c, ensure_ascii=False) + "\n")

    url = qdrant_url or QDRANT_URL
    try:
        if recreate:
            recrear(COLLECTION_CODE, url=url)
        lote = CODE_EMBED_BATCH
        for i in range(0, len(puntos), lote):
            grupo = puntos[i : i + lote]
            vectores = embed([g["text"] for g in grupo], batch_size=lote)
            payloads = [{k: v for k, v in g.items() if k != "chunk_key"} for g in grupo]
            upsert(
                COLLECTION_CODE,
                [g["chunk_key"] for g in grupo],
                vectores,
                payloads,
                lote=64,
                url=url,
            )
            if i % max(lote * 2, 1) == 0 or i + lote >= len(puntos):
                print(
                    f"  indexados {min(i + lote, len(puntos))}/{len(puntos)}",
                    flush=True,
                )
    except EmbeddingsUnavailable:
        raise
    except Exception as exc:
        raise RuntimeError(
            f"Fallo al escribir en Qdrant ({url} / {COLLECTION_CODE}): {exc}\n"
            "En el Mac: docker run -p 6333:6333 -p 6334:6334 qdrant/qdrant"
        ) from exc

    summary.update(
        {
            "collection": COLLECTION_CODE,
            "qdrant_url": url,
            "points_upserted": len(puntos),
            "catalog_lines": len([ln for ln in catalogo.splitlines() if ln.strip()]),
            "catalog_tokens_approx": contar_tokens(catalogo),
            "embedding_backend": backend_embeddings(),
            "chunks_path": str(CHUNKS_PATH),
            "catalog_path": str(CATALOG_PATH),
        }
    )
    INGESTA_PATH.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary


def imprimir_resumen(summary: dict) -> None:
    print(f"DEMO_ATS={summary.get('demo_ats')} exists={summary.get('demo_ats_exists')}")
    print(f"colección={summary.get('collection', COLLECTION_CODE)} url={summary.get('qdrant_url', QDRANT_URL)}")
    roots = summary.get("roots") or {}
    for name in CODE_ROOTS:
        e = roots.get(name) or {}
        if e.get("skipped") or not e.get("exists"):
            print(f"  {name}: SKIP (missing)")
            continue
        resolved = e.get("resolved") or ""
        files = e.get("files", 0)
        chunks = e.get("chunks_indexed", e.get("chunks", 0))
        print(f"  {name}: files={files} chunks={chunks} path={resolved}")
    print(
        f"total files={summary.get('files_read', 0)} "
        f"chunks={summary.get('chunks', 0)} "
        f"points={summary.get('points_upserted', 0)} "
        f"embed={summary.get('embedding_backend', '?')}"
    )
    if summary.get("warning"):
        print(f"WARNING: {summary['warning']}")
