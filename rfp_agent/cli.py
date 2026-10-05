"""CLI: load-db, index-code, index-docs, run, eval-rfp."""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

from rfp_agent.config import FORBIDDEN_GOLDEN_IDS, GOLDEN_IDS, ROOT
from rfp_agent.db import golden_snapshot, load_rfp
from rfp_agent.evaluate import eval_rfp
from rfp_agent.llm import backend_usado, get_llm
from rfp_agent.workflow import RfpWorkflow


def _git_branch() -> str:
    try:
        r = subprocess.run(
            ["git", "rev-parse", "--abbrev-ref", "HEAD"],
            cwd=str(ROOT),
            capture_output=True,
            text=True,
            check=False,
        )
        if r.returncode == 0:
            return (r.stdout or "").strip() or "unknown"
    except Exception:
        pass
    return "unknown"


def _git_head() -> str:
    try:
        r = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=str(ROOT),
            capture_output=True,
            text=True,
            check=False,
        )
        if r.returncode == 0:
            return (r.stdout or "").strip() or "?"
    except Exception:
        pass
    return "?"


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="rfp-agent")
    sub = p.add_subparsers(dest="cmd", required=True)

    sub.add_parser("load-db", help="Carga las 174 filas de data/rfp.xlsx en SQLite")
    sub.add_parser("tools", help="Lista el catálogo MCP descubierto")

    ic = sub.add_parser(
        "index-code",
        help="Indexa Java/XML/properties bajo DEMO_ATS → Qdrant montran_code (CAG)",
    )
    ic.add_argument(
        "--demo-ats",
        default=None,
        help="Override de DEMO_ATS (default: env / ./demo_ats)",
    )
    ic.add_argument(
        "--qdrant-url",
        default=None,
        help="Override de QDRANT_URL (default: http://localhost:6333)",
    )
    ic.add_argument(
        "--no-recreate",
        action="store_true",
        help="No borra la colección antes de upsert (por defecto se recrea)",
    )

    idocs = sub.add_parser(
        "index-docs",
        help="Indexa PDF/DOCX/MD bajo corpus/ (+ corpus/docs/) → Qdrant documentos",
    )
    idocs.add_argument(
        "--corpus",
        default=None,
        help="Override del directorio corpus (default: ./corpus)",
    )
    idocs.add_argument(
        "--qdrant-url",
        default=None,
        help="Override de QDRANT_URL (default: http://localhost:6333)",
    )
    idocs.add_argument(
        "--no-recreate",
        action="store_true",
        help="No borra la colección antes de upsert (por defecto se recrea)",
    )

    r = sub.add_parser("run", help="Corre el grafo sobre un R-nnn")
    r.add_argument("requirement_id")

    e = sub.add_parser("eval-rfp", help=f"Evalúa solo el golden ({', '.join(GOLDEN_IDS)})")
    e.add_argument("--ids", nargs="*", default=list(GOLDEN_IDS))

    args = p.parse_args(argv)
    if args.cmd == "load-db":
        info = load_rfp()
        snap = golden_snapshot()
        payload = {
            **info,
            "git_branch": _git_branch(),
            "git_head": _git_head(),
            "golden_count": snap["golden_count"],
            "golden_ids": snap["golden_ids"],
            "expected_golden": list(GOLDEN_IDS),
            "forbidden_stripped_ok": not any(
                x in FORBIDDEN_GOLDEN_IDS for x in snap["golden_ids"]
            ),
            "repo_root": str(ROOT),
            "db_exists": Path(info["db"]).exists(),
        }
        print(json.dumps(payload, ensure_ascii=False, indent=2))
        if snap["golden_count"] != len(GOLDEN_IDS) or set(snap["golden_ids"]) != set(GOLDEN_IDS):
            print(
                f"ERROR: golden debe ser exactamente {list(GOLDEN_IDS)}. "
                f"Ahora: {snap['golden_ids']}.",
                file=sys.stderr,
            )
            return 1
        return 0
    if args.cmd == "tools":
        wf = RfpWorkflow()
        print(
            json.dumps(
                {
                    "llm": backend_usado(),
                    "tools": wf.cliente.descubrir(),
                    "git_branch": _git_branch(),
                    "golden": golden_snapshot(),
                },
                indent=2,
            )
        )
        return 0
    if args.cmd == "index-code":
        from rfp_agent.code_pipeline import imprimir_resumen, indexar_codigo
        from rfp_agent.embedder import EmbeddingsUnavailable

        demo = Path(args.demo_ats).expanduser() if args.demo_ats else None
        if demo is not None and not demo.is_absolute():
            demo = (ROOT / demo).resolve()
        try:
            summary = indexar_codigo(
                demo=demo,
                qdrant_url=args.qdrant_url,
                recreate=not args.no_recreate,
            )
        except EmbeddingsUnavailable as exc:
            print(str(exc), file=sys.stderr)
            return 1
        except RuntimeError as exc:
            print(str(exc), file=sys.stderr)
            return 1
        imprimir_resumen(summary)
        print(json.dumps(summary, ensure_ascii=False, indent=2, default=str))
        if summary.get("files_read", 0) == 0:
            return 1
        return 0
    if args.cmd == "index-docs":
        from rfp_agent.docs_pipeline import imprimir_resumen as imprimir_docs
        from rfp_agent.docs_pipeline import indexar_documentos
        from rfp_agent.embedder import EmbeddingsUnavailable

        corpus = Path(args.corpus).expanduser() if args.corpus else None
        if corpus is not None and not corpus.is_absolute():
            corpus = (ROOT / corpus).resolve()
        try:
            summary = indexar_documentos(
                corpus=corpus,
                qdrant_url=args.qdrant_url,
                recreate=not args.no_recreate,
            )
        except EmbeddingsUnavailable as exc:
            print(str(exc), file=sys.stderr)
            return 1
        except RuntimeError as exc:
            print(str(exc), file=sys.stderr)
            return 1
        imprimir_docs(summary)
        print(json.dumps(summary, ensure_ascii=False, indent=2, default=str))
        if summary.get("files_indexed", 0) == 0 or summary.get("chunks", 0) == 0:
            return 1
        return 0
    if args.cmd == "run":
        get_llm()
        wf = RfpWorkflow()
        paquete = wf.run(args.requirement_id)
        print(json.dumps(paquete, ensure_ascii=False, indent=2, default=str))
        return 0
    if args.cmd == "eval-rfp":
        # Ignorar R-018 aunque el usuario lo pase en --ids.
        ids = [i for i in (args.ids or list(GOLDEN_IDS)) if i not in FORBIDDEN_GOLDEN_IDS]
        if not ids:
            ids = list(GOLDEN_IDS)
        resumen = eval_rfp(ids=ids)
        resumen["git_branch"] = _git_branch()
        resumen["git_head"] = _git_head()
        resumen["golden_snapshot"] = golden_snapshot()
        print(json.dumps(resumen, ensure_ascii=False, indent=2, default=str))
        return 0 if resumen["failed"] == 0 else 1
    return 2


if __name__ == "__main__":
    sys.exit(main())
