"""CLI: load-db, run, eval-rfp."""
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

    r = sub.add_parser("run", help="Corre el grafo sobre un R-nnn")
    r.add_argument("requirement_id")

    e = sub.add_parser("eval-rfp", help="Evalúa solo el golden (R-001, R-038, R-057)")
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
        if snap["golden_count"] != 3 or set(snap["golden_ids"]) != set(GOLDEN_IDS):
            print(
                "ERROR: golden debe ser exactamente R-001, R-038, R-057. "
                f"Ahora: {snap['golden_ids']}. ¿Estás en cursor/h200-eval-rfp-fix-59e8?",
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
