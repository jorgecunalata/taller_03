"""CLI: load-db, run, eval-rfp."""
from __future__ import annotations

import argparse
import json
import sys

from rfp_agent.config import GOLDEN_IDS
from rfp_agent.db import load_rfp
from rfp_agent.evaluate import eval_rfp
from rfp_agent.llm import backend_usado, get_llm
from rfp_agent.workflow import RfpWorkflow


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
        print(json.dumps(load_rfp(), ensure_ascii=False, indent=2))
        return 0
    if args.cmd == "tools":
        wf = RfpWorkflow()
        print(json.dumps({"llm": backend_usado(), "tools": wf.cliente.descubrir()}, indent=2))
        return 0
    if args.cmd == "run":
        get_llm()
        wf = RfpWorkflow()
        paquete = wf.run(args.requirement_id)
        print(json.dumps(paquete, ensure_ascii=False, indent=2, default=str))
        return 0
    if args.cmd == "eval-rfp":
        resumen = eval_rfp(ids=args.ids)
        print(json.dumps(resumen, ensure_ascii=False, indent=2, default=str))
        return 0 if resumen["failed"] == 0 else 1
    return 2


if __name__ == "__main__":
    sys.exit(main())
