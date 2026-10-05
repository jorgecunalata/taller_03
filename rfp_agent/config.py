"""Rutas y constantes. CODE_ROOTS apuntan a ./demo_ats por defecto (Mac: junto a este repo)."""
from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
load_dotenv(ROOT / ".env")

DATA = ROOT / "data"
CORPUS = ROOT / "corpus"
OUT = ROOT / "outputs"
OUT.mkdir(parents=True, exist_ok=True)

RFP_XLSX = Path(os.environ.get("RFP_XLSX", DATA / "rfp.xlsx"))
RFP_DB = Path(os.environ.get("RFP_DB", DATA / "rfp.sqlite"))
RUNS_DB = Path(os.environ.get("RUNS_DB", DATA / "runs.sqlite"))

DEMO_ATS = Path(os.environ.get("DEMO_ATS", "./demo_ats")).expanduser()
if not DEMO_ATS.is_absolute():
    DEMO_ATS = (ROOT / DEMO_ATS).resolve()

# Subproyectos Java en el Mac. El primer slice no exige que existan.
CODE_ROOTS = [
    "CORE",
    "CORE_BILL",
    "CORE_RTGS",
    "CORE_PO_ATS",
    "CORE_PARSER",
    "CORE_MS",
    "CORE_ACH",
    "CORE_ATS",
]

QDRANT_URL = os.environ.get("QDRANT_URL", "http://localhost:6333")
COLLECTION_DOCS = os.environ.get("COLLECTION_DOCS", "documentos")
COLLECTION_CODE = os.environ.get("COLLECTION_CODE", "montran_code")

H200_HOST = os.environ.get("H200_HOST", "172.28.230.10")
H200_LLM_PORT = int(os.environ.get("H200_LLM_PORT", "12555"))
H200_EMBED_PORT = int(os.environ.get("H200_EMBED_PORT", "11434"))
H200_API_KEY = os.environ.get("H200_API_KEY", "local")
LLM_BACKEND = os.environ.get("LLM_BACKEND", "auto").lower()
# Thinking + tools en vLLM suele quemar MAX_PASOS sin hand-off. Off por defecto.
H200_ENABLE_THINKING = os.environ.get("H200_ENABLE_THINKING", "0").strip() in {
    "1",
    "true",
    "yes",
    "on",
}

MAX_PASOS = int(os.environ.get("MAX_PASOS", "20"))
MAX_REPLAN = int(os.environ.get("MAX_REPLAN", "2"))
TOKEN_BUDGET = int(os.environ.get("TOKEN_BUDGET", "80000"))
QUERY_FACTS_ROW_LIMIT = int(os.environ.get("QUERY_FACTS_ROW_LIMIT", "50"))

# Reader: 1 ronda retrieve basta; el grafo cierra el rol post-tools (no espera al LLM).
TOOL_BUDGET_POR_ROL = {
    "planner": int(os.environ.get("TOOL_BUDGET_PLANNER", "2")),
    "reader": int(os.environ.get("TOOL_BUDGET_READER", "1")),
    "writer": int(os.environ.get("TOOL_BUDGET_WRITER", "1")),
    "verifier": int(os.environ.get("TOOL_BUDGET_VERIFIER", "2")),
    "synthesizer": int(os.environ.get("TOOL_BUDGET_SYNTHESIZER", "0")),
}

ABSTENCION = "El corpus no contiene información suficiente."

# Primer slice: SOLO estos tres. R-018 y otros quedan fuera aunque alguien los pegue en db.py.
GOLDEN_IDS = ("R-001", "R-038", "R-057")
FORBIDDEN_GOLDEN_IDS = frozenset({"R-018"})

EMBED_OLLAMA_NAME = "bge-m3"
EMBED_DIM = 1024
EMBED_MAX_TOKENS = 8192
