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

# Nombres canónicos (mayúsculas). En el Mac el disco suele ser lowercase: se resuelven
# sin distinguir mayúsculas. CORE_MS / core_ms suele faltar → se omite al indexar.
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

# Extras bajo demo_ats que no indexamos por defecto (anidados / demos / runtime).
CODE_ROOTS_SKIP_DEFAULT = frozenset(
    {
        "demo_ats",
        "demo_ats_release",
        "demo_po",
        "core_runtime",
        "CORE_RUNTIME",
    }
)

CODE_EXTS = {".java", ".xml", ".jrxml", ".properties", ".yml", ".yaml", ".json"}

SKIP_DIR_NAMES = {
    "test",
    "tests",
    "bin",
    "build",
    "target",
    ".git",
    ".metadata",
    "node_modules",
    ".gradle",
    "out",
    "dist",
    ".settings",
    "__pycache__",
    ".idea",
    "coverage",
}

# Catálogo CAG (ideas Taller 2). Claves en lowercase para casar con el disco Mac.
PROJECT_PURPOSE = {
    "core_rtgs": "Núcleo RTGS: liquidación bruta en tiempo real.",
    "core_parser": "Parsers de mensajería ISO 20022, SWIFT, UNIFI, ACH y RTGS.",
    "core": "Framework base: seguridad, UI, integración y piezas comunes. No es un producto por sí mismo.",
    "core_bill": "Facturación y comisiones.",
    "core_ats": "Componentes compartidos de ATS (RTGS+ACH).",
    "core_ach": "Núcleo ACH: compensación, clearing, cheques y sesiones.",
    "core_po_ats": "Núcleo del Payment Originator.",
    "core_ms": "Servicios / microservicios (si el árbol lo trae).",
}

# Proyectos RTGS prioritarios si hay que recortar métodos por tope de chunks.
RTGS_PROJECTS = frozenset(
    {"core_rtgs", "core_parser", "core", "core_bill", "core_ats", "core_ach"}
)

CATALOG_TOKEN_BUDGET = 1500
MAX_CODE_CHUNKS = int(os.environ.get("MAX_CODE_CHUNKS", "25000"))
CODE_EMBED_BATCH = int(os.environ.get("CODE_EMBED_BATCH", "64"))

# Documentos (Taller 2: ~512/102). Aprox. por palabras si no hay tokenizer.
CHUNK_TOKENS = int(os.environ.get("CHUNK_TOKENS", "512"))
OVERLAP_TOKENS = int(os.environ.get("OVERLAP_TOKENS", "102"))
DOC_EMBED_BATCH = int(os.environ.get("DOC_EMBED_BATCH", "8"))
PISO_CARACTERES_DOC = int(os.environ.get("PISO_CARACTERES_DOC", "200"))
DOC_EXTS = {".pdf", ".docx", ".md", ".txt"}

QDRANT_URL = os.environ.get("QDRANT_URL", "http://localhost:6333")
COLLECTION_DOCS = os.environ.get("COLLECTION_DOCS", "documentos")
COLLECTION_CODE = os.environ.get("COLLECTION_CODE", "montran_code")

H200_HOST = os.environ.get("H200_HOST", "172.28.230.10")
H200_LLM_PORT = int(os.environ.get("H200_LLM_PORT", "12555"))
H200_EMBED_PORT = int(os.environ.get("H200_EMBED_PORT", "11434"))
H200_API_KEY = os.environ.get("H200_API_KEY", "local")
LLM_BACKEND = os.environ.get("LLM_BACKEND", "auto").lower()
# Embeddings CAG: auto (H200 → local ST si está instalado) | h200 | local
EMBEDDING_BACKEND = os.environ.get("EMBEDDING_BACKEND", "auto").lower()
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
EMBED_MODEL = "BAAI/bge-m3"
EMBED_DIM = 1024
EMBED_MAX_TOKENS = 8192


def resolve_child(parent: Path, name: str) -> Path | None:
    """Resuelve un hijo por nombre sin distinguir mayúsculas (APFS/HFS)."""
    if not parent.exists():
        return None
    direct = parent / name
    if direct.exists():
        return direct
    target = name.lower()
    try:
        for child in parent.iterdir():
            if child.name.lower() == target:
                return child
    except OSError:
        return None
    return None


def resolve_code_root(name: str) -> Path | None:
    """Ruta real bajo DEMO_ATS para un CODE_ROOT canónico, o None si falta."""
    return resolve_child(DEMO_ATS, name)
