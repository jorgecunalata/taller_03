"""Filtros de evidencia: plantillas no validadas y ruido de retrieve."""
from __future__ import annotations

import re

_PROHIBIDO = ("[[rellenar]]", "ejemplo_no_validado")
_CODEY = re.compile(
    r"(package\s+\w+|import\s+[\w.]+;|<\?xml|<bean\s|<property\s|public\s+class\s+|@Override)",
    re.I,
)


def es_plantilla_invalida(texto: str = "", kb_status: str = "", source: str = "") -> bool:
    blob = f"{texto} {kb_status} {source}".lower()
    if any(p in blob for p in _PROHIBIDO):
        return True
    status = (kb_status or "").upper()
    if status in {"EJEMPLO_NO_VALIDADO", "NO_VALIDADO", "PLANTILLA"}:
        return True
    return False


def es_ruido_retrieve(texto: str = "", kb_status: str = "", source: str = "", scope: str = "") -> bool:
    """Descarta plantillas y dumps de código/XML demasiado largos para el Writer.

    Con scope=code se conservan fragmentos Java/XML del CAG (no son ruido).
    Con scope=docs|both se siguen filtrando dumps largos para no ensuciar el golden RFP.
    """
    if es_plantilla_invalida(texto, kb_status, source):
        return True
    if (kb_status or "").upper() in {"ERROR", "AUSENTE"}:
        return True
    scope_l = (scope or "").lower()
    if scope_l == "code":
        return False
    t = texto or ""
    if len(t) > 900 and _CODEY.search(t):
        return True
    src = (source or "").lower()
    if src.endswith((".java", ".xml", ".properties")) and es_plantilla_invalida(t, kb_status, source):
        return True
    return False


def contiene_plantilla(texto: str) -> bool:
    low = (texto or "").lower()
    return any(p in low for p in _PROHIBIDO)
