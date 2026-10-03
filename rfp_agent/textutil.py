"""Normalización compartida con el evaluador (misma idea que Taller 2)."""
from __future__ import annotations

import re
import unicodedata


def normalizar(texto: str) -> str:
    texto = unicodedata.normalize("NFKD", texto or "")
    texto = "".join(c for c in texto if not unicodedata.combining(c))
    texto = texto.lower()
    texto = re.sub(r"\s+", " ", texto)
    return texto.strip()


def se_abstuvo(respuesta: str, frase: str) -> bool:
    return normalizar(frase) in normalizar(respuesta)
