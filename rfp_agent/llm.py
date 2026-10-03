"""Cliente H200 (vLLM :12555, bge-m3 :11434) y LLM simulado sin VPN."""
from __future__ import annotations

import json
import time
import urllib.error
import urllib.request
from typing import Any

from rfp_agent.config import (
    ABSTENCION,
    H200_API_KEY,
    H200_EMBED_PORT,
    H200_HOST,
    H200_LLM_PORT,
    LLM_BACKEND,
)

TOOLS_POR_ROL = {
    "planner": ["get_requirement", "list_sources", "query_facts"],
    "reader": ["retrieve_knowledge", "query_facts", "list_sources"],
    "writer": ["record_evidence"],
    "verifier": ["query_facts", "retrieve_knowledge"],
    "synthesizer": ["query_facts"],
}


def _http_json(url: str, body: dict | None = None, timeout: float = 180) -> dict:
    data = None if body is None else json.dumps(body).encode()
    req = urllib.request.Request(
        url,
        data=data,
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {H200_API_KEY}",
        },
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read())


class H200:
    def __init__(self, timeout: float = 180):
        self.timeout = timeout
        self.host = H200_HOST
        self.modelo = self._get(H200_LLM_PORT, "v1/models")["data"][0]["id"]
        try:
            ids = [m["id"] for m in self._get(H200_EMBED_PORT, "v1/models")["data"]]
            self.modelo_emb = next((m for m in ids if str(m).startswith("bge-m3")), None)
        except Exception:
            self.modelo_emb = "bge-m3"

    def _pedir(self, puerto: int, ruta: str, cuerpo: dict | None = None, timeout: float | None = None) -> dict:
        url = f"http://{self.host}:{puerto}/{ruta}"
        try:
            return _http_json(url, cuerpo, timeout=timeout or self.timeout)
        except urllib.error.URLError as err:
            raise RuntimeError(
                f"Sin respuesta de {self.host}:{puerto} ({err}). ¿Está GlobalProtect conectada?"
            ) from err

    def _get(self, puerto: int, ruta: str) -> dict:
        return self._pedir(puerto, ruta, timeout=8)

    def chat(
        self,
        mensajes: list[dict],
        tools: list[dict] | None = None,
        json_mode: bool = False,
        max_tokens: int = 4096,
        role: str | None = None,
    ) -> dict:
        cuerpo: dict[str, Any] = {
            "model": self.modelo,
            "messages": mensajes,
            "max_tokens": max_tokens,
            "chat_template_kwargs": {"enable_thinking": True},
        }
        if tools:
            cuerpo["tools"] = tools
        if json_mode:
            cuerpo["response_format"] = {"type": "json_object"}
        t0 = time.perf_counter()
        try:
            cuerpo["temperature"] = 0
            r = self._pedir(H200_LLM_PORT, "v1/chat/completions", cuerpo)
        except Exception as e:
            # Familia terra (y similares) puede rechazar temperature.
            if "temperature" in str(e).lower() or True:
                cuerpo.pop("temperature", None)
                try:
                    r = self._pedir(H200_LLM_PORT, "v1/chat/completions", cuerpo)
                except Exception:
                    raise e from None
            else:
                raise
        return {
            "mensaje": r["choices"][0]["message"],
            "uso": r.get("usage", {}),
            "latencia_s": round(time.perf_counter() - t0, 2),
        }

    def embeddings(self, textos: list[str]):
        import numpy as np

        r = self._pedir(
            H200_EMBED_PORT,
            "v1/embeddings",
            {"model": self.modelo_emb, "input": textos},
        )
        X = np.asarray(
            [d["embedding"] for d in sorted(r["data"], key=lambda d: d["index"])],
            dtype="float32",
        )
        normas = np.linalg.norm(X, axis=1, keepdims=True)
        return X / np.clip(normas, 1e-12, None)


class SimulatedLLM:
    """Recorre el mismo contrato de chat/tools que la H200. Sin red."""

    def __init__(self):
        self.modelo = "simulated"
        self.modelo_emb = None
        self._n = 0

    def _id(self) -> str:
        self._n += 1
        return f"sim-{self._n}"

    def chat(
        self,
        mensajes: list[dict],
        tools: list[dict] | None = None,
        json_mode: bool = False,
        max_tokens: int = 4096,
        role: str | None = None,
    ) -> dict:
        t0 = time.perf_counter()
        role = role or _infer_role(mensajes)
        names = [t["function"]["name"] for t in (tools or [])]
        msg = self._act(role, mensajes, names)
        return {
            "mensaje": msg,
            "uso": {"prompt_tokens": 32, "completion_tokens": 48, "total_tokens": 80},
            "latencia_s": round(time.perf_counter() - t0, 4),
        }

    def _tool(self, name: str, args: dict) -> dict:
        return {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "id": self._id(),
                    "type": "function",
                    "function": {"name": name, "arguments": json.dumps(args, ensure_ascii=False)},
                }
            ],
        }

    def _act(self, role: str, mensajes: list[dict], names: list[str]) -> dict:
        user = _last_user(mensajes)
        obs = _tool_obs(mensajes)
        req_id = _find_req_id(mensajes, user)
        texto = _find_req_text(obs, user)

        if role == "planner":
            if "get_requirement" in names and not any(o.get("id") == req_id for o in obs if isinstance(o, dict)):
                return self._tool("get_requirement", {"requirement_id": req_id})
            if "list_sources" in names and not any("qdrant" in json.dumps(o) for o in obs):
                return self._tool("list_sources", {})
            plan = {
                "ruta": "docs",
                "plan": [
                    {"paso": "leer_docs", "objetivo": "Recuperar evidencia del PDF funcional"},
                    {"paso": "verificar_sql", "objetivo": f"Contrastar facts WHERE req_id = '{req_id}'"},
                ],
                "sql_checks": [
                    f"SELECT statement, source_doc, locator FROM facts WHERE req_id = '{req_id}'"
                ],
                "restricciones": [
                    "No inventar TPS ni cifras",
                    "Ignorar [[RELLENAR]] y EJEMPLO_NO_VALIDADO",
                    "kb_status debe viajar en la evidencia",
                ],
            }
            return {"role": "assistant", "content": json.dumps(plan, ensure_ascii=False)}

        if role == "reader":
            if "retrieve_knowledge" in names and not any(
                isinstance(o, dict) and o.get("hits") is not None for o in obs
            ):
                q = texto or user
                scope = "docs"
                if any(w in (texto or user).lower() for w in ("clase", "java", "método", "core_")):
                    scope = "both"
                return self._tool("retrieve_knowledge", {"query": q, "scope": scope, "k": 5})
            hits = []
            for o in obs:
                if isinstance(o, dict) and o.get("hits"):
                    hits = o["hits"]
            return {
                "role": "assistant",
                "content": json.dumps(
                    {"handoff": "writer", "n_hits": len(hits), "objetivo": "Redactar solo con evidence[]"},
                    ensure_ascii=False,
                ),
            }

        if role == "writer":
            evidence = _parse_evidence_from_user(user)
            recorded_keys = set()
            for m in mensajes:
                for tc in m.get("tool_calls") or []:
                    if (tc.get("function") or {}).get("name") != "record_evidence":
                        continue
                    raw = (tc.get("function") or {}).get("arguments") or "{}"
                    try:
                        recorded_keys.add(json.loads(raw).get("chunk_key"))
                    except json.JSONDecodeError:
                        continue
                if m.get("role") == "tool":
                    try:
                        body = json.loads(m.get("content") or "{}")
                    except json.JSONDecodeError:
                        body = {}
                    if isinstance(body, dict) and body.get("chunk_key"):
                        recorded_keys.add(body["chunk_key"])
            if "record_evidence" in names:
                pending = [
                    ev for ev in evidence
                    if ev.get("chunk_key") and ev["chunk_key"] not in recorded_keys
                ]
                if pending:
                    calls = []
                    for ev in pending:
                        calls.append(
                            {
                                "id": self._id(),
                                "type": "function",
                                "function": {
                                    "name": "record_evidence",
                                    "arguments": json.dumps(
                                        {
                                            "requirement_id": req_id,
                                            "chunk_key": ev["chunk_key"],
                                            "cita": (ev.get("text") or "")[:180],
                                        },
                                        ensure_ascii=False,
                                    ),
                                },
                            }
                        )
                    return {"role": "assistant", "content": "", "tool_calls": calls}
            draft = _draft_from_evidence(req_id, texto, evidence)
            return {"role": "assistant", "content": json.dumps(draft, ensure_ascii=False)}

        if role == "verifier":
            if "query_facts" in names and not any(isinstance(o, dict) and "rows" in o for o in obs):
                return self._tool(
                    "query_facts",
                    {"sql": f"SELECT statement, source_doc, locator FROM facts WHERE req_id = '{req_id}'"},
                )
            rows = []
            for o in obs:
                if isinstance(o, dict) and "rows" in o:
                    rows = o["rows"]
            verdict = _verdict(req_id, user, rows)
            return {"role": "assistant", "content": json.dumps(verdict, ensure_ascii=False)}

        if role == "synthesizer":
            packed = _pack(user)
            return {"role": "assistant", "content": json.dumps(packed, ensure_ascii=False)}

        return {"role": "assistant", "content": "{}"}


def _infer_role(mensajes: list[dict]) -> str:
    sys = ""
    for m in mensajes:
        if m.get("role") == "system":
            sys = m.get("content") or ""
            break
    for rol, marca in (
        ("planner", "Planner"),
        ("reader", "Reader"),
        ("writer", "Writer"),
        ("verifier", "Verifier"),
        ("synthesizer", "Output Synthesizer"),
    ):
        if marca in sys:
            return rol
    return "planner"


def _last_user(mensajes: list[dict]) -> str:
    for m in reversed(mensajes):
        if m.get("role") == "user":
            return m.get("content") or ""
    return ""


def _tool_obs(mensajes: list[dict]) -> list[Any]:
    out = []
    for m in mensajes:
        if m.get("role") == "tool":
            try:
                out.append(json.loads(m.get("content") or "{}"))
            except json.JSONDecodeError:
                out.append({"raw": m.get("content")})
    return out


def _find_req_id(mensajes: list[dict], user: str) -> str:
    blob = user + json.dumps(mensajes, ensure_ascii=False)
    m = __import__("re").search(r"R-\d{3}", blob)
    return m.group(0) if m else "R-001"


def _find_req_text(obs: list[Any], user: str) -> str:
    for o in obs:
        if isinstance(o, dict) and o.get("texto"):
            return o["texto"]
    if "texto:" in user:
        return user.split("texto:", 1)[-1].strip()
    return user


def _parse_evidence_from_user(user: str) -> list[dict]:
    try:
        if "evidence=" in user:
            raw = user.split("evidence=", 1)[1]
            # hasta el próximo bloque marcado
            if "\n---" in raw:
                raw = raw.split("\n---", 1)[0]
            data = json.loads(raw)
            if isinstance(data, list):
                return data
    except json.JSONDecodeError:
        pass
    return []


def _draft_from_evidence(req_id: str, texto: str, evidence: list[dict]) -> dict:
    citations = []
    piezas = []
    for i, ev in enumerate(evidence, start=1):
        citations.append(
            {
                "n": i,
                "chunk_key": ev.get("chunk_key"),
                "source": ev.get("source"),
                "locator": ev.get("locator"),
                "kb_status": ev.get("kb_status"),
                "text": ev.get("text"),
            }
        )
        piezas.append(f"[{i}] {ev.get('text')}")
    joined = " ".join(e.get("text") or "" for e in evidence)
    pide_tps = any(w in (texto or "").lower() for w in ("tps", "capacidad máxima", "capacidad maxima"))
    hay_tps = "tps" in joined.lower() and any(ch.isdigit() for ch in joined)
    if pide_tps and not hay_tps:
        return {
            "draft": ABSTENCION + " El PDF funcional no publica un máximo transaccional (TPS).",
            "citations": citations,
            "abstain": True,
        }
    if req_id == "R-001":
        cuerpo = (
            "La solución garantiza liquidación de fondos final e irrevocable en tiempo real "
            "entre los miembros del servicio RTGS, de forma continua [1]."
        )
    elif req_id == "R-038":
        cuerpo = (
            "El sistema RTGS de Montran se ha mejorado con capacidades para funcionar 24/7 [1]. "
            "El RTGS de Montran puede funcionar y procesar pagos de forma 24/7/365 [2]."
        )
    else:
        cuerpo = ("Con la evidencia recuperada: " + " ".join(piezas[:3])).strip()
        if not cuerpo:
            cuerpo = ABSTENCION
    return {"draft": cuerpo, "citations": citations, "abstain": False}


def _verdict(req_id: str, user: str, rows: list[dict]) -> dict:
    draft = ""
    if "draft=" in user:
        draft = user.split("draft=", 1)[1]
        if "\n---" in draft:
            draft = draft.split("\n---", 1)[0]
    low = (draft or "").lower()
    if "[[rellenar]]" in low or "ejemplo_no_validado" in low:
        return {"verdict": "fail", "critique": "El draft cita plantilla no validada.", "sql_used": rows}
    if req_id == "R-057":
        ok = ABSTENCION.lower() in low
        inventa = ("tps" in low and any(c.isdigit() for c in low))
        if ok and not inventa:
            return {"verdict": "pass", "critique": "", "sql_used": rows}
        return {
            "verdict": "fail",
            "critique": "R-057 exige abstención: el PDF no da TPS.",
            "sql_used": rows,
        }
    missing = []
    for row in rows:
        st = (row.get("statement") or "").lower()
        clave = st[:40]
        if clave and clave not in low and not all(w in low for w in st.split()[:4]):
            # cobertura laxa: palabras distintivas
            tokens = [w for w in st.split() if len(w) > 4][:3]
            if tokens and not all(t in low for t in tokens):
                missing.append(row.get("statement"))
    if missing:
        return {
            "verdict": "fail",
            "critique": "Faltan hechos SQL en el draft: " + "; ".join(missing),
            "sql_used": rows,
        }
    return {"verdict": "pass", "critique": "", "sql_used": rows}


def _pack(user: str) -> dict:
    # El synthesizer recibe el estado serializado.
    status = "completed"
    if "verdict=fail" in user:
        status = "rejected_insufficient_evidence"
    answer = ""
    citations: list = []
    if "draft=" in user:
        chunk = user.split("draft=", 1)[1]
        if "\n---" in chunk:
            chunk = chunk.split("\n---", 1)[0]
        try:
            data = json.loads(chunk)
            answer = data.get("draft") or chunk
            citations = data.get("citations") or []
        except json.JSONDecodeError:
            answer = chunk.strip()
    return {"answer": answer, "citations": citations, "status": status}


_LLM: H200 | SimulatedLLM | None = None
_BACKEND_USED: str | None = None


def get_llm() -> H200 | SimulatedLLM:
    global _LLM, _BACKEND_USED
    if _LLM is not None:
        return _LLM
    if LLM_BACKEND == "simulated":
        _LLM = SimulatedLLM()
        _BACKEND_USED = "simulated"
        return _LLM
    if LLM_BACKEND == "h200":
        _LLM = H200()
        _BACKEND_USED = "h200"
        return _LLM
    try:
        _LLM = H200()
        _BACKEND_USED = "h200"
        return _LLM
    except Exception:
        _LLM = SimulatedLLM()
        _BACKEND_USED = "simulated"
        return _LLM


def backend_usado() -> str:
    get_llm()
    return _BACKEND_USED or "unknown"


def embed_query(texto: str):
    llm = get_llm()
    if isinstance(llm, H200) and llm.modelo_emb:
        return llm.embeddings([texto])[0].tolist()
    return None


def reset_llm() -> None:
    global _LLM, _BACKEND_USED
    _LLM = None
    _BACKEND_USED = None
