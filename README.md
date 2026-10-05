# Taller 3 — respuesta automatizada de RFP (primer slice)

Grafo **LangGraph 1.2.12** con cinco agentes (Planner, Reader, Writer, Verifier, Output Synthesizer) y un servidor MCP simulado. El evaluador puntúa **solo los ítems golden** con SQL, no las 174 filas del Excel y no con un LLM-as-judge.

En el Mac del estudiante este repo vive en `taller_3/` junto a `demo_ats/`. El default `DEMO_ATS=./demo_ats` apunta ahí. **El primer slice no necesita el árbol Java** ni Qdrant: `retrieve_knowledge` usa un stub de facts + citas del PDF funcional.

## Qué hace este corte

- Carga `data/rfp.xlsx` → SQLite `rfp_items` con ids `R-001`…`R-174`.
- Golden (`in_golden=1`): `R-001`/`R-038` (PDF), `R-037` (Java/CAG settlement window, `CODIGO`) y `R-057` (abstención: no hay TPS).
- MCP: `retrieve_knowledge`, `get_requirement`, `query_facts`, `record_evidence` (+ `list_sources` para el criterio N+1).
- LLM: H200 `http://172.28.230.10:12555/v1` (chat) y `:11434` (`bge-m3`) si hay VPN; si no, el mismo grafo corre con un LLM simulado.

No hay UI, LangSmith, ni `create_agent` / `create_react_agent`.

## Cómo correrlo

Python ≥ 3.11.

```bash
python3 -m pip install -e ".[dev]"
cp .env.example .env
python3 -m rfp_agent.cli load-db
python3 -m rfp_agent.cli tools
python3 -m rfp_agent.cli run R-001
python3 -m rfp_agent.cli eval-rfp
# Índices Qdrant (embeddings H200 o local):
# python3 -m rfp_agent.cli index-docs   # corpus/ → documentos
# python3 -m rfp_agent.cli index-code   # demo_ats → montran_code
python3 -m pytest -q
```

Si tienes `venv`:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

`eval-rfp` escribe `outputs/eval_rfp.json` y sale con código 1 si algún golden falla.

### H200 (corrida calificable)

Con GlobalProtect: en `.env` deja `LLM_BACKEND=auto` (o `h200`). El cliente pregunta `/v1/models`; no hay un id de modelo fijado en el código. Si el modelo rechaza `temperature`, se reintenta sin ella.

Anti-thrash (importante en H200): `H200_ENABLE_THINKING=0`, presupuestos `TOOL_BUDGET_*` por rol, hand-off forzado cuando ya hay evidence/SQL, y draft determinista con literales del PDF/código si el modelo no emite JSON. Golden: `R-001`, `R-037`, `R-038`, `R-057` (no R-018; código como `CODIGO`, no `VALIDADO` automático).

Sin VPN: `LLM_BACKEND=simulated` (o `auto`, que cae solo). El grafo y el evaluador SQL siguen siendo los mismos.

### Documentos (`index-docs`)

Con Qdrant en `:6333` y embeddings H200 (o `EMBEDDING_BACKEND=local`):

```bash
python3 -m rfp_agent.cli index-docs
```

Indexa PDF (y `.docx` / `.md` / `.txt`) bajo `corpus/` y `corpus/docs/` hacia la colección Qdrant `documentos`. Chunking ~512/102 (Taller 2). El PDF funcional va como `kb_status=VALIDADO`; plantillas `.md` / `[[RELLENAR]]` / `EJEMPLO_NO_VALIDADO` **no** se promueven a VALIDADO (el retrieve las filtra). Omite `corpus/docs/` si no existe. Resumen: files / pages / chunks.

`retrieve_knowledge` con `scope=docs|both` usa esa colección cuando está presente.

### Código Java / CAG (`index-code`)

En el Mac, con `demo_ats` hermano (o symlink) y Qdrant en `:6333`:

```bash
# en .env: DEMO_ATS=../demo_ats   (o la ruta absoluta)
# Embeddings: H200 Ollama bge-m3 (VPN) o EMBEDDING_BACKEND=local
python3 -m rfp_agent.cli index-code
```

Indexa `.java` / `.xml` / `.properties` (y similares) bajo los `CODE_ROOTS` hacia la colección Qdrant `montran_code`. Omite raíces ausentes (p. ej. `CORE_MS`). Resuelve nombres sin distinguir mayúsculas (`core_rtgs` ≡ `CORE_RTGS`). No indexa por defecto `demo_ats_release`, `demo_po`, `core_runtime` ni el `demo_ats` anidado.

`retrieve_knowledge` con `scope=code|both` consulta esa colección si existe. Comandos Mac exactos: Project store `docs/mac-runbook.md` (§2 index-code, §4 index-docs).

## Layout

| Ruta | Rol |
|---|---|
| `rfp_agent/graph.py` | `StateGraph` de cinco roles + arista de tope |
| `rfp_agent/mcp.py` | Servidor simulado `tools/list` / `tools/call` |
| `rfp_agent/db.py` | Carga xlsx + golden + facts |
| `rfp_agent/evaluate.py` | Ejecuta `sql_verificacion` |
| `data/rfp.xlsx` | Corpus de 174 requisitos |
| `corpus/descripcion_funcional.pdf` | Fuente de los hechos golden |
| `.env.example` | Plantilla; no hay secretos en el repo |

Baseline de recuperación: ideas de [taller02](https://github.com/jorgecunalata/taller02) (Qdrant, `bge-m3`, `kb_status`). No se vendió el zip ni el pipeline Retrieve-Read de generación única.
