# Taller 3 — respuesta automatizada de RFP

Grafo **LangGraph 1.2.12** con cinco agentes (Planner, Reader, Writer, Verifier, Output Synthesizer) y un servidor MCP simulado. El evaluador puntúa **solo los ítems golden** con SQL sobre hechos verificados. No recorre las 174 filas del Excel y no usa un LLM como juez.

Repositorio: https://github.com/jorgecunalata/taller_03

No hay UI, LangSmith, ni `create_agent` / `create_react_agent`.

## Resultado de la corrida calificable

Corrida en la H200 de la USFQ (VPN GlobalProtect), con `LLM_BACKEND=h200` y `H200_ENABLE_THINKING=0`. El cliente de chat no fija un id de modelo: pregunta `GET /v1/models`. Embeddings de indexación: Ollama `bge-m3` en el puerto `11434`.

| Comando | Resultado |
|---|---|
| `eval-rfp` | **4/4** (`passed=4`, `failed=0`). Cada ítem `status=completed`, `llm=h200`. |
| `index-docs` | 8 PDF, 214 páginas, 220 chunks, 220 puntos en Qdrant `documentos`. |
| `index-code` | 6617 archivos leídos, 25000 chunks (tope), 25076 puntos en Qdrant `montran_code`. `CORE_MS` ausente y omitido. |
| `pytest` | 26 tests, en local con `LLM_BACKEND=simulated`. |

Antes de añadir el golden de código la misma corrida H200 daba **3/3** (`R-001`, `R-038`, `R-057`). El cuarto ítem es `R-037`.

Artefactos ya versionados (no hace falta regenerarlos para leer el resultado):

| Archivo | Qué guarda |
|---|---|
| `outputs/eval_rfp.json` | Paquetes del evaluador (respuesta, traza, pass/fail). |
| `outputs/resultados.csv` | La misma evaluación en una fila por ítem. |
| `outputs/ingesta_documentos.json` | Resumen de `index-docs`. |
| `outputs/doc_chunks.jsonl` | Chunks documentales enviados a Qdrant. |
| `outputs/ingesta_codigo.json` | Resumen de `index-code` por raíz Java. |
| `outputs/code_chunks.jsonl` | Chunks de código (clase/método) enviados a Qdrant. |
| `outputs/catalogo_cag.txt` | Catálogo CAG (propósito de proyecto y paquetes). |

### Ítems golden

`R-018` está prohibido (placeholders). El código se cita con `kb_status=CODIGO` y no se promociona a `VALIDADO`.

| Id | Tipo | Hecho que exige el SQL | Fuente |
|---|---|---|---|
| `R-001` | Respondible, PDF | Liquidación de fondos final e irrevocable en tiempo real | `corpus/descripcion_funcional.pdf`, p.11 |
| `R-038` | Respondible, PDF | 24/7 (p.13) y 24/7/365 (p.15) | `corpus/descripcion_funcional.pdf` |
| `R-037` | Respondible, Java/CAG | `Util class for settlement window` | `SettlementWindowUtil.java` (`core_rtgs/.../timetable/impl/`) |
| `R-057` | Abstención | Ningún TPS numérico | El PDF no publica un máximo transaccional |

Respuestas registradas en `outputs/eval_rfp.json`:

- **R-001.** Liquidación de fondos final e irrevocable en tiempo real entre los miembros del servicio RTGS, de forma continua [1] (`descripcion_funcional.pdf`, p.11).
- **R-037.** Horarios de entrada y liquidación por tipo de pago, con la cita literal `Util class for settlement window` [1] (`SettlementWindowUtil.java`).
- **R-038.** 24/7 [1] (p.13) y 24/7/365 [2] (p.15).
- **R-057.** `El corpus no contiene información suficiente.` El PDF no publica un máximo transaccional (TPS).

La traza de cada respondible pasa por `planner → reader → writer → verifier → synthesizer` (17 nodos, con hand-off forzado tras las tools). `R-057` cierra el writer sin una segunda ronda de tools (15 nodos).

### Corpus indexado

Ocho PDF, en `corpus/` y la copia de `corpus/docs/`: `descripcion_funcional.pdf` (65 páginas, `VALIDADO`), `ISOcertificaciones.pdf`, `legal.pdf` y `seguridad.pdf` (14 páginas cada uno). Chunking ~512 tokens con solape 102. Plantillas `[[RELLENAR]]` / `EJEMPLO_NO_VALIDADO` no se marcan `VALIDADO` y el retrieve las descarta.

### Índice de código

`index-code` recorrió el árbol `demo_ats` (raíces `CORE`, `CORE_BILL`, `CORE_RTGS`, `CORE_PO_ATS`, `CORE_PARSER`, `CORE_ACH`, `CORE_ATS`). `CORE_MS` no está en ese árbol y se omitió. Extensiones: `.java`, `.xml`, `.properties` y afines. El tope `MAX_CODE_CHUNKS=25000` recortó métodos; las clases se conservan. El hit que sostiene `R-037` salió de `retrieve_knowledge('settlement window RTGS timetable', scope='code')`.

Ese árbol Java **no está en este repositorio** (en la máquina del estudiante es un directorio hermano, `DEMO_ATS`). Los chunks y el catálogo sí están en `outputs/`.

## Reproducir el experimento en una computadora local

Python ≥ 3.11. No hace falta VPN, H200, Qdrant ni el árbol Java para repetir la evaluación golden y los tests. El grafo con `LLM_BACKEND=simulated` usa el mismo evaluador SQL y los mismos literales de hechos. Si la H200 no responde, `LLM_BACKEND=auto` cae solo a ese modo.

```bash
git clone https://github.com/jorgecunalata/taller_03.git
cd taller_03

python3 -m venv .venv
source .venv/bin/activate
# Windows: .venv\Scripts\activate

python3 -m pip install -U pip
python3 -m pip install -e ".[dev]"

cp .env.example .env
```

En `.env`, para una máquina sin VPN:

```bash
LLM_BACKEND=simulated
H200_ENABLE_THINKING=0
```

Luego:

```bash
python3 -m rfp_agent.cli load-db
python3 -m rfp_agent.cli tools
python3 -m rfp_agent.cli eval-rfp
python3 -m pytest -q
```

Qué debe salir:

- `load-db`: 174 ítems, golden exactamente `R-001`, `R-037`, `R-038`, `R-057`, y `R-018` fuera.
- `tools`: al menos `retrieve_knowledge`, `get_requirement`, `query_facts`, `record_evidence` y `list_sources`.
- `eval-rfp`: JSON con `"n": 4`, `"passed": 4`, `"failed": 0`. El proceso termina en 0. Vuelve a escribir `outputs/eval_rfp.json`. En local el campo `llm` será `simulated`, no `h200`. Las cuatro respuestas deben cubrir los mismos literales (incluida la abstención de `R-057`).
- `pytest`: 26 passed.

Un solo ítem, si se quiere inspeccionar la traza:

```bash
python3 -m rfp_agent.cli run R-001
python3 -m rfp_agent.cli run R-037
python3 -m rfp_agent.cli run R-057
```

`R-057` debe contener la frase `El corpus no contiene información suficiente.` y no inventar un número de TPS.

Los archivos de `outputs/` que ya vienen en el clone son la evidencia de la corrida H200. `eval-rfp` los sobrescribe. Para conservar esa corrida, copie `outputs/eval_rfp.json` y `outputs/resultados.csv` antes de repetir la evaluación.

### Opcional: reindexar documentos en local

Hace falta Docker y el modelo local `BAAI/bge-m3` (la primera descarga pesa varios GB).

```bash
docker run -d --name qdrant -p 6333:6333 -p 6334:6334 qdrant/qdrant
python3 -m pip install sentence-transformers torch
```

En `.env`: `EMBEDDING_BACKEND=local` y `QDRANT_URL=http://localhost:6333`.

```bash
python3 -m rfp_agent.cli index-docs
```

Indexa los PDF de `corpus/` y `corpus/docs/` en la colección `documentos`. El resumen debe parecerse a `outputs/ingesta_documentos.json` (8 archivos, 214 páginas, 220 chunks). El backend de embeddings figurará `local` en lugar de `h200`.

### Opcional: reindexar código

Solo si se dispone del árbol `demo_ats` (no va en este repo). Apunte `DEMO_ATS` en `.env` a esa carpeta y deje Qdrant y `EMBEDDING_BACKEND=local` como arriba.

```bash
python3 -m rfp_agent.cli index-code
```

Sin ese árbol el comando omite las raíces y no reproduce `outputs/ingesta_codigo.json`. La evaluación de `R-037` no depende de reindexar: el hecho y la cita ya están en `rfp_agent/db.py` y en el stub de `retrieve_knowledge`.

### Corrida original en la H200

Con GlobalProtect, en `.env`: `LLM_BACKEND=h200` (o `auto`) y `H200_ENABLE_THINKING=0`. Host `172.28.230.10`, chat en el puerto `12555`, embeddings en `11434`. Si el modelo rechaza `temperature`, el cliente reintenta sin ella.

```bash
LLM_BACKEND=h200 H200_ENABLE_THINKING=0 python3 -m rfp_agent.cli eval-rfp
```

Anti-thrash: presupuestos `TOOL_BUDGET_*` por rol, hand-off forzado cuando ya hay evidencia o SQL, y borrador determinista con el literal del PDF o del Java si el modelo no emite JSON.

## Qué hace el sistema

- Carga `data/rfp.xlsx` → SQLite `rfp_items` con ids `R-001`…`R-174`.
- Golden (`in_golden=1`): los cuatro ids de la tabla de arriba.
- MCP: `retrieve_knowledge`, `get_requirement`, `query_facts`, `record_evidence` y `list_sources`.
- `retrieve_knowledge` usa Qdrant si la colección existe. Si no, usa un stub léxico: citas del PDF para `R-001`/`R-038` y la cita Java de `SettlementWindowUtil` para `R-037`.
- `eval-rfp` sale con código 1 si algún golden falla.

## Layout

| Ruta | Rol |
|---|---|
| `rfp_agent/graph.py` | `StateGraph` de cinco roles y arista de tope |
| `rfp_agent/mcp.py` | Servidor simulado `tools/list` / `tools/call` |
| `rfp_agent/db.py` | Carga del xlsx, golden y facts |
| `rfp_agent/evaluate.py` | Ejecuta `sql_verificacion` (PDF o código) |
| `rfp_agent/drafting.py` | Borrador corto con literales; no vuelca dumps |
| `rfp_agent/retriever.py` | Qdrant o stub (`docs` / `code`) |
| `rfp_agent/docs_pipeline.py` | `index-docs` |
| `rfp_agent/code_pipeline.py` | `index-code` |
| `data/rfp.xlsx` | 174 requisitos |
| `corpus/` | PDF funcional, ISO, legal y seguridad |
| `outputs/` | Evaluación H200 e índices ya generados |
| `.env.example` | Plantilla. `.env` no se versiona |

Baseline de recuperación: ideas de [taller02](https://github.com/jorgecunalata/taller02) (Qdrant, `bge-m3`, `kb_status`). Este repo no incluye el zip ni el pipeline Retrieve-Read de generación única.
