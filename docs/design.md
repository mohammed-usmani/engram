# Engram — Design

Date: 2026-10-01 · Status: implemented

## Goal

One memory for every AI assistant I use (Claude Code, claude.ai, ChatGPT,
anything MCP or HTTP). Agents write what happens; any agent can ask
`recall(situation)` and get a compact, ranked brief that blends all memory
types — profile, current session, events and their aggregates, facts,
procedures, lessons, documents. "I'm applying for jobs" must return how many
applications, since when, outcomes, interview patterns, and how I apply.

Single user, local-first, runs as one local service.

## Memory types → storage

| # | Type | Lives in | Written by |
|---|---|---|---|
| 1 | Working/context | Not stored — the brief `recall()` returns, token-budgeted | built per call |
| 2 | Short-term | `session_state` (session_id, key, value, expires_at) | `note()`, extractor |
| 3 | Episodic | `episodes` (occurred_at, kind, entities[], outcome, sentiment, importance, summary, payload, embedding) | extractor |
| 4 | Semantic | mem0 OSS on pgvector (`user_id=me`), plus `context_documents` (curated career docs) | extractor, consolidation |
| 5 | Procedural | mem0 with `metadata.kind=procedure` | `teach()`, consolidation |
| 6 | User/personal | `profile` block (row in `memory_blocks`) always included | consolidation |
| 7 | Shared/org | mem0 metadata `agent`, `scope` (`global` / `project:<x>`) | any agent |
| 8 | External/retrieval | `context_documents` hybrid search (existing) | seed/ingest |
| 9 | Reflective | `reflections` (lesson, evidence episode ids, entities, confidence) + entity `digest` | consolidation |

`entities` (slug, kind, name, aliases[], digest, dirty) is the join key:
episodes, reflections and mem0 metadata reference entity slugs, so matching an
entity in a situation pulls every layer.

Note on mem0: v2 OSS writes are ADD-only and graph stores were removed. We
call `add(..., infer=False)` (our extractor already structured the data, so
mem0 makes no LLM call) and implement supersession ourselves via metadata
`valid_to` / `superseded_by`.

## Write path

Entry points: MCP/REST `remember(text, agent, session_id?, occurred_at?)`,
`note(session_id, key, value)`, `teach(name, steps)`, Claude Code
`SessionEnd` hook → `POST /api/memory/ingest`.

1. Enqueue into `ingest_jobs` (content sha256 unique → idempotent). Return job id.
2. Worker: redact secrets (API keys, bearer tokens, card numbers).
3. One structured LLM call → `{entities, episodes, facts, preferences,
   procedures, session_notes}`; relative dates resolved against `occurred_at`.
4. Entity resolution: slug/alias match, then embedding similarity ≥ 0.9, else create.
5. Store episodes; facts/preferences/procedures → mem0 `infer=False` with
   metadata `{kind, entities, importance, scope, agent, valid_to: null}`.
6. Reconcile: new fact vs top-5 similar same-entity facts with sim ≥ 0.85 →
   one batched LLM call → `duplicate | supersedes | compatible`. Duplicates
   deleted (new one), superseded ones get `valid_to`, `superseded_by`.
7. Mark touched entities dirty.

Failures: retry with backoff, 3 attempts, then `failed` (visible via
`GET /api/memory/jobs?status=failed`). Raw text never dropped.

## Recall

`recall(situation, budget_tokens=1500, session_id?, scope?, fast=false)`

1. **Plan.** Free: alias/name match of situation against `entities`. Then LLM
   planner (skipped if `fast`, cached by hash) → `{entities, kinds weights,
   since, aggregates: [...]}`. Aggregates come from a fixed menu —
   `count_by_kind`, `first_last`, `outcome_breakdown`, `recent_n` — never LLM SQL.
2. **Channels (parallel):** profile block · session state · entity digests ·
   episode aggregates · episodes (vector+FTS, entity/time filtered) · mem0
   facts (valid only, scoped) · mem0 procedures · reflections · documents
   (existing hybrid `retrieve`).
3. **Rank:** `score = rrf_relevance × kind_weight × importance × recency × confidence`.
   Recency half-life: episodes 30d, reflections 90d, facts/procedures none.
4. **Pack:** greedy by score/token under budget, MMR drop at sim > 0.9, sectioned
   markdown with ids (`e:41`, `f:<uuid>`, `r:7`, `d:slug`).
5. **Learn:** bump `access_count`, `last_accessed` on served items.

Also: `expand(id)`, `timeline(entity, since?)`, `forget(id)` (hard delete).

## Consolidation

Runs every 6h in the worker, on `SessionEnd`, or `POST /api/memory/consolidate`.

- Per dirty entity: rebuild digest from aggregates + recent episodes; derive
  reflections from ≥3 related episodes (with evidence ids); mark clean.
- Global: rebuild profile block from top facts by importance × access; expire
  session_state > 7d (promote unsaved notes to episodes first).
- Nothing is deleted by consolidation.

## LLM providers

Reuse `src/services/providers`. Add Mistral (OpenAI-compatible,
`https://api.mistral.ai/v1`). Add `json` mode to `generate`. Refresh stale
model IDs from each provider's `/models`.

Each memory task (`extract`, `reconcile`, `plan`, `consolidate`) uses a
fallback chain, default `gemini → groq → mistral → ollama` (free first);
`claude`, `openai` opt-in via env. 429 / timeout / invalid JSON → next
provider, logged. Keys from env (`GEMINI_API_KEY`, `GROQ_API_KEY`,
`MISTRAL_API_KEY`, `ANTHROPIC_API_KEY`, `OPENAI_API_KEY`) or the existing
`provider_settings` table. Chain override: `MEMORY_LLM_CHAIN=gemini,groq,...`.

Embeddings: local Ollama `nomic-embed-text` (768-d, matches existing vectors).

## Surfaces

- MCP (stdio + HTTP :8001, bearer auth): existing tools + `recall`,
  `remember`, `note`, `teach`, `expand`, `timeline`, `forget`.
- REST under `/api/memory/*` (same operations) for non-MCP assistants and the
  OpenAPI spec (Gemini Gems, ChatGPT Actions).
- Claude Code hooks: `SessionStart` injects `recall`, `SessionEnd` ingests.
- Worker: `python -m src.memory.worker` (queue + scheduled consolidation).

## Migration

Alembic `0004`, additive only (new tables). Existing `memories` rows copied
into mem0/episodes; old table kept. Backfill a markdown applications tracker (`applications.md`)
rows into `applied` episodes.

## Testing

pytest on the test DB; LLM calls replaced by fixture JSON. Golden test: ingest
~12 scripted job-search events → `recall("applying for jobs")` contains correct
count, first date, outcome split, the system-design reflection, within budget.
Fallback test: mocked 429 advances the chain.

## Skipped

Weight-tuning UI, recall feedback tool, graph DB. Add when recall quality plateaus.
