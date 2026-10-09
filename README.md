<p align="center">
  <img src="docs/logo.svg" width="112" alt="Engram logo">
</p>

<h1 align="center">Engram</h1>

<p align="center">
  <b>One memory for all your AI assistants.</b><br>
  Claude Code, ChatGPT, Gemini, Cursor — they all write to it and recall from it, over MCP or REST.
</p>

<p align="center">
  <a href="LICENSE"><img alt="License: MIT" src="https://img.shields.io/badge/license-MIT-7C3AED"></a>
  <img alt="Python 3.11+" src="https://img.shields.io/badge/python-3.11%2B-4338CA">
  <img alt="MCP" src="https://img.shields.io/badge/MCP-streamable%20HTTP-4338CA">
  <img alt="Postgres + pgvector" src="https://img.shields.io/badge/Postgres-pgvector-4338CA">
</p>

<p align="center">
  <img src="docs/recall.gif" width="720" alt="recall works out which entities a request is about: name match, embeddings and an LLM planner, then one hop through the graph"><br>
  <sub>How <code>recall</code> works out what a request is about, before it searches anything.</sub>
</p>

---

Every assistant starts from zero. Tell Claude you're job hunting, and ChatGPT still doesn't know.
Ask any of them *"how is my job search going?"* and the best a vector store can do is return the five
sentences that look most similar — it can't tell you that you've applied **14 times in 16 days**,
that **3 became interviews**, or that **system design came up weak in both onsites**.

**Engram** is a single, self-hosted memory service that every assistant shares. Agents call
`remember` when something happens and `recall(situation)` before they act. Engram turns free text
into typed memories in the background and answers with one ranked, token-budgeted brief:

An illustrative brief:

```text
recall("I'm applying for jobs")

## You
- Backend/AI engineer, prefers concise answers, targets remote roles

## Current task
- deadline: CV to Acme by Friday

## Job search so far
14 events since 2026-09-14 (16d), last 2026-09-30 · applied 10 · interview 3 · rejection 1 · outcomes: rejected 1
Momentum is good; DSA rounds go well, system design is the recurring weak spot.

## Lessons
- [r:7] System design was flagged in 2/2 onsite interviews — prep it before the next one

## Relevant history
- [e:41] 2026-09-29 Acme onsite: system design round went poorly
- [e:38] 2026-09-24 Beta Labs rejection after recruiter screen

## How you do it
- [f:2c…] How to apply: 1. cv send <company> → 2. log it → 3. follow up in 7 days
```

## See it in two minutes

https://github.com/user-attachments/assets/efbecddf-52d2-4037-bbf8-89955e87603e

## How it works

Engram uses every kind of memory where it fits, and one retrieval pipeline ranks them together.

| Memory type | Lives in | Example |
|---|---|---|
| **Working / context** | the brief `recall()` returns, sized to your token budget | — |
| **Short-term** | `session_state` — expires after the task | `format: PDF`, `due: Friday` |
| **Episodic** | `episodes` table → **SQL counts, first/last dates, outcome splits** | "applied 10 · interview 3" |
| **Semantic** | [mem0](https://github.com/mem0ai/mem0) on pgvector, with supersession | "Prefers FastAPI over NestJS" |
| **Procedural** | mem0 (`teach`) | "How to deploy: test → build → push → verify" |
| **User profile** | an always-included block built from your most important facts | — |
| **Shared** | one store for every agent, each write tagged with the agent | — |
| **External** | your documents (resume, projects…) with hybrid vector + full-text search | — |
| **Reflective** | `reflections` with evidence + per-topic digests, built by a background "sleep" pass | "System design is the weak spot" |

Everything is joined by **entities** (`topic:job-search`, `company:acme`, `project:cityfix`):
mention one in a situation and every layer that touches it comes along.

```mermaid
flowchart LR
    A["Any assistant<br/>(MCP / REST / hooks)"] -- remember --> Q[(ingest queue)]
    Q --> X["extract<br/>1 LLM call → typed JSON"]
    X --> R["resolve entities<br/>split · dedupe · redact"]
    R --> E[(episodes)]
    R --> F[(mem0 facts &<br/>procedures)]
    R --> S[(session state)]
    F --> C{"reconcile<br/>duplicate / supersedes"}
    E & F --> Z["consolidate (sleep pass)<br/>digests · reflections · profile"]
    A -- "recall(situation)" --> P["plan<br/>entities · weights"]
    P --> CH["channels: profile · session · aggregates ·<br/>episodes · facts · procedures · lessons · docs"]
    CH --> K["rank: RRF × kind weight × importance ×<br/>recency × confidence × entity boost"]
    K --> B["token-budgeted brief"] --> A
```

**Write path.** `remember` returns immediately; a worker extracts entities, events, facts,
procedures and session notes in **one LLM call**, resolves dates in your timezone, redacts secrets,
splits multi-company career events so counts stay right, de-duplicates events that were mentioned
twice, and reconciles new facts against old ones (a changed preference *supersedes* the old one —
hidden from recall, kept in history).

**Read path.** `recall` matches entities (names and aliases, embeddings, an LLM planner unless `fast`, then one
hop of co-occurrence), runs
the channels, fuses them with reciprocal-rank fusion, applies importance, recency half-lives and
confidence, and packs a sectioned brief under your token budget. `fast=true` uses no LLM at all,
and recall keeps working (profile, counts, full-text) even if the embedding model is down.

**Sleep pass.** Every 6 hours (and after sessions end) Engram rebuilds topic digests, derives
lessons from ≥2 related events (with the events as evidence), refreshes your profile and expires
old session notes. Nothing is deleted.

Full design: [`docs/design.md`](docs/design.md).

## Quick start

Requirements: Python 3.11+ with [uv](https://docs.astral.sh/uv/), Postgres 16 with
[pgvector](https://github.com/pgvector/pgvector), and [Ollama](https://ollama.com) for local embeddings.

```bash
git clone https://github.com/mohammed-usmani/engram && cd engram
uv sync
ollama pull nomic-embed-text            # embeddings run locally and free

cat > .env <<'ENV'
CONNECTION_STRING=postgresql+asyncpg://postgres:postgres@localhost:5432/engram
DATA_DIR=data/examples                  # text files imported once on first start (optional)
USER_NAME=Alex                          # how prompts refer to you
USER_TZ=Asia/Kolkata                    # "yesterday" means *your* yesterday
TOGETHER_API_KEY=...                    # or DASHSCOPE_/GEMINI_/GROQ_/MISTRAL_/OPENAI_/ANTHROPIC_API_KEY
MEMORY_LLM_CHAIN=together,ollama        # tried in order; local Ollama as the fallback
MEMORY_MODEL_TOGETHER=deepseek-ai/DeepSeek-V4-Flash-0731
ENV

uv run alembic upgrade head
uv run uvicorn main:app --host 127.0.0.1 --port 8001
```

On macOS, `scripts/launchd/install.sh` installs it as a login service (plus nightly backups).
On Linux, `scripts/systemd/install.sh` does the same with systemd user units (`sudo loginctl enable-linger $USER` keeps it running while logged out).

Try it:

```bash
curl -X POST localhost:8001/api/memory/remember -H 'Content-Type: application/json' \
  -d "{\"text\": \"Applied to Zomato and Swiggy for SDE-2 roles today\", \"agent\": \"curl\",
       \"occurred_at\": \"$(date +%Y-%m-%dT%H:%M:%S%z)\"}"
# a few seconds later
curl -X POST localhost:8001/api/memory/recall -H 'Content-Type: application/json' \
  -d '{"situation": "how is my job search going?", "fast": true}'
```

## Connect your assistants

| Assistant | Setup |
|---|---|
| **Claude Code** | `claude mcp add --transport http -s user engram http://localhost:8001/mcp` — and add [`scripts/claude_memory_hook.py`](scripts/claude_memory_hook.py) as a `UserPromptSubmit` + `PreCompact` + `SessionEnd` hook (injects a recall on the first prompt, saves what's unsaved before compaction and at the end) |
| **Codex CLI** | `~/.codex/config.toml` → `[mcp_servers.engram]` `url = "http://localhost:8001/mcp"` |
| **Gemini CLI** | `~/.gemini/settings.json` → `{"mcpServers": {"engram": {"httpUrl": "http://localhost:8001/mcp"}}}` |
| **Cursor / Windsurf / VS Code** | `{"mcpServers": {"engram": {"url": "http://localhost:8001/mcp"}}}` |
| **Claude Desktop** | `{"mcpServers": {"engram": {"command": "npx", "args": ["mcp-remote", "http://localhost:8001/mcp"]}}}` |
| **claude.ai** | set `ADMIN_TOKEN`, expose port 8001 (e.g. `tailscale funnel --bg 8001`), add `https://<public-url>/mcp` as a custom connector with request header `Authorization: Bearer <token>` |
| **ChatGPT / other OAuth-only apps** | also set `PUBLIC_URL=https://<public-url>`; add `https://<public-url>/mcp` with **OAuth**, then approve on Engram's consent page by entering `ADMIN_TOKEN` |
| **Anything else** | REST under `/api/memory/*`, OpenAPI at `/openapi.json` |

**Conversations save themselves.** [`scripts/transcript_sync.py`](scripts/transcript_sync.py) (installed by the
launchd/systemd scripts, every 10 minutes) reads Claude Code, Codex CLI and Gemini CLI transcripts and sends
new user/assistant prose to Engram in chunks, each stamped with the agent and the time of its last message.
Long and still-open sessions are captured in full; one offset file shared with the hook means nothing is sent
twice. Tools whose transcripts aren't readable (Antigravity, web apps) save through `remember`.

**Every write says who and when.** `remember` requires `agent` (claude-code, claude-web, chatgpt, gemini,
codex…) and `occurred_at` (when it happened, ISO 8601); `teach` and the document tools require `agent`, and
documents record it as `updated_by`. Writes without them are rejected.

Local tools on `http://localhost:8001/mcp` (Claude Code, Codex, Gemini CLI, Antigravity, Cursor) need no token
or sign-in; the token and OAuth only apply to requests that arrive through a tunnel.

For assistants without hooks, add one custom instruction: *"Before answering anything personal or
starting a task, call `recall` with my words. When I share a fact, preference, decision or outcome,
call `remember`."*

**Tools:** `recall` · `remember` · `note` · `teach` · `expand` · `timeline` · `forget`, and for
documents `list_documents` · `get_document` · `search_context` · `edit_document` · `save_document` ·
`delete_document`

## Your documents

Resume, projects, skills, experience, education, achievements, certifications and **profiles** live
in the database and are edited in the browser at **http://localhost:8001/admin**, or by any assistant over MCP (`edit_document` for a
small change, `save_document` to create or replace) — no files, no reseeding. Write plain text: `Key: value`
lines at the top become fields, a line like `Summary:` starts a section; both are re-derived and the
document is re-embedded on every save, so recall sees the change immediately. Documents appear in
briefs under *Documents* (e.g. `[d:resume/master_resume]`).

A **profile** document (`profile/linkedin`, `profile/indeed`, `profile/github`…) holds the exact text
a public profile currently shows, so an assistant asked "what does my LinkedIn say?" quotes it instead
of guessing. Note anything you haven't captured yet in the document itself, so nothing gets invented.

Documents aren't only for careers. Each has a **type**: the career ones above, plus everyday
built-ins (`note`, `person`, `health`, `finance`, `home`, `travel`, `learning`, `reference`,
`interview`). Any other type works too: give it a one-line description the first time it is used,
and it joins the registry (`GET /api/types`, `list_document_types`). Near-duplicates (`projects` when
`project` exists) are refused with a suggestion.

Each document also has a **privacy** level:

| Level | Behaviour |
|---|---|
| `normal` | everywhere: recall briefs, search, every assistant |
| `private` | never in automatic recall or search; listed by title, returned when asked for by slug |
| `local-only` | as private, and invisible to anything arriving through the public link (web assistants) |

**Dates you care about** are just fields. Put `Expires: 2027-03-14`, `Renews: 14/03/2027`, `Due:`,
`Deadline:`, `Appointment:`, `Next service:` or `Matures:` (one-off), or `Birthday: 10 Oct 1976` or
`Anniversary:` (yearly), at the top of any document. Every recall brief then opens with a
**Coming up** section: the next 60 days, plus anything overdue in the last 30. The same list is on
the `/memory` overview, from the `upcoming` tool and from `GET /api/upcoming`. A private document's
date shows only as a reminder line (title, label, date), never its content.

**Attachments**: attach PDFs, scans, photos or text files to any document from its admin page,
over MCP (`attach_file`, base64), or with `curl -F file=@policy.pdf
localhost:8001/api/admin/<slug>/attachments`. Text is extracted (the PDF text layer via poppler,
OCR via tesseract for images and scans), so search and recall find the document by what is *in*
the file. Files live in Postgres, so backups and machine moves carry them, and they share their
document's privacy. Install the extractors with `brew install poppler tesseract` or
`pacman -S poppler tesseract tesseract-data-eng`; without them, files are still stored, just not
read.

The admin pages follow the same rule as everything else: no token from your own browser on
`localhost`, the token for anything arriving through a tunnel.

Text files are only an import path: on an empty database Engram imports `DATA_DIR`
(`data/examples/` ships a fictional sample), and **Import new files** adds files that aren't in the
database yet. Importing never overwrites a document you've edited.

## See what it knows

Open **http://localhost:8001/memory** for a browser view of the whole memory:

| Tab | Shows |
|---|---|
| **Overview** | counts per layer, topics with their summaries, recent events, last backup — and a **recall box**: type what you'd say to an assistant and see the exact brief it would receive |
| **Topics** | every entity (project, company, topic…) with event counts and its "so far" summary; click through to its timeline |
| **Events** | everything that happened, filterable by text, kind and topic |
| **Facts** | facts, preferences and procedures, searchable (superseded ones are hidden but kept) |
| **Lessons** | reflections with confidence and how many events support them |
| **Notes** | short-term notes for tasks in progress |
| **Queue** | every `remember` / session save, what it extracted, errors — with **Retry** |

Anything wrong? **Forget** removes it everywhere at once — including the profile at the top of every
brief, which is built live from your current facts.

## Choosing the extraction model

Every memory write costs one LLM call (plus a small one when a new fact may contradict an old one).
Any OpenAI-compatible provider works: DashScope (Alibaba), Together, Groq, Mistral, Gemini, Cerebras,
OpenAI, Anthropic, or local Ollama, and you switch per job in **Settings → Which model does what**. Pick
one with the extraction test set (see Evaluation below): labelled cases for multi-company applications,
relative dates, interview + offer in one sentence, procedures, session notes, noisy coding transcripts and
small talk that must be ignored, plus regressions from real failures (history recaps, plans that aren't
applications, quoted examples that didn't happen):

```bash
uv run python scripts/eval_extraction.py dashscope deepseek-v4-flash-0731
uv run python scripts/eval_extraction.py ollama qwen2.5-coder:7b --concurrency 1
```

Results at the time of writing:

| Model | Score | Median latency | Est. cost / month* |
|---|---|---|---|
| `deepseek-ai/DeepSeek-V4-Flash-0731` (Together) | 96%, before the multi-company split fix that targets its only miss | 21 s | ≈ $1.30 |
| `qwen2.5-coder:7b` (local Ollama, M4 16 GB) | 92% | 20 s | free |
| `qwen3:8b` (local Ollama, thinking off) | 83% | 20 s | free |

<sub>*≈ 15 coding sessions, 20 notes and 30 recalls a day. Writes are asynchronous, so latency never blocks your assistant.</sub>

### Batch mode for big backlogs

When the queue piles up (an import, a long day of sessions), the **Queue** tab on the Memory page has a
**Switch queue to batch mode** button. The live workers then stand aside and, about once a minute, every
waiting job goes to Together's Batch API as one batch; results come back in minutes to hours and are saved
exactly like live extractions. Batches already sent are still collected after you switch back. Batches cost
about half and use their own rate limits, but Together only batches serverless models (DeepSeek-V4-Flash is
refused), so they use `MEMORY_BATCH_MODEL`, default `openai/gpt-oss-120b` (126/126 on the eval above).

## Evaluation

When something looks wrong (an assistant got an empty brief, a fact is stale, an event was invented),
the **Evaluation** and **Traces** pages show where.

| Layer | What it checks | Cost |
|---|---|---|
| **Data health** | Facts that contradict each other (look-alike pairs with different numbers or tense, confirmed by one batched LLM call and cached), duplicate events, one thing under two topic names, one-off topics, unreadable or wrong-weekday dates, oversized facts, failed queue jobs, recall and extraction quality from the last 7 days of traces, backup age | free + a fraction of a cent |
| **Golden questions** | Questions you'd really ask, each with text the brief must contain and must not contain (e.g. "3.9", not "3.6"). Save one from the Recall inspector or a trace | free (fast recall) |
| **Extraction test set** | The labelled cases above, against any provider/model, without changing what live extraction uses | ~20 LLM calls |
| **Traces** | Every recall (who asked, plan, planner status, items, budget by section, timings, the brief) and every extraction (provider, fallbacks, what was kept, what the grounding check dropped and why), kept 30 days, with 👍/👎 feedback | free |

Data health and golden questions run every night at 03:00 (after the backup); extraction runs when you ask.
From the command line: `uv run python -m src.eval.runner health|recall|extract [provider model]`.

## Your data stays yours

- **Local-first.** One process on `127.0.0.1:8001`; embeddings run on your machine.
- **Secrets are redacted before anything is stored** — API keys, tokens, private keys, URL
  credentials, card numbers (Luhn-checked), "my password is …".
- **Auth.** With `ADMIN_TOKEN` set, every path except `/api/health` (and the OAuth endpoints below)
  requires the bearer token for remote or tunnelled requests; direct local use keeps working.
- **OAuth for apps that can't send a header** (ChatGPT, Gemini). Set `PUBLIC_URL` to the HTTPS
  address the app reaches (e.g. your Tailscale Funnel URL) alongside `ADMIN_TOKEN`. Engram then serves
  MCP OAuth discovery, client registration, `/authorize`, `/token` and `/revoke`. Connecting sends you
  to `/oauth/consent`, where you approve by typing `ADMIN_TOKEN`, so OAuth never grants more than the
  token does. Clients and token hashes are kept in `OAUTH_STORE` (default
  `~/.config/engram/oauth.json`); access tokens last an hour and refresh silently, refresh tokens 180
  days. Delete that file to sign every app out.
- **Encrypted backups.** `scripts/backup.sh` dumps Postgres + your documents + fact history into one
  [age](https://age-encryption.org)-encrypted archive (nightly via launchd, keeps 14). Point
  `ENGRAM_BACKUP_DIR` at a **private** git repo and every backup is pushed off-machine.
  `scripts/restore.sh` rebuilds everything on a new machine and refuses to overwrite a database that
  already holds memories unless you pass `--force`.
- **Several devices.** Run Engram on one machine and connect the others over
  [Tailscale](https://tailscale.com) with `ADMIN_TOKEN` — one memory, no sync conflicts.

## API

| | |
|---|---|
| `POST /api/memory/remember` | `{text, agent, occurred_at, session_id?}` → queued job |
| `POST /api/memory/recall` | `{situation, budget_tokens?, session_id?, fast?}` → `{brief, items, plan}` |
| `POST /api/memory/note` | short-term note for a session |
| `POST /api/memory/teach` | save a procedure |
| `GET /api/memory/item/{id}` · `DELETE …` | expand / forget an item (`e:`, `f:`, `r:`, `d:` ids) |
| `GET /api/memory/timeline/{entity}` | chronological events for `topic:job-search`, `company:acme`, … |
| `GET /api/memory/jobs` · `POST …/jobs/{id}/retry` | extraction queue status / retry |
| `POST /api/memory/consolidate` | run the sleep pass soon |
| `GET /api/upcoming?days=60` | tracked dates coming up (and recently overdue) |
| `GET /api/types` | document types with descriptions and counts |
| `POST /api/admin/{slug}/attachments` | multipart `file` → attach to a document |
| `GET /api/admin/attachments/{id}` | download an attachment |

Engram also ships a document manager (`/admin`), a document store with hybrid search, and a RAG chat
(`/chat`) over your documents.

## Development

```bash
createdb engram_test   # TEST_DATABASE_URL defaults to <CONNECTION_STRING db>_test
uv run pytest
```

```
src/memory/      models · llm (provider chain) · facts (mem0) · entities · extract · ingest
                 recall · consolidate · worker · api · dashboard (/memory) · backfill
src/mcp_server.py  MCP tools (also served at /mcp by main.py)
scripts/         claude_memory_hook.py · eval_extraction.py · backup.sh · restore.sh · launchd/
data/examples/   fictional sample documents
docs/design.md   design notes
```

## License

[MIT](LICENSE) © 2026 Mohammed Usmani
