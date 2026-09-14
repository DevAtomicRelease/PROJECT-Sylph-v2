# Project Sylph — Agent Architecture

Sylph is a **local-first desktop agent** with a VRM avatar, voice, screen
awareness, persistent memory, and a permission-gated set of actions (apps,
reminders, code, web). Everything runs on one consumer machine — brain, voice,
and ears — with no cloud dependency and no per-use cost.

This document describes how the agent is put together and how to extend it.

---

## 1. High-level architecture

```
┌───────────────────────────── Tauri desktop app (Rust + WebView) ─────────────────────────────┐
│                                                                                               │
│   Window "main"  (avatar overlay)                Window "dashboard"  (chat + controls)         │
│   React + three.js:                              React:                                        │
│     • VRM render, lip-sync, expression             • text/speech chat                          │
│     • plays TTS audio, mic capture                 • Controls tab (capability toggles)         │
│     • global push-to-talk (Ctrl+Shift+Space)       • its own socket, no audio                  │
│                          │                                     │                               │
│                          └───────── WebSocket /ws + REST /api/* ┘                              │
│   Rust side: window mgmt, tray, hotkey, screen capture (xcap), sidecar supervisor              │
└───────────────────────────────────────────────┬──────────────────────────────────────────────┘
                                                 │  spawns + supervises
                    ┌────────────────────────────▼─────────────────────────────┐
                    │   Python "brain" sidecar  — FastAPI on 127.0.0.1:8420      │
                    │                                                            │
                    │   VAD → STT → Planner (LLM tool-loop) → TTS → dual payload │
                    │   Memory (short + long)  ·  Permissions  ·  Scheduler      │
                    │   Tools: comms · web · files · screen · system · code      │
                    └───────────────────┬────────────────────────────────────────┘
                                        │  local HTTP
                              ┌─────────▼──────────┐
                              │  Ollama daemon      │   qwen3.5:4b (local, GPU)
                              └────────────────────┘
```

Three runtimes:

| Layer | Runtime | Role |
|---|---|---|
| Shell / overlay | Tauri v2 (Rust + WebView) | windows, tray, hotkey, mic + screen capture, sidecar supervision |
| UI | React + three.js | avatar rendering & lip-sync (main), chat + controls (dashboard) |
| Brain | Python 3.12 (FastAPI) | LLM orchestration, STT, TTS, memory, vision, tools |
| Voice worker | Python (isolated venv, optional) | Chatterbox TTS subprocess (only when `TTS_ENGINE=chatterbox`) |

---

## 2. The two windows

Both windows load the **same** frontend bundle; `src/main.tsx` routes by Tauri
window **label**:

- **`main`** — the always-on, transparent, click-through-ish avatar overlay
  above the taskbar. Renders the VRM, plays TTS audio, drives lip-sync and
  expression, owns the microphone and global push-to-talk.
- **`dashboard`** — a normal resizable window opened from the tray
  ("Open Dashboard"). Text/speech chat plus the **Controls** tab. It runs its
  own `SidecarSocket` and registers **no audio handler**, so spoken replies play
  only through the avatar overlay (never doubled).

The sidecar broadcasts to all connected clients, so multiple windows coexist.

---

## 3. The brain sidecar

FastAPI app (`sidecar/main.py`) on `127.0.0.1:8420`.

- **WebSocket `/ws`** — the main IPC channel. Text frames are JSON messages
  (`chat`, `vad_flush`, `speak`, `screen_capture_response`, `confirm_response`,
  `mood_update`, `transcript`, `turn_complete`, `tts_start`, `reminder`, …).
  Binary frames carry the dual audio+viseme payload.
- **REST `/api/*`** — health, memory viewer, settings, and `/api/permissions`
  (read/write the capability config for the Controls tab).

### Voice pipeline (voice → voice)

```
mic (frontend) → PCM frames over /ws
  → VAD (Silero, CPU) detects an utterance
  → faster-whisper STT  (energy + no-speech gate to reject hallucinations)
  → Planner.process_user_message()
       ├─ deterministic intent router (avatar moves etc., no LLM)
       ├─ retrieve memories (Chroma) + build system prompt
       ├─ LLM tool-loop (qwen3.5:4b, bounded by MAX_TOOL_ITERATIONS)
       └─ sentence-split → TTS (Kokoro) → dual payload (audio + visemes)
  → frontend AudioPlayer plays audio; VisemeScheduler drives lip-sync;
    ExpressionDriver flashes the per-clause emotion the sidecar sends.
```

Text chat is the same pipeline via `handle_chat_request`, with a per-turn
`speak` flag that mutes TTS for text-only replies.

### Screen vision

When a message matches a screen query (or the model calls `look_at_screen`),
the sidecar asks the frontend to capture the primary monitor (Rust `xcap` →
JPEG), OCRs it (RapidOCR) **and** hands the actual image to the multimodal
`qwen3.5:4b` so it truly sees pixels, not just OCR text. The avatar's own
window is excluded from capture (`WDA_EXCLUDEFROMCAPTURE`) so Sylph never sees
herself.

---

## 4. The planner (agent loop)

`sidecar/llm/planner.py` — a lean, hand-rolled tool-calling loop tuned for a
**local small model** (deterministic where possible, LLM for intent, bounded
steps). Per turn:

1. **Intent router** handles spatial/avatar commands with zero LLM latency.
2. Assemble system prompt + memories (+ screen image if present).
3. Loop up to `MAX_TOOL_ITERATIONS` (6):
   - Advertise **only the tools whose capability domain is enabled** (permissions).
   - Stream from Ollama; decide tool-call vs text.
   - On a tool call: **permission check** → optional confirm → execute → feed
     the result back.
   - On text: stream clauses to TTS via a decoupled queue.
4. Infer the response emotion (keyword) → mood impulse → avatar expression.

Frameworks note: `langgraph` is a dependency but the loop is intentionally
custom — heavy multi-agent frameworks assume frontier models and flail on a 9b.

---

## 5. Capabilities & tools

Tools live in `sidecar/tools/` and are registered in the planner's `TOOLS`
list + `_tool_map`. Each maps to a **capability domain**.

| Domain | Tools | Module |
|---|---|---|
| `comms` | list/read/draft/send email, calendar events | `tools/gmail_tool.py`, `tools/calendar_tool.py` |
| `web` | `search_web`, `read_page`, `web_navigate`, `web_click`, `web_type`, `web_page_text` | `tools/web_tool.py`, `tools/web_session.py` |
| `files` | `list_files`, `read_file`, `search_files` | `tools/file_tool.py` |
| `screen` | `look_at_screen` | (capture path in `main.py`) |
| `system` | `open_app`, `focus_app`, `list_open_windows` | `tools/system.py` |
| `schedule` | `schedule_reminder`, `list_reminders`, `cancel_reminder` | `tools/schedule_tool.py`, `scheduler.py` |
| `code` | `write_code`, `read_code`, `list_workspace`, `run_python` | `tools/code_tool.py` |
| (ungated) | `move_avatar` — Sylph controlling her own body | planner |

**Non-destructive by design.** No delete, kill/close app, auto-send, purchase,
or arbitrary shell. `run_python` is jailed to `~/.sylph/workspace` with a
timeout; `web_*` only drives a browser; file access can be restricted to an
allowlist.

---

## 6. Permission model

`sidecar/agent_permissions.py` — a single JSON file is the source of truth for
what Sylph may do.

- **File:** `~/.sylph/config/permissions.json`
- **Shape:** `{ "domains": { "<domain>": { "enabled": bool, ... } } }`; the
  `files` domain additionally carries `"allowed_dirs": []` (empty = unrestricted).
- **Enforcement (every turn):** disabled domains' tools are neither advertised
  to the model nor executed; each call is re-checked before running
  (`check_tool`), and file paths are validated against the allowlist.
- **Fail-safe:** a missing/partial/corrupt config falls back to defaults (all
  enabled, merged per-key); ungated tools are always allowed.
- **UI:** the Dashboard **Controls** tab reads/writes this via
  `GET/POST /api/permissions`. New domains appear automatically because the UI
  iterates the config.

There is no destructive tier — actions are either **read** (auto) or **act**
(gated by the domain toggle, with an allowlist for files and a confirm dialog
retained for the pre-existing email/calendar mutations).

---

## 7. Reminders & routines

`sidecar/scheduler.py` — an `AsyncIOScheduler` started in the FastAPI lifespan.

- Natural `when` phrases → one-shot / interval / cron triggers
  ("in 30 minutes", "at 09:00", "every 2 hours", "daily at 5 pm").
- Specs persist to `~/.sylph/config/schedules.json`; recurring reminders
  re-register on start, lapsed one-shots are dropped.
- On fire: Sylph **speaks** the reminder and the frontend gets a `reminder` event.

---

## 8. Memory

- **Short-term** — `sidecar/memory/short_term.py`, SQLite at
  `~/.sylph/memory/sessions.db`. Message **content is encrypted at rest**.
  On startup the planner reattaches to the most recent thread and reloads its
  window, so conversation survives restarts.
- **Long-term** — `sidecar/memory/long_term.py`, ChromaDB at
  `~/.sylph/memory/longterm/`. Facts are stored **encrypted**, but embeddings
  are computed from the plaintext (all-MiniLM, self-embedded), so semantic
  search still works over encrypted-at-rest documents.
- **Encryption** — `sidecar/memory/crypto.py`, Fernet. Key from the OS keyring
  if available, else a `0600` key file at `~/.sylph/config/memory.key`.
  Version-tagged ciphertext; legacy plaintext reads back unchanged
  (forward-compatible). Toggle with `MEMORY_ENCRYPTION=off`.

---

## 9. Files & config on disk

| Path | Purpose |
|---|---|
| `.env` (repo root) | model, TTS engine, STT device, timeouts, encryption toggle |
| `~/.sylph/config/permissions.json` | capability config (Controls tab) |
| `~/.sylph/config/schedules.json` | persisted reminders |
| `~/.sylph/config/memory.key` | memory encryption key (fallback to keyring) |
| `~/.sylph/config/token.json` | Google OAuth token (email/calendar) |
| `credentials.json` (repo root) | Google OAuth client (optional; enables comms) |
| `~/.sylph/memory/sessions.db` | short-term memory |
| `~/.sylph/memory/longterm/` | long-term vector store |
| `~/.sylph/workspace/` | sandbox for `code` tools |

---

## 10. Key `.env` knobs

```env
MODEL_TAG=qwen3.5:4b            # local brain. 4b = fast/live; 9b = smarter but slower here
OLLAMA_BASE_URL=http://127.0.0.1:11434
TTS_ENGINE=kokoro              # kokoro (smooth, live) | chatterbox (emotional, needs venv) | zonos
SPEECH_CACHE=off              # Zonos-voiced mumble cache; off = one consistent Kokoro voice
STT_DEVICE=auto               # auto|cuda|cpu — cpu frees ~1 GB VRAM for a bigger model
STT_MODEL=distil-small.en     # faster-whisper model
MEMORY_ENCRYPTION=on
# SYLPH_CONFIRM_TIMEOUT / SYLPH_CAPTURE_TIMEOUT / SYLPH_CODE_TIMEOUT
```

Hardware note (RTX 4060, 8 GB): 4b + GPU STT + Kokoro co-reside comfortably.
9b needs the GPU largely to itself (`STT_DEVICE=cpu`) and answers slower — good
for text chat, not ideal for live voice.

---

## 11. Adding a new capability (extension guide)

1. Write the tool functions in `sidecar/tools/<name>.py` (async if they do I/O
   or subprocess work, so they don't block the event loop).
2. Export them in `sidecar/tools/__init__.py`.
3. Add JSON tool definitions to `TOOLS` and entries to `_tool_map` in
   `sidecar/llm/planner.py`.
4. Map each tool to a domain in `TOOL_DOMAINS`, and add the domain to
   `DEFAULT_CONFIG` + `DOMAIN_META` in `sidecar/agent_permissions.py` — it then
   appears in the Controls tab automatically.
5. Mention the ability in `sidecar/llm/prompts/system_prompt.txt`.
6. Add tests under `sidecar/tests/`.

Keep new actions **non-destructive**, and prefer deterministic behavior over
"let the model figure it out" — the local model rewards narrow, well-described
tools.

---

## 12. Testing

`sidecar/tests/` (pytest): emotion maps, phoneme/duration/viseme, memory
round-trip, memory encryption, planner history, permissions, scheduler parsing,
and the code sandbox. Run:

```bash
cd sidecar && ../venv/Scripts/python.exe -m pytest tests -q
```

Frontend type-check: `cd tauri-app && node_modules/.bin/tsc --noEmit`.
Rust check: `cargo check` in `tauri-app/src-tauri`.
