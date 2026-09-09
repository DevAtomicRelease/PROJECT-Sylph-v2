# Project Sylph

A **local-first** AI desktop companion with a VRM 3D avatar, voice interaction,
screen awareness, and persistent memory — brain, voice, and ears all running on
a single consumer GPU with **no cloud dependency and $0 running cost**.

## ✨ Features

- **3D Avatar Overlay** — VRM avatar rendered with three.js, always on-screen above the taskbar
- **Local Brain** — `qwen3.5:4b` (multimodal, tool-calling) served by Ollama, entirely on your GPU. No API keys, no quotas, no network.
- **Voice Interaction** — Push-to-talk with GPU STT (faster-whisper `distil-large-v3`) and realtime **emotional** TTS (Chatterbox Turbo)
- **Emotional Voice** — Chatterbox's emotion-exaggeration dial is driven by the mood engine and per-clause tone inference; Kokoro is kept as an automatic fallback
- **Phoneme Lip-Sync** — Viseme-driven mouth animation synced to speech (misaki G2P)
- **Living Body** — Spring-bone physics (hair/skirt), mood-tied gestures, speech gesticulation, idle fidgets
- **Screen Awareness** — Captures and OCR-reads your screen on demand via the `look_at_screen` tool
- **Persistent Memory** — Short-term (SQLite) + long-term (Chroma vectors) with automatic background sync
- **Personality Engine** — Mood state machine + autonomous behaviours (glances, fidgets, mumbles), all LLM-free
- **Tool Calling** — Email, calendar, file search, web search via the local model's tool-use
- **Barge-In** — Interrupt Sylph mid-sentence; she stops and listens

## 🏗 Architecture

| Layer | Runtime | Role |
|---|---|---|
| Shell / Overlay | Tauri v2 (Rust + WebView) | Window management, hotkeys, mic capture, screen capture |
| Avatar | React + three.js | 3D rendering, lip-sync, expression blending |
| Brain Sidecar | Python 3.12 (FastAPI, `:8420`) | LLM orchestration, STT, TTS, memory, vision |
| Voice Worker | Python 3.12 (isolated venv) | Chatterbox TTS subprocess (see note below) |

### Why two Python environments

Chatterbox hard-pins `transformers==5.2.0` and `numpy<2.0`, which conflict with
the main sidecar's dependencies (Kokoro, faster-whisper). So Chatterbox lives in
a **dedicated venv** (`.venv-chatterbox`) and runs as a persistent subprocess
worker that the sidecar talks to over a small stdin/stdout protocol (audio is
passed as temp WAV files). The main venv is never perturbed, and if the worker
or its venv is missing the sidecar automatically falls back to Kokoro.

## 🖥 Target Hardware (verified)

| Component | Minimum | Notes |
|---|---|---|
| GPU | RTX 4060 (8 GB VRAM) | LLM + TTS + STT run **co-resident at ~7.5 GB / 8 GB** |
| RAM | 16 GB | |
| CPU | 6+ cores | |
| OS | Windows 10/11 | Screen capture + tray are Windows-first today |

Measured on an RTX 4060 Laptop (2026-09-09): qwen3.5:4b ≈ 3.4 GB, Chatterbox
Turbo ≈ 2.7 GB, distil-large-v3 (int8_float16) ≈ 1 GB. Chatterbox Turbo runs at
~0.8× realtime with ~1.8 s first-clause latency; the planner streams the LLM and
TTS through a decoupled queue so short clauses flow while the next is generated.

## 🚀 Getting Started

### Prerequisites

- **Python 3.12** (the `py -3.12` launcher on Windows)
- **Node.js 18+** and npm
- **Rust** (via rustup) — for the Tauri shell
- **Ollama** installed and running (the daemon serves the local model)
- **CUDA 12** GPU (RTX 4060-class) — used by the LLM, STT, and Chatterbox TTS

### Setup

```bash
# 1. Clone
git clone https://github.com/DevAtomicRelease/PROJECT-Sylph-v2.git
cd PROJECT-Sylph-v2

# 2. Main sidecar venv (CUDA torch FIRST, then the rest)
py -3.12 -m venv venv
venv\Scripts\activate                                   # Windows
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu124
pip install -r sidecar/requirements.txt
# Optional legacy engine (only if you set TTS_ENGINE=zonos):
#   git clone --depth 1 https://github.com/Zyphra/Zonos.git vendor/Zonos
#   pip install -e vendor/Zonos --no-deps

# 3. Chatterbox voice — DEDICATED venv (isolated from the main one)
py -3.12 -m venv .venv-chatterbox
.venv-chatterbox\Scripts\python -m pip install --upgrade pip
.venv-chatterbox\Scripts\pip install torch==2.6.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cu124
.venv-chatterbox\Scripts\pip install -r sidecar/requirements-chatterbox.txt

# 4. Frontend
cd tauri-app
npm install
cd ..

# 5. Local model (no sign-in, no cloud)
ollama pull qwen3.5:4b

# 6. Run (from tauri-app)
cd tauri-app
npm run tauri dev
```

On the **first run** the sidecar downloads its model weights once (faster-whisper
STT ≈ 1.5 GB, Chatterbox ≈ 2 GB, plus a one-time Chroma embedding model). It
also renders a one-time speaker reference for Chatterbox from Kokoro. Subsequent
starts are fast.

### Environment Variables

`.env` at the repo root (loaded by the sidecar at startup):

```env
MODEL_TAG=qwen3.5:4b                       # local multimodal brain
OLLAMA_BASE_URL=http://127.0.0.1:11434     # local Ollama daemon
TTS_ENGINE=chatterbox                      # chatterbox | kokoro | zonos
# CHATTERBOX_VARIANT=turbo                  # turbo (default) | nano | original
# CHATTERBOX_PYTHON=...\.venv-chatterbox\Scripts\python.exe   # override worker interpreter
```

The `.env` file documents fallbacks: `TTS_ENGINE=kokoro` needs no second venv;
`MODEL_TAG=gemma4:cloud` reverts the brain to Ollama's free cloud tier (needs a
signed-in daemon); `MODEL_TAG=qwen3.5:9b` is a bigger local brain if you free the
GPU by using Kokoro for voice.

### Google tools (optional)

Email and calendar tools need a Google OAuth client. Download `credentials.json`
from the Google Cloud Console and place it at the repo root; the first tool call
opens a browser consent flow and caches the token under `~/.sylph/config/`.
Without it, the other features (chat, voice, screen, memory) work normally.

## 🎮 Usage

| Action | How |
|---|---|
| Talk to Sylph | Press `Ctrl+Shift+Space` (push-to-talk) |
| Ask about screen | Say "read my screen" or "what's on my display" |
| Interrupt | Start talking while Sylph is speaking |
| Quiet mode | Toggle via system tray |

### Quick sidecar check (without the UI)

With the sidecar running (`python sidecar/main.py`):

```bash
curl http://127.0.0.1:8420/health          # {"vad":true,"stt":true,"tts":true,"ollama":true}
curl "http://127.0.0.1:8420/ask?text=say%20hello"
```

## 🔧 Troubleshooting

- **`/health` shows `ollama: false`** — the Ollama daemon isn't running, or
  `qwen3.5:4b` isn't pulled. Run `ollama pull qwen3.5:4b` and confirm
  `ollama ps`.
- **Voice is flat / no emotion** — the Chatterbox worker didn't start, so the
  sidecar fell back to Kokoro. Check `models/chatterbox/worker.log`, confirm
  `.venv-chatterbox` exists and `CHATTERBOX_PYTHON` points at it.
- **CUDA out of memory at startup** — close other GPU apps; the stack needs
  ~7.5 GB. As a lighter setup, use `TTS_ENGINE=kokoro` (frees ~2.7 GB) or
  `CHATTERBOX_VARIANT=nano`.
- **First run seems to hang** — it's downloading model weights (STT/TTS). Watch
  `sidecar.log`.

## 📜 License

Provided as-is for personal and educational use. See `LICENSE`.
