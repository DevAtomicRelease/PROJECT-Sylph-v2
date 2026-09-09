# Project Sylph

A local-first AI desktop companion with a VRM 3D avatar, voice interaction, screen awareness, and persistent memory — running entirely on a single consumer machine with zero cloud dependencies.

## ✨ Features

- **3D Avatar Overlay** — VRM avatar rendered with three.js, always on-screen above the taskbar
- **Voice Interaction** — Push-to-talk with real-time STT (faster-whisper) and emotional TTS (Zonos)
- **Emotional Voice** — Zonos emotion vector (happiness/sadness/anger/fear/surprise/…) driven by the mood engine and per-clause tone inference; Kokoro kept as automatic fallback
- **Phoneme Lip-Sync** — Viseme-driven mouth animations synced to speech output (misaki G2P)
- **Living Body** — Spring-bone physics (hair/skirt/bust), mood-tied gestures (happy bounce, slump, shy turn-away, hum sway), speech gesticulation, idle fidgets
- **Screen Awareness** — Captures and OCR-reads your screen on demand via `look_at_screen` tool
- **Persistent Memory** — Short-term (SQLite) + long-term (JSON) memory with automatic sync
- **Personality Engine** — Mood state machine, autonomous behaviors (glances, fidgets, mumbles) — all LLM-free
- **Tool Calling** — Email, calendar, file search, web search via Ollama tool-use
- **Barge-In Support** — Interrupt Sylph mid-sentence; she stops and listens
- **$0 Stack** — LLM on Ollama Cloud's free tier (`gemma4:cloud`, multimodal); STT + TTS + memory fully local. The local GPU is dedicated to voice, not the LLM.

## 🏗 Architecture

| Layer | Runtime | Role |
|---|---|---|
| Shell / Overlay | Tauri v2 (Rust + WebView) | Window management, hotkeys, mic capture, screen capture |
| Avatar | React + three.js | 3D rendering, lip-sync, expression blending |
| Brain Sidecar | Python 3.12 (FastAPI) | LLM orchestration, STT, TTS, memory, vision |

## 🖥 Target Hardware

| Component | Minimum |
|---|---|
| GPU | RTX 4060 (8 GB VRAM) |
| RAM | 16 GB |
| CPU | 6+ cores |
| OS | Windows 10 2004+ |

## 📁 Project Structure

```
project-sylph/
├── tauri-app/               # Tauri v2 + React frontend
│   ├── src/                 # React components, services, stores
│   └── src-tauri/           # Rust backend (window, hotkeys, capture)
├── sidecar/                 # Python brain sidecar
│   ├── llm/                 # Ollama client, planner, prompts
│   ├── stt/                 # VAD + faster-whisper transcriber
│   ├── tts/                 # Kokoro synthesizer + viseme builder
│   ├── memory/              # Short-term + long-term memory
│   ├── personality/         # Mood, autonomous behavior, interrupt gate
│   ├── vision/              # Screen analyzer (OCR + triage)
│   ├── tools/               # Email, calendar, file, web tools
│   └── main.py              # FastAPI entry point
├── assets/                  # VRM model, animations, config files
│   ├── avatar/              # VRM 3D model + animations
│   ├── audio/               # Mumble lines
│   └── config/              # Viseme map, personality config
└── .env                     # Environment variables
```

## 🚀 Getting Started

### Prerequisites

- **Python 3.12** with a virtual environment
- **Node.js 18+** and npm
- **Rust** (via rustup)
- **Ollama** installed and signed in (`ollama signin`) — the daemon proxies the free-tier cloud model
- **CUDA 12** GPU (RTX 4060-class) — used by Zonos TTS and faster-whisper

### Setup

```bash
# 1. Clone the repo
git clone https://github.com/DevAtomicRelease/PROJECT-Sylph.git
cd PROJECT-Sylph

# 2. Python sidecar (CUDA torch FIRST, then requirements, then Zonos editable)
python -m venv venv
venv\Scripts\activate        # Windows
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu124
pip install -r sidecar/requirements.txt
git clone --depth 1 https://github.com/Zyphra/Zonos.git vendor/Zonos
pip install -e vendor/Zonos --no-deps

# 3. Frontend
cd tauri-app
npm install

# 4. Cloud LLM (free tier, proxied through the signed-in daemon)
ollama signin
ollama pull gemma4:cloud     # manifest-only stub, no local weights

# 5. Run
npm run tauri dev
```

### Environment Variables

`.env` at the repo root (loaded by the sidecar at startup):

```env
MODEL_TAG=gemma4:cloud                     # free-tier multimodal cloud model
OLLAMA_BASE_URL=http://127.0.0.1:11434     # signed-in daemon proxies to cloud
TTS_ENGINE=zonos                           # zonos | kokoro
ZONOS_MODEL=Zyphra/Zonos-v0.1-transformer
```

See the comments in `.env` for the direct-API alternative (`https://ollama.com`
+ `OLLAMA_API_KEY`) and for reverting to a fully local LLM.

## 🎮 Usage

| Action | How |
|---|---|
| Talk to Sylph | Press `Ctrl+Shift+Space` (push-to-talk) |
| Ask about screen | Say "read my screen" or "what's on my display" |
| Interrupt | Start talking while Sylph is speaking |
| Quiet mode | Toggle via system tray |

## 📜 License

This project is provided as-is for personal and educational use.
