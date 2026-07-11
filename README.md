# Project Sylph

A local-first AI desktop companion with a VRM 3D avatar, voice interaction, screen awareness, and persistent memory — running entirely on a single consumer machine with zero cloud dependencies.

## ✨ Features

- **3D Avatar Overlay** — VRM avatar rendered with three.js, always on-screen above the taskbar
- **Voice Interaction** — Push-to-talk with real-time STT (faster-whisper) and TTS (Kokoro)
- **Phoneme Lip-Sync** — Viseme-driven mouth animations synced to speech output
- **Screen Awareness** — Captures and OCR-reads your screen on demand via `look_at_screen` tool
- **Persistent Memory** — Short-term (SQLite) + long-term (JSON) memory with automatic sync
- **Personality Engine** — Mood state machine, autonomous behaviors (glances, fidgets, mumbles)
- **Tool Calling** — Email, calendar, file search, web search via Ollama tool-use
- **Barge-In Support** — Interrupt Sylph mid-sentence; she stops and listens
- **Fully Local** — Qwen3.5-9B via Ollama, no cloud APIs, $0 cost

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

- **Python 3.11+** with a virtual environment
- **Node.js 18+** and npm
- **Rust** (via rustup)
- **Ollama** running locally with `qwen3.5:9b` pulled
- **CUDA 12** (optional, for GPU-accelerated STT)

### Setup

```bash
# 1. Clone the repo
git clone https://github.com/DevAtomicRelease/PROJECT-Sylph.git
cd PROJECT-Sylph

# 2. Python sidecar
python -m venv venv
venv\Scripts\activate        # Windows
pip install -r sidecar/requirements.txt

# 3. Frontend
cd tauri-app
npm install

# 4. Pull the LLM model
ollama pull qwen3.5:9b

# 5. Run
npm run tauri dev
```

### Environment Variables

Copy `.env.example` to `.env` and configure:

```env
OLLAMA_BASE_URL=http://localhost:11434
STT_MODEL=distil-large-v3
TTS_VOICE=af_bella
```

## 🎮 Usage

| Action | How |
|---|---|
| Talk to Sylph | Press `Ctrl+Shift+Space` (push-to-talk) |
| Ask about screen | Say "read my screen" or "what's on my display" |
| Interrupt | Start talking while Sylph is speaking |
| Quiet mode | Toggle via system tray |

## 📜 License

This project is provided as-is for personal and educational use.
