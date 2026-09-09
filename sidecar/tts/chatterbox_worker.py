"""
Chatterbox TTS worker — runs INSIDE the dedicated .venv-chatterbox.

Loads Chatterbox on CUDA once, then serves one clause per stdin request. The
parent (sidecar/tts/chatterbox_synthesizer.py, in the main venv) speaks to it
over a tiny line protocol:

  stdout (protocol only, marker-prefixed):
    @@READY@@ {"sr": 24000}                     once, after the model loads
    @@RESP@@  {"ok": true,  "sr": 24000}        per generate (wav at out_path)
    @@RESP@@  {"ok": false, "error": "..."}     on failure

  stdin (one JSON object per line):
    {"text","exaggeration","cfg_weight","audio_prompt_path","out_path"}
    {"cmd": "quit"}

Every library print is redirected to stderr so it can never corrupt the
protocol on stdout. stderr is captured to models/chatterbox/worker.log by the
parent.
"""

import json
import sys

# Redirect ALL library stdout to stderr; keep the real stdout for protocol only.
_real_stdout = sys.stdout
sys.stdout = sys.stderr

import logging
logging.basicConfig(stream=sys.stderr, level=logging.INFO,
                    format="%(asctime)s | chatterbox-worker | %(levelname)s | %(message)s")
log = logging.getLogger("chatterbox.worker")


def emit(marker: str, obj: dict) -> None:
    _real_stdout.write(f"{marker} {json.dumps(obj)}\n")
    _real_stdout.flush()


def main() -> int:
    import os
    # Variant: turbo (default) is ~2x faster than original on an RTX 4060 with
    # half the first-clause latency and still honours exaggeration/cfg_weight
    # (benchmarked 2026-09-09). nano trades quality for more speed; original is
    # highest quality but slowest.
    variant = os.environ.get("CHATTERBOX_VARIANT", "turbo").strip().lower()

    try:
        import torch
        import torchaudio
        if variant == "original":
            from chatterbox.tts import ChatterboxTTS
            load = lambda: ChatterboxTTS.from_pretrained(device="cuda")
        elif variant == "nano":
            # `nano` is passed only for this variant; some releases expose it as
            # a from_pretrained kwarg, others don't — fall back to plain Turbo.
            from chatterbox.tts_turbo import ChatterboxTurboTTS
            def load():
                try:
                    return ChatterboxTurboTTS.from_pretrained(device="cuda", nano=True)
                except TypeError:
                    log.warning("nano kwarg unsupported in this build — using plain Turbo")
                    return ChatterboxTurboTTS.from_pretrained(device="cuda")
        else:  # turbo (default)
            from chatterbox.tts_turbo import ChatterboxTurboTTS
            load = lambda: ChatterboxTurboTTS.from_pretrained(device="cuda")
    except Exception as e:
        log.exception("import failed")
        emit("@@RESP@@", {"ok": False, "error": f"import failed: {e}"})
        return 1

    if not torch.cuda.is_available():
        emit("@@RESP@@", {"ok": False, "error": "CUDA not available in worker venv"})
        return 1

    try:
        log.info("loading Chatterbox variant='%s' on cuda...", variant)
        model = load()
        sr = int(getattr(model, "sr", 24000))
        log.info("model loaded (variant=%s, sr=%d)", variant, sr)
    except Exception as e:
        log.exception("model load failed")
        emit("@@RESP@@", {"ok": False, "error": f"model load failed: {e}"})
        return 1

    emit("@@READY@@", {"sr": sr})

    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            req = json.loads(line)
        except json.JSONDecodeError:
            emit("@@RESP@@", {"ok": False, "error": "bad json request"})
            continue

        if req.get("cmd") == "quit":
            log.info("quit requested")
            break

        text = (req.get("text") or "").strip()
        out_path = req.get("out_path")
        if not text or not out_path:
            emit("@@RESP@@", {"ok": False, "error": "missing text or out_path"})
            continue

        kwargs = {
            "exaggeration": float(req.get("exaggeration", 0.5)),
            "cfg_weight": float(req.get("cfg_weight", 0.5)),
        }
        ref = req.get("audio_prompt_path")
        if ref:
            kwargs["audio_prompt_path"] = ref

        try:
            with torch.inference_mode():
                wav = model.generate(text, **kwargs)
            if hasattr(wav, "detach"):
                wav = wav.detach().cpu()
                if wav.dim() == 1:
                    wav = wav.unsqueeze(0)   # torchaudio wants [channels, time]
            torchaudio.save(out_path, wav, sr)
            emit("@@RESP@@", {"ok": True, "sr": sr})
        except Exception as e:
            log.exception("generate failed")
            emit("@@RESP@@", {"ok": False, "error": str(e)})

    return 0


if __name__ == "__main__":
    sys.exit(main())
