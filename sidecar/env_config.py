"""
Minimal .env loader for the sidecar.

The repo-root .env promises "the sidecar reads this at startup" — this module
makes that true regardless of how the process was launched (Tauri shell,
terminal, service). Values already present in the process environment win, so
`OLLAMA_API_KEY=... python main.py` still overrides the file.

Import this before any module that reads os.environ at import time.
"""

import logging
import os

logger = logging.getLogger("sylph.env")

_WORKSPACE_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_ENV_PATH = os.path.join(_WORKSPACE_ROOT, ".env")


def load_env_file(path: str = _ENV_PATH) -> int:
    """Load KEY=VALUE lines into os.environ (existing vars win). Returns count loaded."""
    loaded = 0
    try:
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                key, _, value = line.partition("=")
                key = key.strip()
                value = value.strip().strip('"').strip("'")
                if key and key not in os.environ:
                    os.environ[key] = value
                    loaded += 1
    except FileNotFoundError:
        pass
    except Exception as e:
        logger.warning("Failed to read %s: %s", path, e)
    return loaded


# Load on import so `import env_config` at the top of main.py is enough.
load_env_file()
