"""
Agent permissions — what Sylph is allowed to do (Phase 2).

A single JSON file (~/.sylph/config/permissions.json) is the source of truth for
which capability DOMAINS are enabled and, for the files domain, which directories
are reachable. The planner consults this every turn: disabled domains' tools are
neither advertised to the model nor executed, and file access can be restricted
to an allowlist. The Dashboard's Controls tab edits this file via the sidecar's
REST endpoints.

Design: non-destructive + fail-safe. Unknown tools are allowed (e.g. the avatar
moving its own body is never gated); a missing/corrupt config falls back to the
defaults below rather than locking Sylph out.
"""

import json
import logging
import os
import threading
from typing import Optional

logger = logging.getLogger("sylph.permissions")

_CONFIG_PATH = os.path.expanduser("~/.sylph/config/permissions.json")

# tool name -> capability domain. Tools not listed here are ungated (always
# allowed) — e.g. move_avatar, which is Sylph controlling her own body.
TOOL_DOMAINS: dict[str, str] = {
    "list_recent_emails": "comms",
    "read_email": "comms",
    "draft_email": "comms",
    "send_email": "comms",
    "list_events_today": "comms",
    "list_events_range": "comms",
    "create_event": "comms",
    "delete_event": "comms",
    "search_web": "web",
    "read_page": "web",
    "list_files": "files",
    "read_file": "files",
    "search_files": "files",
    "look_at_screen": "screen",
    "open_app": "system",
    "focus_app": "system",
    "list_open_windows": "system",
}

# The path-bearing argument for each file tool, checked against allowed_dirs.
_FILE_TOOL_PATH_ARG = {
    "list_files": "directory",
    "read_file": "file_path",
    "search_files": "directory",
}

DEFAULT_CONFIG = {
    "version": 1,
    "domains": {
        "comms": {"enabled": True},
        "web": {"enabled": True},
        "files": {"enabled": True, "allowed_dirs": []},  # [] = unrestricted
        "screen": {"enabled": True},
        "system": {"enabled": True},
    },
}

# Human-facing metadata for the Controls UI (labels + which tools each covers).
DOMAIN_META = {
    "comms":  {"label": "Email & Calendar", "tools": "read/draft/send mail, calendar"},
    "web":    {"label": "Web",              "tools": "search, read pages"},
    "files":  {"label": "Files",            "tools": "list, read, search files"},
    "screen": {"label": "Screen",           "tools": "capture + read your screen"},
    "system": {"label": "Apps & System",    "tools": "open / focus apps, list open windows"},
}


class PermissionStore:
    """Thread-safe loader/saver for the permissions config."""

    def __init__(self, path: str = _CONFIG_PATH):
        self._path = path
        self._lock = threading.Lock()
        self._config = self._load()

    def _load(self) -> dict:
        try:
            with open(self._path, "r", encoding="utf-8") as f:
                cfg = json.load(f)
            return self._merge_defaults(cfg)
        except FileNotFoundError:
            return json.loads(json.dumps(DEFAULT_CONFIG))
        except Exception as e:
            logger.warning("permissions.json unreadable (%s) — using defaults", e)
            return json.loads(json.dumps(DEFAULT_CONFIG))

    @staticmethod
    def _merge_defaults(cfg: dict) -> dict:
        """Fill any missing domain/keys from defaults so partial files are safe."""
        merged = json.loads(json.dumps(DEFAULT_CONFIG))
        for dom, vals in (cfg.get("domains") or {}).items():
            if dom in merged["domains"] and isinstance(vals, dict):
                merged["domains"][dom].update(vals)
        return merged

    def get(self) -> dict:
        with self._lock:
            return json.loads(json.dumps(self._config))

    def update(self, new_cfg: dict) -> dict:
        with self._lock:
            self._config = self._merge_defaults(new_cfg)
            try:
                os.makedirs(os.path.dirname(self._path), exist_ok=True)
                with open(self._path, "w", encoding="utf-8") as f:
                    json.dump(self._config, f, indent=2)
                logger.info("permissions updated: %s", {d: v.get("enabled") for d, v in self._config["domains"].items()})
            except Exception as e:
                logger.error("failed to persist permissions: %s", e)
            return json.loads(json.dumps(self._config))

    # -- queries used by the planner -----------------------------------------

    def is_domain_enabled(self, domain: str) -> bool:
        return bool(self._config["domains"].get(domain, {}).get("enabled", True))

    def is_tool_allowed(self, tool_name: str) -> bool:
        domain = TOOL_DOMAINS.get(tool_name)
        if domain is None:
            return True  # ungated tool
        return self.is_domain_enabled(domain)

    def is_path_allowed(self, path: str) -> bool:
        """For the files domain: True if allowed_dirs is empty (unrestricted) or
        `path` resolves under one of them."""
        allowed = self._config["domains"].get("files", {}).get("allowed_dirs") or []
        if not allowed:
            return True
        try:
            p = os.path.realpath(os.path.expanduser(path))
        except Exception:
            return False
        for d in allowed:
            base = os.path.realpath(os.path.expanduser(d))
            if p == base or p.startswith(base + os.sep):
                return True
        return False

    def check_tool(self, tool_name: str, args: dict) -> Optional[str]:
        """Return None if allowed, else a short reason string to relay."""
        if not self.is_tool_allowed(tool_name):
            domain = TOOL_DOMAINS.get(tool_name, "that")
            label = DOMAIN_META.get(domain, {}).get("label", domain)
            return f"The '{label}' capability is turned off in your settings."
        path_arg = _FILE_TOOL_PATH_ARG.get(tool_name)
        if path_arg and args.get(path_arg):
            if not self.is_path_allowed(str(args[path_arg])):
                return "That folder is outside the allowed directories in your settings."
        return None


# Process-wide singleton.
permissions = PermissionStore()
