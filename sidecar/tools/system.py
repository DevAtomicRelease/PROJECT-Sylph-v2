"""
System control tools — Phase 3.

Non-destructive desktop actions: open an app, focus an open window, and list
what's open. There is deliberately NO close/kill here — those are destructive
and were cut from the plan. Gated by the 'system' capability domain, so the
user can turn it off from the Dashboard Controls tab.

Heavy imports (AppOpener scans the Start menu once on import) are done lazily
inside the functions so they don't slow sidecar startup.
"""

import logging

logger = logging.getLogger("sylph.tools.system")


def open_app(name: str) -> str:
    """
    Open / launch an application by name (e.g. "chrome", "notepad", "spotify").
    Matches the closest installed app. Non-destructive.
    """
    name = (name or "").strip()
    if not name:
        return "No app name given."
    try:
        from AppOpener import open as _open
        _open(name, match_closest=True, throw_error=True, output=False)
        logger.info("Opened app '%s'", name)
        return f"Opened '{name}'."
    except Exception as e:
        logger.warning("open_app('%s') failed: %s", name, e)
        return f"Couldn't open '{name}'. It may not be installed."


def focus_app(title: str) -> str:
    """
    Bring an already-open window to the front by a (partial) title match,
    e.g. "chrome", "code", "spotify".
    """
    title = (title or "").strip()
    if not title:
        return "No window title given."
    try:
        import pygetwindow as gw
        wins = [w for w in gw.getWindowsWithTitle(title) if w.title.strip()]
        if not wins:
            return f"No open window matching '{title}'."
        w = wins[0]
        try:
            if w.isMinimized:
                w.restore()
        except Exception:
            pass
        try:
            w.activate()
        except Exception:
            # Windows sometimes refuses activate() unless the app has focus
            # rights; minimize+restore is a reliable fallback that raises it.
            try:
                w.minimize(); w.restore()
            except Exception:
                pass
        logger.info("Focused window '%s'", w.title)
        return f"Focused '{w.title}'."
    except Exception as e:
        logger.warning("focus_app('%s') failed: %s", title, e)
        return f"Couldn't focus '{title}': {e}"


def list_open_windows() -> list[str]:
    """List the titles of currently open application windows."""
    try:
        import pygetwindow as gw
        seen = set()
        out = []
        for t in gw.getAllTitles():
            t = (t or "").strip()
            if t and t not in seen:
                seen.add(t)
                out.append(t)
        return out[:40]
    except Exception as e:
        logger.warning("list_open_windows failed: %s", e)
        return []
