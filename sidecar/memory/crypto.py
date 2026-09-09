"""
Memory-at-rest encryption (Phase C+).

Encrypts the *sensitive text* Sylph persists — conversation messages
(short-term SQLite) and extracted facts (long-term Chroma documents) — so a
copy of the memory files on disk doesn't hand over their contents in plain
text. Structural fields (ids, timestamps, thread ids, vector embeddings) stay
clear so queries and lip-sync-free bookkeeping keep working.

Design notes:
- Symmetric AES (Fernet, from `cryptography`). The key is fetched from the OS
  keyring when `keyring` is installed (Windows Credential Manager / macOS
  Keychain / Secret Service); otherwise it falls back to a 0600 key file under
  ~/.sylph/config/. This is local-app protection, not defence against an
  attacker who already has the key — documented as such.
- Ciphertext is tagged with a version marker (`enc:v1:`). decrypt() returns
  anything without the marker unchanged, so pre-encryption (legacy plaintext)
  rows and encryption-disabled installs both read back fine — a zero-downtime,
  forward-compatible migration.
- Toggle with MEMORY_ENCRYPTION=off. If `cryptography` is missing the layer
  disables itself and logs once, rather than breaking memory.
"""

import logging
import os
from typing import Optional

logger = logging.getLogger("sylph.memory.crypto")

_MARKER = "enc:v1:"
_KEYRING_SERVICE = "sylph-memory"
_KEYRING_USER = "fernet-key"
_KEY_FILE = os.path.expanduser("~/.sylph/config/memory.key")


class MemoryCipher:
    """Lazy, fail-open text encryptor for the memory stores."""

    def __init__(self) -> None:
        self._fernet = None
        self._enabled = False
        self._resolved = False

    # -- setup ---------------------------------------------------------------

    def _resolve(self) -> None:
        if self._resolved:
            return
        self._resolved = True

        if os.environ.get("MEMORY_ENCRYPTION", "on").strip().lower() == "off":
            logger.info("Memory encryption disabled via MEMORY_ENCRYPTION=off")
            return

        try:
            from cryptography.fernet import Fernet
        except Exception:
            logger.warning("`cryptography` not installed — memory stored in plain text")
            return

        try:
            key = self._load_or_create_key()
            self._fernet = Fernet(key)
            self._enabled = True
            logger.info("Memory encryption active (key source: %s)", self._key_source)
        except Exception as e:
            logger.error("Memory encryption setup failed (%s) — storing plain text", e)

    _key_source = "unknown"

    def _load_or_create_key(self) -> bytes:
        from cryptography.fernet import Fernet

        # 1) OS keyring, if available.
        try:
            import keyring
            existing = keyring.get_password(_KEYRING_SERVICE, _KEYRING_USER)
            if existing:
                self._key_source = "keyring"
                return existing.encode()
            key = Fernet.generate_key()
            keyring.set_password(_KEYRING_SERVICE, _KEYRING_USER, key.decode())
            self._key_source = "keyring (new)"
            return key
        except Exception:
            pass  # no keyring backend — fall through to a key file

        # 2) Key file under the user config dir, 0600.
        if os.path.exists(_KEY_FILE):
            with open(_KEY_FILE, "rb") as f:
                self._key_source = "key file"
                return f.read().strip()

        os.makedirs(os.path.dirname(_KEY_FILE), exist_ok=True)
        key = Fernet.generate_key()
        with open(_KEY_FILE, "wb") as f:
            f.write(key)
        try:
            os.chmod(_KEY_FILE, 0o600)
        except OSError:
            pass  # best effort on Windows
        self._key_source = "key file (new)"
        return key

    @property
    def enabled(self) -> bool:
        self._resolve()
        return self._enabled

    # -- api -----------------------------------------------------------------

    def encrypt(self, text: Optional[str]) -> Optional[str]:
        """Return a tagged ciphertext, or the input unchanged when disabled."""
        self._resolve()
        if not self._enabled or not text:
            return text
        try:
            token = self._fernet.encrypt(text.encode("utf-8")).decode("ascii")
            return _MARKER + token
        except Exception as e:
            logger.warning("encrypt failed (%s) — storing plain text", e)
            return text

    def decrypt(self, text: Optional[str]) -> Optional[str]:
        """Decrypt a tagged ciphertext; pass through anything untagged."""
        if not text or not isinstance(text, str) or not text.startswith(_MARKER):
            return text
        self._resolve()
        if not self._enabled:
            # Encrypted data but no key — return the marker-stripped blob rather
            # than raising, so the app stays up.
            logger.warning("Found encrypted memory but encryption is unavailable")
            return text
        try:
            token = text[len(_MARKER):].encode("ascii")
            return self._fernet.decrypt(token).decode("utf-8")
        except Exception as e:
            logger.warning("decrypt failed (%s) — returning raw", e)
            return text


# Process-wide singleton.
cipher = MemoryCipher()
