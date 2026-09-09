"""Memory-at-rest encryption: round-trip, tagging, and fail-open behaviour."""
import memory.crypto as crypto_mod
from memory.crypto import MemoryCipher, _MARKER


def test_disabled_is_passthrough(monkeypatch):
    monkeypatch.setenv("MEMORY_ENCRYPTION", "off")
    c = MemoryCipher()
    assert c.enabled is False
    assert c.encrypt("secret") == "secret"
    assert c.decrypt("secret") == "secret"


def test_enabled_roundtrip_with_keyfile(monkeypatch, tmp_path):
    monkeypatch.setenv("MEMORY_ENCRYPTION", "on")
    monkeypatch.setattr(crypto_mod, "_KEY_FILE", str(tmp_path / "memory.key"))
    c = MemoryCipher()
    assert c.enabled is True

    token = c.encrypt("my private note")
    assert token.startswith(_MARKER)
    assert "my private note" not in token          # actually encrypted at rest
    assert c.decrypt(token) == "my private note"    # recovers exactly


def test_untagged_text_passes_through_even_when_enabled(monkeypatch, tmp_path):
    monkeypatch.setenv("MEMORY_ENCRYPTION", "on")
    monkeypatch.setattr(crypto_mod, "_KEY_FILE", str(tmp_path / "memory.key"))
    c = MemoryCipher()
    # legacy plaintext row (no marker) reads back unchanged
    assert c.decrypt("legacy plaintext fact") == "legacy plaintext fact"


def test_corrupt_ciphertext_fails_open(monkeypatch, tmp_path):
    monkeypatch.setenv("MEMORY_ENCRYPTION", "on")
    monkeypatch.setattr(crypto_mod, "_KEY_FILE", str(tmp_path / "memory.key"))
    c = MemoryCipher()
    garbage = _MARKER + "not-a-valid-token"
    # must not raise — returns the raw value rather than crashing memory reads
    assert c.decrypt(garbage) == garbage


def test_empty_and_none(monkeypatch, tmp_path):
    monkeypatch.setenv("MEMORY_ENCRYPTION", "on")
    monkeypatch.setattr(crypto_mod, "_KEY_FILE", str(tmp_path / "memory.key"))
    c = MemoryCipher()
    assert c.encrypt("") == ""
    assert c.encrypt(None) is None
    assert c.decrypt(None) is None
