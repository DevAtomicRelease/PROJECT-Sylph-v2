"""Put the sidecar package root on sys.path so tests can import its modules
(llm, tts, memory, tools) the same way main.py does when run from sidecar/."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Keep the memory round-trip tests hermetic: the default-on encryption would
# otherwise create a real key file under ~/.sylph/config during tests. The
# encryption path itself is covered explicitly in test_crypto.py with a
# temp key file.
os.environ.setdefault("MEMORY_ENCRYPTION", "off")
