"""Put the sidecar package root on sys.path so tests can import its modules
(llm, tts, memory, tools) the same way main.py does when run from sidecar/."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
