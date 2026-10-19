"""Offline mode must work with NO Gemini SDK and NO network access.

This runs in a separate Python process where importing `google.*` is blocked and every
socket operation raises, then proves that questions are still answered locally."""

import os
import subprocess
import sys
import unittest

import helpers

SCRIPT = r'''
import socket, sys
sys.modules["google"] = None            # `import google...` now raises ImportError
sys.modules["google.genai"] = None

def blocked(*a, **k):
    raise RuntimeError("network access attempted")
socket.socket.connect = blocked
socket.create_connection = blocked
socket.getaddrinfo = blocked

sys.path.insert(0, ROOT)
from rag_engine import GeminiConfig, GeminiService, RAGEngine, read_folder

engine = RAGEngine()
engine.sync_documents(read_folder(SAMPLE))

# 1. no Gemini object at all, 2. a service with no key, 3. a service WITH a key but no SDK installed
services = [None, GeminiService(GeminiConfig(api_key=None)), GeminiService(GeminiConfig(api_key="AIzaFAKEKEY1234567890"))]
for svc in services:
    a = engine.answer("What is the late fee per day for library books?", gemini=svc)
    assert "INR 5" in a.text, a.text
    assert "Gemini (online)" not in a.label, a.label
    assert a.mode in ("offline", "offline_fallback"), a.mode
    assert a.sources and a.sources[0]["source"] == "sample_library_rules.pdf", a.sources
    assert engine.answer("hello", gemini=svc).mode == "smalltalk"
    assert engine.answer("What is the capital of France?", gemini=svc).mode == "no_match"

assert services[2].status().state == "disabled"          # SDK missing -> reported once, never retried
assert sys.modules.get("google.genai") is None            # the SDK was never imported on the offline paths
print("OFFLINE_OK")
'''


class OfflineWithoutNetworkOrSdkTests(unittest.TestCase):
    def test_local_rag_needs_no_sdk_and_no_network(self):
        code = SCRIPT.replace("ROOT", repr(helpers.ROOT)).replace("SAMPLE", repr(helpers.SAMPLE_DIR))
        proc = subprocess.run([sys.executable, "-I", "-c", code], capture_output=True, text=True,
                              cwd=os.path.dirname(helpers.ROOT), timeout=120)
        self.assertEqual(proc.returncode, 0, msg=proc.stdout + proc.stderr)
        self.assertIn("OFFLINE_OK", proc.stdout)


if __name__ == "__main__":
    unittest.main()
