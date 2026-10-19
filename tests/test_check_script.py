"""check_gemini.py: with no key it explains what to do, exits non-zero, and never crashes."""

import os
import shutil
import subprocess
import sys
import tempfile
import unittest

import helpers


class CheckScriptTests(unittest.TestCase):
    def test_without_a_key_it_explains_and_exits_with_an_error_code(self):
        env = {k: v for k, v in os.environ.items() if not k.startswith("GEMINI_")}
        with tempfile.TemporaryDirectory() as sandbox:
            # Run a COPY in an empty folder: the script looks for a .env next to itself, so
            # this guarantees a developer's real .env can never leak into the test (which
            # would make a real, billable Gemini request).
            for name in ("check_gemini.py", "rag_engine.py"):
                shutil.copy(os.path.join(helpers.ROOT, name), sandbox)
            proc = subprocess.run(
                [sys.executable, os.path.join(sandbox, "check_gemini.py")],
                cwd=sandbox, env=env, capture_output=True, text=True, timeout=60,
            )
        self.assertEqual(proc.returncode, 1)
        self.assertIn("GEMINI_API_KEY", proc.stdout)
        self.assertNotIn("Traceback", proc.stderr)


if __name__ == "__main__":
    unittest.main()
