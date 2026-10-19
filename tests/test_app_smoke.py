"""Runs the real app.py top to bottom, several times (like Streamlit reruns), against a
STAND-IN for Streamlit and a scripted fake Gemini client.

What this proves: the app's own logic (answer once, no duplicates on rerun, no re-indexing
when nothing changed, document removal, honest status labels, status refresh after a failure).
What it does NOT prove: that real Streamlit renders the widgets correctly. Check that by
running `streamlit run app.py` yourself (see README, "Manual checks")."""

import os
import runpy
import sys
import types
import unittest
from unittest import mock

import helpers
from helpers import FakeAPIError, ScriptedClient

APP = os.path.join(helpers.ROOT, "app.py")
QUESTION = "What is the late fee per day for library books?"


class _Ctx:
    def __init__(self, on_enter=None):
        self._on_enter = on_enter

    def __enter__(self):
        if self._on_enter:
            self._on_enter()
        return self

    def __exit__(self, *exc):
        return False


class _Slot:
    """Like st.empty(): each time you open it, its previous contents are replaced."""

    def __init__(self, fake):
        self._fake, self.contents = fake, []

    def container(self):
        def enter():
            self.contents = []
            self._fake.slot = self
        return _SlotCtx(self._fake, enter)


class _SlotCtx(_Ctx):
    def __exit__(self, *exc):
        self._fake_exit()
        return False

    def __init__(self, fake, on_enter):
        super().__init__(on_enter)
        self._fake = fake

    def _fake_exit(self):
        self._fake.slot = None


class _Session(dict):
    def __getattr__(self, k):
        try:
            return self[k]
        except KeyError:
            raise AttributeError(k)

    def __setattr__(self, k, v):
        self[k] = v


class FakeStreamlit(types.ModuleType):
    def __init__(self):
        super().__init__("streamlit")
        self.session_state = _Session()
        self.secrets = types.SimpleNamespace(get=lambda name, default=None: default)
        self.uploads = []        # files "uploaded" in the sidebar
        self.questions = []      # questions the user "types"
        self.rendered = []       # (role, kind, text) drawn during the current run
        self.slot = None
        self.role = None
        self.sidebar = _Ctx()
        self._cache = {}

    # --- caching (per fake instance, like a fresh server) ---
    def cache_resource(self, *a, **k):
        def deco(fn):
            def wrapper(*args, **kw):
                key = (fn.__name__, args[:1])
                if key not in self._cache:
                    self._cache[key] = fn(*args, **kw)
                return self._cache[key]
            return wrapper
        return deco

    def cache_data(self, *a, **k):
        def deco(fn):
            def wrapper(*args, **kw):
                key = (fn.__name__, args)
                if key not in self._cache:
                    self._cache[key] = fn(*args, **kw)
                return self._cache[key]
            return wrapper
        return deco

    # --- output ---
    def _out(self, kind, text=""):
        if self.slot is not None:
            self.slot.contents.append(str(text))
        self.rendered.append((self.role, kind, str(text)))

    def set_page_config(self, **k): pass
    def title(self, t): self._out("title", t)
    def caption(self, t): self._out("caption", t)
    def header(self, t): self._out("header", t)
    def divider(self): pass
    def markdown(self, t, **k): self._out("markdown", t)
    def warning(self, t, **k): self._out("warning", t)
    def info(self, t, **k): self._out("info", t)
    def success(self, t, **k): self._out("success", t)
    def error(self, t, **k): self._out("error", t)
    def write(self, t): self._out("write", t)
    def spinner(self, *a, **k): return _Ctx()
    def expander(self, *a, **k): return _Ctx()
    def empty(self): return _Slot(self)

    def chat_message(self, role):
        return _Ctx(lambda: setattr(self, "role", role))

    # --- input ---
    def slider(self, label, **k): return k.get("value", 3)
    def file_uploader(self, *a, **k): return list(self.uploads)
    def checkbox(self, label, value=False, **k): return value
    def chat_input(self, *a, **k): return self.questions.pop(0) if self.questions else None

    # --- helpers for tests ---
    def rerun(self):
        self.rendered, self.role, self.slot = [], None, None
        runpy.run_path(APP, run_name="__main__")

    def texts(self, kind=None, role=None):
        return [t for r, k, t in self.rendered if (kind is None or k == kind) and (role is None or r == role)]


class Upload:
    def __init__(self, name, text):
        self.name, self._data = name, text.encode("utf-8")

    def getvalue(self):
        return self._data


class AppSmokeTests(unittest.TestCase):
    def setUp(self):
        self.st = FakeStreamlit()
        self.client = ScriptedClient()
        genai = types.ModuleType("google.genai")
        genai.Client = lambda api_key=None, http_options=None: self.client
        google = types.ModuleType("google")
        google.genai = genai
        dotenv = types.ModuleType("dotenv")
        dotenv.load_dotenv = lambda *a, **k: False  # never read a developer's real .env in tests
        patches = [
            mock.patch.dict(sys.modules, {"streamlit": self.st, "google": google,
                                          "google.genai": genai, "dotenv": dotenv}),
            mock.patch.dict(os.environ, {}, clear=False),
        ]
        for p in patches:
            p.start()
            self.addCleanup(p.stop)
        for name in ("GEMINI_API_KEY", "GEMINI_MODEL", "GEMINI_API", "RAG_MIN_SCORE"):
            os.environ.pop(name, None)

    def ask(self, question):
        self.st.questions.append(question)
        self.st.rerun()

    def chat(self):
        return self.st.session_state.chat

    # ------------------------------------------------------------------ offline
    def test_no_key_answers_offline_once_and_reruns_do_not_duplicate_or_reindex(self):
        self.ask(QUESTION)
        msgs = self.chat().messages
        self.assertEqual([m.role for m in msgs], ["user", "assistant"])
        label = msgs[1].meta["label"]
        self.assertIn("Local RAG (offline)", label)
        self.assertNotIn("Gemini (online)", label)
        self.assertIn("INR 5 per day", msgs[1].content)

        builds = self.st.session_state.engine.index_builds
        for _ in range(3):  # plain reruns (e.g. moving a slider): nothing new should happen
            self.st.rerun()
        self.assertEqual(len(self.chat().messages), 2)
        self.assertEqual(self.st.session_state.engine.index_builds, builds)
        answer_text = msgs[1].content.replace("$", "\\$")  # app.py escapes $ before drawing
        drawn = [t for t in self.st.texts("markdown", "assistant") if t == answer_text]
        self.assertEqual(len(drawn), 1)  # the answer is drawn once per run, never twice
        self.assertEqual(self.client.calls, [])  # no Gemini request without a key

    def test_greeting_is_answered_locally_without_search_or_gemini(self):
        os.environ["GEMINI_API_KEY"] = helpers.TEST_KEY
        self.ask("hello")
        self.assertEqual(self.chat().messages[1].meta["mode"], "smalltalk")
        self.assertEqual(self.client.calls, [])

    # ------------------------------------------------------------------- online
    def test_gemini_answer_is_labelled_gemini_and_sidebar_status_updates_after_it(self):
        os.environ["GEMINI_API_KEY"] = helpers.TEST_KEY
        self.client.script = ["The late fee is INR 5 per day per book [1]."]
        self.ask(QUESTION)
        reply = self.chat().messages[1]
        self.assertEqual(reply.meta["mode"], "gemini")
        self.assertIn("Gemini (online)", reply.meta["label"])
        self.assertEqual(len(self.client.calls), 1)
        self.assertIn("The late fee is INR 5 per day per book [1].", reply.content)
        # the sidebar status was redrawn AFTER the answer and now says online
        self.assertTrue(any("Gemini online" in t for t in self.st.texts("caption")))

    def test_invalid_key_falls_back_labels_it_honestly_and_is_not_retried(self):
        os.environ["GEMINI_API_KEY"] = helpers.TEST_KEY
        self.client.script = [FakeAPIError(400, "INVALID_ARGUMENT", "API key not valid. Please pass a valid API key.")]
        self.ask(QUESTION)
        reply = self.chat().messages[1]
        self.assertEqual(reply.meta["mode"], "offline_fallback")
        self.assertNotIn("Gemini (online)", reply.meta["label"])
        self.assertIn("rejected the API key", reply.meta["notice"])
        self.assertNotIn(helpers.TEST_KEY, repr(self.chat().messages))  # key never ends up in chat data
        self.assertTrue(any("Gemini unavailable" in t for t in self.st.texts("caption")))  # status refreshed
        self.ask("What is the maximum notice period for full-time staff?")
        self.assertEqual(len(self.client.calls), 1)  # permanent error: no second request

    def test_network_outage_falls_back_then_pauses_without_hammering(self):
        os.environ["GEMINI_API_KEY"] = helpers.TEST_KEY
        self.client.script = [ConnectionError("Connection refused")] * 2  # first try + one bounded retry
        with mock.patch("time.sleep"):
            self.ask(QUESTION)
        self.assertEqual(self.chat().messages[1].meta["mode"], "offline_fallback")
        self.assertIn("network", self.chat().messages[1].meta["notice"].lower())
        calls_after_first = len(self.client.calls)
        self.assertLessEqual(calls_after_first, 2)
        self.ask(QUESTION)  # inside the cooldown: answered locally, no new request
        self.assertEqual(len(self.client.calls), calls_after_first)

    # ---------------------------------------------------------------- documents
    def test_uploaded_document_can_be_answered_and_then_removed(self):
        self.st.uploads = [Upload("zebra_notes.txt", "The zebra enclosure opens at nine in the morning daily.")]
        self.ask("When does the zebra enclosure open?")
        engine = self.st.session_state.engine
        self.assertIn("zebra_notes.txt", [n for n, _ in engine.document_summary()])
        self.assertEqual(self.chat().messages[1].meta["sources"][0]["source"], "zebra_notes.txt")

        self.st.uploads = []  # the user removes the file
        self.st.rerun()
        self.assertNotIn("zebra_notes.txt", [n for n, _ in engine.document_summary()])
        self.assertFalse(any(c.source == "zebra_notes.txt" for c in engine.chunks))
        self.assertGreater(len(engine.chunks), 0)  # the rest of the index is intact
        self.assertEqual(len(self.chat().messages), 2)  # chat history survived the change

    def test_two_uploads_cite_their_own_files(self):
        self.st.uploads = [Upload("a_menu.txt", "The cafe sells masala chai for forty rupees."),
                           Upload("b_gym.txt", "The gym membership costs fifteen hundred rupees monthly.")]
        self.ask("How much is the gym membership?")
        self.assertEqual(self.chat().messages[1].meta["sources"][0]["source"], "b_gym.txt")
        self.ask("What does masala chai cost?")
        self.assertEqual(self.chat().messages[3].meta["sources"][0]["source"], "a_menu.txt")


if __name__ == "__main__":
    unittest.main()
