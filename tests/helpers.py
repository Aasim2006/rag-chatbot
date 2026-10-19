"""Shared test helpers: fake Gemini clients, a fake clock, and sample documents."""

import os
import sys
import types

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
SAMPLE_DIR = os.path.join(ROOT, "sample_docs")

from rag_engine import (ErrorKind, GeminiConfig, GeminiResult, GeminiService,  # noqa: E402
                        RAGEngine, read_folder)

TEST_KEY = "AIzaSyTESTKEY1234567890abcdef"


class FakeAPIError(Exception):
    """Looks like google.genai.errors.APIError: has .code, .status, .message."""

    def __init__(self, code, status, message):
        super().__init__(f"{code} {status}. {message}")
        self.code, self.status, self.message = code, status, message


class FakeClock:
    def __init__(self):
        self.now = 1000.0

    def __call__(self):
        return self.now

    def advance(self, seconds):
        self.now += seconds


class ScriptedClient:
    """A fake google-genai client. Each request consumes one item from `script`:
    a string is returned as the answer, an exception is raised, a callable is called."""

    def __init__(self, script=(), style="both"):
        self.script = list(script)
        self.calls = []
        if style in ("both", "interactions"):
            self.interactions = types.SimpleNamespace(create=self._interactions_create)
        if style in ("both", "generate_content"):
            self.models = types.SimpleNamespace(generate_content=self._generate_content)

    def _next(self, kwargs):
        self.calls.append(kwargs)
        item = self.script.pop(0) if self.script else "DEFAULT ANSWER"
        if callable(item):
            item = item()
        if isinstance(item, BaseException):
            raise item
        return item

    def _interactions_create(self, **kw):
        return types.SimpleNamespace(output_text=self._next(kw))

    def _generate_content(self, **kw):
        return types.SimpleNamespace(text=self._next(kw))


def make_service(script=(), style="both", api_key=TEST_KEY, **config_kwargs):
    """Returns (service, client, clock, sleeps). No real SDK or network is used."""
    client = ScriptedClient(script, style)
    clock = FakeClock()
    sleeps = []
    config = GeminiConfig(api_key=api_key, **config_kwargs)
    service = GeminiService(config, client_factory=lambda cfg: client, clock=clock, sleep=sleeps.append)
    return service, client, clock, sleeps


class FakeGemini:
    """Stands in for GeminiService when testing RAGEngine.answer()."""

    def __init__(self, text=None, error_kind=None, user_message=None, configured=True, raises=None):
        self.is_configured = configured
        self.model = "gemini-test"
        self.prompts = []
        self._text, self._kind, self._msg, self._raises = text, error_kind, user_message, raises

    def generate(self, prompt):
        self.prompts.append(prompt)
        if self._raises:
            raise self._raises
        if self._text is not None:
            return GeminiResult(ok=True, text=self._text, model=self.model)
        return GeminiResult(ok=False, kind=self._kind or ErrorKind.UNKNOWN,
                            user_message=self._msg or "Gemini failed.", model=self.model)


def sample_files():
    return read_folder(SAMPLE_DIR)


def sample_engine(**kwargs):
    engine = RAGEngine(**kwargs)
    engine.sync_documents(sample_files())
    return engine
