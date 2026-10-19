"""Online/offline decision logic in RAGEngine.answer(): Gemini success, every fallback path,
and honest status labels. Gemini is always mocked."""

import unittest

import helpers
from helpers import FakeAPIError, FakeGemini, make_service, sample_engine

from rag_engine import ErrorKind

QUESTION = "What is the late fee per day for library books?"


class GeminiAnswerTests(unittest.TestCase):
    def test_gemini_answer_is_grounded_in_the_retrieved_passages(self):
        gem = FakeGemini(text="The late fee is INR 5 per day per book [1].")
        a = sample_engine().answer(QUESTION, gemini=gem)
        self.assertEqual(a.mode, "gemini")
        self.assertTrue(a.used_gemini)
        self.assertIn("Gemini (online)", a.label)
        self.assertIn("gemini-test", a.label)
        self.assertEqual(a.text, "The late fee is INR 5 per day per book [1].")
        self.assertIsNone(a.notice)
        # the question and the SAME passages the retriever found were sent to Gemini
        prompt = gem.prompts[0]
        self.assertIn(QUESTION, prompt)
        self.assertIn("INR 5 per day per book", prompt)
        self.assertIn("(source: sample_library_rules.pdf, page 2)", prompt)
        # citations shown to the user come from retrieval metadata, not from Gemini's text
        self.assertEqual(a.sources[0]["source"], "sample_library_rules.pdf")
        self.assertEqual(a.sources[0]["location"], "page 2")

    def test_citation_numbers_gemini_invents_are_stripped(self):
        gem = FakeGemini(text="INR 5 per day [1]. Confirmed elsewhere [9].")
        a = sample_engine().answer(QUESTION, top_k=2, gemini=gem)
        self.assertIn("[1]", a.text)
        self.assertNotIn("[9]", a.text)

    def test_multiple_documents_are_all_available_as_numbered_sources(self):
        gem = FakeGemini(text="ok [1]")
        a = sample_engine().answer("How many books and how many leave days?", top_k=4, gemini=gem)
        self.assertEqual([s["n"] for s in a.sources], list(range(1, len(a.sources) + 1)))
        self.assertGreaterEqual(len({s["source"] for s in a.sources}), 2)


class FallbackTests(unittest.TestCase):
    def test_not_configured_is_plain_offline_mode_and_gemini_is_never_called(self):
        gem = FakeGemini(text="NO", configured=False)
        a = sample_engine().answer(QUESTION, gemini=gem)
        self.assertEqual(a.mode, "offline")
        self.assertEqual(gem.prompts, [])
        self.assertIn("offline", a.label.lower())
        self.assertNotIn("Gemini (online)", a.label)

    def test_no_gemini_object_at_all_is_offline(self):
        a = sample_engine().answer(QUESTION, gemini=None)
        self.assertEqual(a.mode, "offline")

    def test_failure_falls_back_and_never_claims_gemini_answered(self):
        gem = FakeGemini(error_kind=ErrorKind.RATE_LIMIT, user_message="Gemini is rate-limiting requests right now.")
        a = sample_engine().answer(QUESTION, gemini=gem)
        self.assertEqual(a.mode, "offline_fallback")
        self.assertFalse(a.used_gemini)
        self.assertEqual(a.label, "Local RAG (offline fallback)")
        self.assertNotIn("Gemini (online)", a.label)
        self.assertIn("rate-limiting", a.notice)
        self.assertIn("local document search", a.notice)
        self.assertIn("INR 5", a.text)                       # still a useful, source-backed answer
        self.assertTrue(a.sources)

    def test_an_unexpected_exception_from_gemini_cannot_crash_the_answer(self):
        gem = FakeGemini(raises=RuntimeError("kaboom with a stack trace"))
        a = sample_engine().answer(QUESTION, gemini=gem)
        self.assertEqual(a.mode, "offline_fallback")
        self.assertNotIn("kaboom", (a.notice or "") + a.text)


class EndToEndWithRealServiceTests(unittest.TestCase):
    """RAGEngine + the real GeminiService, with only the SDK client faked."""

    def test_success_then_network_outage_then_recovery(self):
        svc, client, clock, _ = make_service(
            ["INR 5 per day [1]", ConnectionError("down"), ConnectionError("down"), "Back [1]"])
        engine = sample_engine()

        a1 = engine.answer(QUESTION, gemini=svc)
        self.assertEqual(a1.mode, "gemini")

        a2 = engine.answer(QUESTION, gemini=svc)                 # internet drops
        self.assertEqual(a2.mode, "offline_fallback")
        self.assertIn("network", a2.notice.lower())
        calls_after_outage = len(client.calls)

        a3 = engine.answer(QUESTION, gemini=svc)                 # paused: instant, no new request
        self.assertEqual(a3.mode, "offline_fallback")
        self.assertEqual(len(client.calls), calls_after_outage)
        self.assertIn("tried again automatically", a3.notice)

        clock.advance(31)
        a4 = engine.answer(QUESTION, gemini=svc)                 # service is tried again and works
        self.assertEqual(a4.mode, "gemini")

    def test_invalid_key_falls_back_once_and_is_never_retried(self):
        err = FakeAPIError(400, "INVALID_ARGUMENT", "API key not valid. Please pass a valid API key.")
        svc, client, _, _ = make_service([err])
        engine = sample_engine()
        for _ in range(3):
            a = engine.answer(QUESTION, gemini=svc)
            self.assertEqual(a.mode, "offline_fallback")
            self.assertIn("API key", a.notice)
        self.assertEqual(len(client.calls), 1)

    def test_missing_key_never_touches_the_sdk(self):
        svc, client, _, _ = make_service([], api_key=None)
        a = sample_engine().answer(QUESTION, gemini=svc)
        self.assertEqual(a.mode, "offline")
        self.assertEqual(client.calls, [])



class RegressionTests(unittest.TestCase):
    def test_answer_that_is_only_an_invented_citation_falls_back_instead_of_showing_nothing(self):
        gem = FakeGemini(text="[9]")  # only a citation that does not exist
        a = sample_engine().answer(QUESTION, top_k=2, gemini=gem)
        self.assertEqual(a.mode, "offline_fallback")
        self.assertNotIn("Gemini (online)", a.label)
        self.assertTrue(a.text.strip())
        self.assertIsNotNone(a.notice)

    def test_a_vague_429_is_a_short_rate_limit_not_a_long_quota_pause(self):
        svc, _client, clock, _sleeps = make_service([FakeAPIError(429, "RESOURCE_EXHAUSTED", "slow down")])
        r = svc.generate("PROMPT")
        self.assertEqual(r.kind, ErrorKind.RATE_LIMIT)
        clock.advance(61)  # default rate-limit pause is 60 seconds
        self.assertNotEqual(svc.status().state, "paused")

if __name__ == "__main__":
    unittest.main()
