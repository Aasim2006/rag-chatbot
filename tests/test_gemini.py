"""Gemini service: success, missing key, network failure, invalid key, rate limit,
quota, unavailable model, retries, cooldowns, timeouts and secret handling.
All Gemini calls are mocked, so no key, network or money is needed."""

import socket
import time
import unittest

import helpers
from helpers import FakeAPIError, make_service

from rag_engine import (SYSTEM_INSTRUCTION, ErrorKind, GeminiConfig, GeminiService,
                        classify_error, redact)


class ReadTimeout(Exception):  # looks like httpx.ReadTimeout (matched by class name)
    pass


class SuccessTests(unittest.TestCase):
    def test_gemini_generates_an_answer_with_the_interactions_api(self):
        svc, client, _, _ = make_service(["The late fee is INR 5 per day. [1]"])
        r = svc.generate("PROMPT")
        self.assertTrue(r.ok)
        self.assertEqual(r.text, "The late fee is INR 5 per day. [1]")
        call = client.calls[0]
        self.assertEqual(call["model"], svc.model)
        self.assertEqual(call["input"], "PROMPT")
        self.assertEqual(call["system_instruction"], SYSTEM_INSTRUCTION)
        self.assertEqual(svc.status().state, "online")

    def test_uses_generate_content_when_sdk_has_no_interactions_api(self):
        svc, client, _, _ = make_service(["OK"], style="generate_content")
        self.assertTrue(svc.generate("PROMPT").ok)
        self.assertEqual(client.calls[0]["contents"], "PROMPT")
        self.assertEqual(client.calls[0]["config"], {"system_instruction": SYSTEM_INSTRUCTION})

    def test_api_style_override_forces_generate_content(self):
        svc, client, _, _ = make_service(["OK"], style="both", api_style="generate_content")
        self.assertTrue(svc.generate("P").ok)
        self.assertIn("contents", client.calls[0])

    def test_signature_mismatch_falls_back_without_a_second_network_request(self):
        client = helpers.ScriptedClient(["FROM_GENERATE_CONTENT"], style="generate_content")

        def bad_create(**kw):
            raise TypeError("unexpected keyword argument 'system_instruction'")

        import types
        client.interactions = types.SimpleNamespace(create=bad_create)
        svc = GeminiService(GeminiConfig(api_key=helpers.TEST_KEY), client_factory=lambda c: client,
                            clock=helpers.FakeClock(), sleep=lambda s: None)
        r = svc.generate("P")
        self.assertTrue(r.ok)
        self.assertEqual(r.text, "FROM_GENERATE_CONTENT")
        self.assertEqual(len(client.calls), 1)


class NotConfiguredTests(unittest.TestCase):
    def test_missing_key_makes_no_sdk_or_network_call(self):
        created = []
        svc = GeminiService(GeminiConfig(api_key=None), client_factory=lambda c: created.append(1))
        r = svc.generate("P")
        self.assertFalse(r.ok)
        self.assertEqual(r.kind, ErrorKind.NOT_CONFIGURED)
        self.assertFalse(svc.is_configured)
        self.assertEqual(created, [])
        self.assertEqual(svc.status().state, "not_configured")

    def test_placeholder_and_blank_keys_count_as_missing(self):
        for value in ("your_api_key_here", "  ", "", None, "<paste key>", "YOUR-KEY"):
            cfg = GeminiConfig.from_getter({"GEMINI_API_KEY": value}.get)
            self.assertFalse(cfg.configured, value)

    def test_config_reads_environment_and_ignores_bad_numbers(self):
        cfg = GeminiConfig.from_env({"GEMINI_API_KEY": ' "AIzaREALKEY123456" ', "GEMINI_MODEL": "my-model",
                                     "GEMINI_TIMEOUT_SECONDS": "abc", "GEMINI_MAX_RETRIES": "99",
                                     "GEMINI_API": "nonsense"})
        self.assertEqual(cfg.api_key, "AIzaREALKEY123456")
        self.assertEqual(cfg.model, "my-model")
        self.assertEqual(cfg.timeout_s, 25.0)   # bad value -> default
        self.assertEqual(cfg.max_retries, 3)    # clamped to the allowed range
        self.assertEqual(cfg.api_style, "auto")
        self.assertEqual(GeminiConfig.from_env({}).model, "gemini-3.8-flash")

    def test_key_is_never_in_repr_or_fingerprint(self):
        cfg = GeminiConfig(api_key=helpers.TEST_KEY)
        self.assertNotIn(helpers.TEST_KEY, repr(cfg))
        self.assertNotIn(helpers.TEST_KEY, cfg.fingerprint)
        self.assertNotEqual(cfg.fingerprint, GeminiConfig(api_key="another-key-123456").fingerprint)


class NetworkFailureTests(unittest.TestCase):
    def test_network_failure_retries_once_then_falls_back_then_pauses(self):
        svc, client, clock, sleeps = make_service([ConnectionError("Connection refused")] * 2 + ["BACK ONLINE"])
        r = svc.generate("P")
        self.assertFalse(r.ok)
        self.assertEqual(r.kind, ErrorKind.NETWORK)
        self.assertEqual(len(client.calls), 2)          # 1 try + exactly 1 bounded retry
        self.assertEqual(len(sleeps), 1)
        self.assertIn("network", r.user_message.lower())

        # Next question while paused: instant fallback, no new request at all.
        r2 = svc.generate("P")
        self.assertEqual(r2.kind, ErrorKind.NETWORK)
        self.assertEqual(len(client.calls), 2)
        st = svc.status()
        self.assertEqual(st.state, "paused")
        self.assertGreater(st.retry_in_s, 0)

        # After the short cooldown it is tried again and recovers.
        clock.advance(31)
        r3 = svc.generate("P")
        self.assertTrue(r3.ok)
        self.assertEqual(r3.text, "BACK ONLINE")
        self.assertEqual(svc.status().state, "online")

    def test_dns_failure_is_a_network_error(self):
        svc, client, _, _ = make_service([socket.gaierror("Name or service not known")] * 2)
        self.assertEqual(svc.generate("P").kind, ErrorKind.NETWORK)

    def test_retries_can_be_turned_off(self):
        svc, client, _, sleeps = make_service([ConnectionError("x")] * 3, max_retries=0)
        svc.generate("P")
        self.assertEqual(len(client.calls), 1)
        self.assertEqual(sleeps, [])

    def test_slow_request_hits_the_hard_timeout_and_is_not_retried(self):
        svc, client, _, sleeps = make_service([lambda: time.sleep(0.6) or "too late"], timeout_s=0.05)
        t0 = time.monotonic()
        r = svc.generate("P")
        self.assertLess(time.monotonic() - t0, 0.5)     # returned at the deadline, not after 0.6 s
        self.assertEqual(r.kind, ErrorKind.TIMEOUT)
        self.assertEqual(len(client.calls), 1)
        self.assertEqual(sleeps, [])


class PermanentErrorTests(unittest.TestCase):
    def test_invalid_key_is_not_retried_and_gemini_stays_disabled(self):
        err = FakeAPIError(400, "INVALID_ARGUMENT", "API key not valid. Please pass a valid API key.")
        svc, client, clock, sleeps = make_service([err, "SHOULD NEVER BE USED"])
        r = svc.generate("P")
        self.assertEqual(r.kind, ErrorKind.AUTH)
        self.assertIn("API key", r.user_message)
        self.assertEqual(len(client.calls), 1)
        self.assertEqual(sleeps, [])
        self.assertEqual(svc.status().state, "disabled")

        clock.advance(100000)                            # even much later: never retried
        self.assertEqual(svc.generate("P").kind, ErrorKind.AUTH)
        self.assertEqual(len(client.calls), 1)

    def test_401_and_403_are_auth_errors(self):
        for code, status in ((401, "UNAUTHENTICATED"), (403, "PERMISSION_DENIED")):
            svc, client, _, _ = make_service([FakeAPIError(code, status, "denied")])
            self.assertEqual(svc.generate("P").kind, ErrorKind.AUTH)
            self.assertEqual(len(client.calls), 1)

    def test_unavailable_model_is_reported_and_not_retried(self):
        err = FakeAPIError(404, "NOT_FOUND", "models/gemini-old is not found for API version v1beta")
        svc, client, clock, _ = make_service([err])
        r = svc.generate("P")
        self.assertEqual(r.kind, ErrorKind.MODEL)
        self.assertIn("model", r.user_message.lower())
        clock.advance(100000)
        svc.generate("P")
        self.assertEqual(len(client.calls), 1)
        self.assertEqual(svc.status().state, "disabled")

    def test_missing_sdk_is_reported_once(self):
        calls = []

        def factory(cfg):
            calls.append(1)
            raise ImportError("No module named 'google'")

        svc = GeminiService(GeminiConfig(api_key=helpers.TEST_KEY), client_factory=factory)
        self.assertEqual(svc.generate("P").kind, ErrorKind.SDK_MISSING)
        svc.generate("P")
        self.assertEqual(len(calls), 1)


class RateLimitAndQuotaTests(unittest.TestCase):
    def test_rate_limit_is_not_retried_and_pauses_for_the_advised_delay(self):
        err = FakeAPIError(429, "RESOURCE_EXHAUSTED", "Too many requests per minute. Please retry in 12.5s.")
        svc, client, clock, sleeps = make_service([err, "RECOVERED"])
        r = svc.generate("P")
        self.assertEqual(r.kind, ErrorKind.RATE_LIMIT)
        self.assertEqual(len(client.calls), 1)
        self.assertEqual(sleeps, [])
        clock.advance(10)
        self.assertEqual(svc.generate("P").kind, ErrorKind.RATE_LIMIT)  # still paused
        self.assertEqual(len(client.calls), 1)
        clock.advance(3)
        self.assertTrue(svc.generate("P").ok)                            # advised delay has passed
        self.assertEqual(len(client.calls), 2)

    def test_daily_quota_exhaustion_pauses_for_a_long_time(self):
        err = FakeAPIError(429, "RESOURCE_EXHAUSTED",
                           "You exceeded your current quota. Quota exceeded for metric "
                           "GenerateRequestsPerDayPerProjectPerModel-FreeTier")
        svc, client, clock, _ = make_service([err, "RECOVERED"])
        self.assertEqual(svc.generate("P").kind, ErrorKind.QUOTA)
        clock.advance(600)
        self.assertEqual(svc.generate("P").kind, ErrorKind.QUOTA)        # 10 min later: still paused
        self.assertEqual(len(client.calls), 1)
        clock.advance(400)                                                # > 15 min
        self.assertTrue(svc.generate("P").ok)


class TransientServerErrorTests(unittest.TestCase):
    def test_503_is_retried_and_can_recover_within_one_question(self):
        svc, client, _, sleeps = make_service([FakeAPIError(503, "UNAVAILABLE", "The model is overloaded."),
                                               "Recovered"])
        r = svc.generate("P")
        self.assertTrue(r.ok)
        self.assertEqual(len(client.calls), 2)
        self.assertEqual(len(sleeps), 1)

    def test_persistent_503_gives_up_after_the_bounded_retries(self):
        err = FakeAPIError(503, "UNAVAILABLE", "overloaded")
        svc, client, _, _ = make_service([err] * 5)
        r = svc.generate("P")
        self.assertEqual(r.kind, ErrorKind.SERVER)
        self.assertEqual(len(client.calls), 2)

    def test_empty_answer_is_a_per_question_problem_not_a_cooldown(self):
        svc, client, _, _ = make_service(["   ", "A real answer"])
        self.assertEqual(svc.generate("P").kind, ErrorKind.EMPTY)
        self.assertTrue(svc.generate("P").ok)                            # next question tried right away
        self.assertEqual(len(client.calls), 2)


class SecretHandlingTests(unittest.TestCase):
    def test_key_is_not_leaked_in_results_or_logs(self):
        leak = Exception(f"boom while calling https://x.test/v1?key={helpers.TEST_KEY} with {helpers.TEST_KEY}")
        svc, _, _, _ = make_service([leak, leak])
        with self.assertLogs("rag_chatbot", level="WARNING") as logs:
            r = svc.generate("a private question about salaries")
        everything = " ".join(logs.output) + " " + (r.user_message or "")
        self.assertNotIn(helpers.TEST_KEY, everything)
        self.assertNotIn("salaries", everything)                         # questions are not logged

    def test_redact_removes_key_shapes(self):
        text = "failed key=AIzaSyABCDEFGHIJKLMNOP and Bearer abc.def.ghi and AIzaSyZZZZZZZZZZZZZZ"
        out = redact(text, "other-secret-value")
        self.assertNotIn("AIza", out)
        self.assertNotIn("abc.def.ghi", out)


class ClassificationTests(unittest.TestCase):
    def test_error_classification_table(self):
        K = ErrorKind
        cases = [
            (FakeAPIError(400, "INVALID_ARGUMENT", "API key not valid. Please pass a valid API key."), K.AUTH),
            (FakeAPIError(401, "UNAUTHENTICATED", "bad"), K.AUTH),
            (FakeAPIError(403, "PERMISSION_DENIED", "x"), K.AUTH),
            (FakeAPIError(403, "PERMISSION_DENIED", "Your API key was reported as leaked."), K.AUTH),
            (FakeAPIError(400, "FAILED_PRECONDITION", "User location is not supported"), K.AUTH),
            (FakeAPIError(402, "PAYMENT_REQUIRED", "credit balance depleted"), K.QUOTA),
            (FakeAPIError(404, "NOT_FOUND", "models/foo is not found for API version v1beta"), K.MODEL),
            (FakeAPIError(404, "NOT_FOUND", "model_not_found"), K.MODEL),
            (FakeAPIError(429, "RESOURCE_EXHAUSTED", "Too many requests per minute"), K.RATE_LIMIT),
            (FakeAPIError(429, "RESOURCE_EXHAUSTED", "quota exceeded: GenerateRequestsPerDay"), K.QUOTA),
            (FakeAPIError(429, "RESOURCE_EXHAUSTED", "You exceeded your current quota"), K.QUOTA),
            (FakeAPIError(429, "RESOURCE_EXHAUSTED", "slow down"), K.RATE_LIMIT),
            (FakeAPIError(500, "INTERNAL", "oops"), K.SERVER),
            (FakeAPIError(503, "UNAVAILABLE", "overloaded"), K.SERVER),
            (FakeAPIError(504, "DEADLINE_EXCEEDED", "x"), K.SERVER),
            (FakeAPIError(400, "INVALID_ARGUMENT", "bad parameter"), K.BAD_REQUEST),
            (ConnectionError("refused"), K.NETWORK),
            (socket.gaierror("Name or service not known"), K.NETWORK),
            (OSError("Network is unreachable"), K.NETWORK),
            (TimeoutError("timed out"), K.TIMEOUT),
            (ReadTimeout("read timed out"), K.TIMEOUT),
            (ImportError("No module named google"), K.SDK_MISSING),
            (ValueError("???"), K.UNKNOWN),
        ]
        for exc, expected in cases:
            with self.subTest(exc=repr(exc)):
                self.assertEqual(classify_error(exc), expected)


if __name__ == "__main__":
    unittest.main()
