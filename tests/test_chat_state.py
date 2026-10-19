"""Chat history: persistence, no duplicate questions, no duplicate answers."""

import unittest

import helpers  # noqa: F401  (puts the project root on sys.path)
from chat_state import ChatHistory


class ChatHistoryTests(unittest.TestCase):
    def test_history_keeps_order_and_each_answers_metadata(self):
        chat = ChatHistory()
        q1 = chat.add_user("first question")
        chat.add_assistant(q1.id, "first answer", {"label": "Gemini (online)", "sources": [{"n": 1}]})
        q2 = chat.add_user("second question")
        chat.add_assistant(q2.id, "second answer", {"label": "Local RAG (offline fallback)"})
        self.assertEqual([m.role for m in chat.messages], ["user", "assistant", "user", "assistant"])
        self.assertEqual(chat.messages[1].meta["label"], "Gemini (online)")
        self.assertEqual(chat.messages[3].meta["label"], "Local RAG (offline fallback)")
        self.assertIsNone(chat.pending())

    def test_a_question_can_only_be_answered_once(self):
        chat = ChatHistory()
        q = chat.add_user("question")
        self.assertIsNotNone(chat.add_assistant(q.id, "answer", {}))
        self.assertIsNone(chat.add_assistant(q.id, "duplicate answer", {}))      # rerun/race: ignored
        self.assertEqual([m.content for m in chat.messages], ["question", "answer"])

    def test_answer_to_an_unknown_question_is_ignored(self):
        chat = ChatHistory()
        self.assertIsNone(chat.add_assistant("no-such-id", "x", {}))
        self.assertEqual(chat.messages, [])

    def test_resubmitting_the_same_text_while_unanswered_does_not_duplicate_it(self):
        chat = ChatHistory()
        a = chat.add_user("same question")
        b = chat.add_user("same question")
        self.assertIs(a, b)
        self.assertEqual(len(chat.messages), 1)

    def test_asking_the_same_thing_again_after_it_was_answered_is_a_new_question(self):
        chat = ChatHistory()
        q = chat.add_user("repeat me")
        chat.add_assistant(q.id, "answer", {})
        again = chat.add_user("repeat me")
        self.assertIsNot(q, again)
        self.assertEqual(len(chat.messages), 3)

    def test_interrupted_run_leaves_a_pending_question_that_is_answered_exactly_once(self):
        chat = ChatHistory()
        chat.add_user("question interrupted mid-answer")           # run 1 stops before answering
        pending = chat.pending()                                    # run 2 (a rerun) picks it up
        self.assertEqual(pending.content, "question interrupted mid-answer")
        chat.add_assistant(pending.id, "answer", {})
        self.assertIsNone(chat.pending())
        self.assertEqual(len(chat.messages), 2)

    def test_multiple_pending_questions_are_answered_oldest_first(self):
        chat = ChatHistory()
        chat.add_user("one")
        chat.add_user("two")
        first = chat.pending()
        self.assertEqual(first.content, "one")
        chat.add_assistant(first.id, "a1", {})
        self.assertEqual(chat.pending().content, "two")

    def test_blank_input_is_ignored(self):
        chat = ChatHistory()
        self.assertIsNone(chat.add_user("   "))
        self.assertIsNone(chat.add_user(None))
        self.assertEqual(chat.messages, [])


if __name__ == "__main__":
    unittest.main()
