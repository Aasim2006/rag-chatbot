"""
check_gemini.py - confirm that Gemini works with YOUR key and model.

Run it from the project folder:
    python check_gemini.py

It uses exactly the same code path as the app (same SDK call, timeout and error
handling) and sends one tiny made-up passage, never your documents. It prints the model
name and a plain-language result. Your API key is never printed.
"""

import sys

try:  # optional: read GEMINI_API_KEY from a local .env file
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    pass

from rag_engine import Chunk, GeminiConfig, GeminiService, build_prompt


def main() -> int:
    config = GeminiConfig.from_env()
    print(f"Model: {config.model}")
    if not config.configured:
        print("No GEMINI_API_KEY found.")
        print("Put it in a .env file (copy .env.example) or set it as an environment variable.")
        print("The app still works without it, using local document search.")
        return 1

    prompt = build_prompt(
        "What colour is the sky?",
        [(Chunk("The sky is blue on a clear day.", "check.txt", 0), 1.0)],
    )
    result = GeminiService(config).generate(prompt)
    if result.ok:
        print("OK: Gemini answered, so online mode will work.")
        print("Answer:", (result.text or "")[:200])
        return 0

    print(f"FAILED ({result.kind.value}): {result.user_message}")
    if result.kind is not None and result.kind.value == "model_unavailable":
        print("Set GEMINI_MODEL to a current model name: https://ai.google.dev/gemini-api/docs/models")
    return 1


if __name__ == "__main__":
    sys.exit(main())
