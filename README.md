# 📄 RAG Document Q&A Chatbot

Ask questions about your own documents. The app finds the most relevant passages
(TF-IDF + cosine similarity), then answers with **Google Gemini** when it is available.
If Gemini is not set up or not working, it **switches to local document search
automatically**. There is nothing to select and no "online/offline" switch.

- Reads **TXT, MD, PDF** (page-aware) and **PPTX** (slide text, tables, speaker notes)
- Every answer shows its sources, with **page or slide number** when the file has them
- A status label on each answer says exactly where it came from:
  `Gemini (online) · <model>` or `Local RAG (offline fallback)`. It never claims Gemini
  answered when it didn't.
- Greetings ("hi", "thanks", "bye") are answered locally and never sent to Gemini
- If nothing relevant is found, it says so instead of guessing

## Requirements

- Python **3.10 or newer**
- The packages in `requirements.txt` (Streamlit, scikit-learn, pypdf, python-pptx,
  google-genai, python-dotenv)

## Quick start

```bash
python3 -m venv venv
source venv/bin/activate            # Windows: venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
```

Open http://localhost:8501. With no API key the app already works in local mode.

## Add Gemini (optional but recommended)

1. **Get a key** from Google AI Studio: https://aistudio.google.com/apikey
2. **Save it locally.** Copy `.env.example` to `.env` and edit it:
   ```
   GEMINI_API_KEY=your_real_key_here
   ```
   `.env` is listed in `.gitignore`, so it is never committed. You can also set
   `GEMINI_API_KEY` as a normal environment variable instead.
3. **Check it works** (uses the same code path as the app):
   ```bash
   python check_gemini.py
   ```
4. Restart the app. The sidebar shows the current Gemini status.

### Choosing the model

The default is `gemini-3.8-flash`, which Google's model list shows as stable at the time of
writing. Model names change over time. To use another one, set `GEMINI_MODEL`
(in `.env`, the environment, or Streamlit secrets). No code change is needed. If the app
says *"The configured Gemini model isn't available"*, pick a current name from
https://ai.google.dev/gemini-api/docs/models.

### All settings

| Name | Default | Meaning |
|---|---|---|
| `GEMINI_API_KEY` | none | Your key. Without it the app uses local mode. |
| `GEMINI_MODEL` | `gemini-3.8-flash` | Model ID. |
| `GEMINI_TIMEOUT_SECONDS` | `25` | Give up on one request after this long (1 to 120). |
| `GEMINI_MAX_RETRIES` | `1` | Extra attempts for temporary errors only (0 to 3). |
| `GEMINI_API` | `auto` | `auto`, `interactions` or `generate_content`. Escape hatch if the SDK changes. |
| `RAG_MIN_SCORE` | `0.08` | How similar a passage must be to count as relevant (0 to 1). |

## How automatic online / offline works

For each question the app: (1) answers greetings locally, (2) retrieves passages locally,
(3) if nothing relevant is found, says so and stops, (4) otherwise sends the question and
**only those passages** to Gemini, and (5) shows the answer with sources and a status label.

| Situation | What happens |
|---|---|
| No API key | Local answer. No request is ever made. |
| Invalid key / access denied | Local answer. **Not retried**; Gemini stays off until you fix the key and restart. |
| Unknown or retired model | Local answer. **Not retried**; fix `GEMINI_MODEL` and restart. |
| Rate limit (too many requests) | Local answer. Gemini is paused for about a minute (or the delay Google asks for), then tried again automatically. |
| Quota used up | Local answer. Paused for about 15 minutes, then tried again. |
| Network down, server error | One retry for a brief hiccup, then a local answer. Paused about 30 seconds, then tried again. |
| Request too slow | Stops after the timeout, local answer, short pause. |

No request is sent just to "test the connection". The app learns Gemini's state from real
answers, and the pause is remembered in memory only. A Gemini failure can never crash the app.

## Security

- The key is read only from the environment, `.env`, or Streamlit secrets. It is never typed
  into the page, never sent to the browser, and never printed or logged. Error text is
  redacted before it is logged.
- Uploaded documents are treated as **untrusted data**. The prompt tells Gemini to ignore
  instructions that appear inside them, and document text cannot close or fake the prompt's
  delimiters.
- Citation numbers that Gemini invents are stripped. Source names and page or slide numbers
  shown in the Sources panel come from the retrieval step, not from the model's text.
- File names and document contents are not written to logs.
- Never commit `.env` or `.streamlit/secrets.toml`. Both are in `.gitignore`.

## Deploy on Streamlit Community Cloud

1. Push the project to GitHub (without `.env`).
2. https://share.streamlit.io → **New app** → pick the repo → main file `app.py`.
3. Open **App settings → Secrets** and paste (using your real key):
   ```toml
   GEMINI_API_KEY = "your_real_key_here"
   # GEMINI_MODEL = "gemini-3.8-flash"
   ```
   `.streamlit/secrets.toml.example` shows the format.
4. Save. The app restarts and uses Gemini.

**Cost warning:** on a public app, **every visitor's question uses your key and your quota**.
Watch your usage, set spending limits in Google Cloud if you enable billing, or keep the app
private.

## Gemini quota and billing

Google offers a free tier with rate limits, and paid usage is billed per use. Limits and
prices differ by model and change over time, so check the current numbers in Google's Gemini
API documentation: https://ai.google.dev/gemini-api/docs. When a limit is hit, the app keeps
working in local mode and tells you what happened.

## Limits of the local (offline) answer

The local answer is **not** written by an AI. It shows the best-matching passage from your
documents word for word, and lists other related passages under Sources.

- It matches on shared words, so it can miss answers phrased very differently from the
  question. Weak matches are flagged as "Low-confidence".
- It cannot combine facts from several passages or summarise.
- Scanned PDFs (images of text) have no text layer and can't be read without OCR.
- For better local matching you can install `sentence-transformers` and set
  `USE_SENTENCE_EMBEDDINGS = True` in `rag_engine.py`. This is optional and downloads a model.

## Project structure

```
app.py               Streamlit page (uploads, chat, status)
rag_engine.py        Parsing, chunking, retrieval, Gemini service, answer logic
chat_state.py        Chat history (answers each question once, no duplicates)
check_gemini.py      Quick "does my key and model work?" check
sample_docs/         Demo documents (TXT, PDF, PPTX)
tests/               Automated tests
.env.example         Template for local settings (placeholders only)
.streamlit/secrets.toml.example   Template for Streamlit Cloud secrets
```

## Tests

```bash
python -m unittest discover -s tests
```

No API key, network or paid request is needed: Gemini is replaced by scripted fakes. The
tests cover Gemini success, missing key, network failure, invalid key, rate limit and
quota, unavailable model, greetings, no relevant passage, multiple documents and citations,
chat history and duplicate prevention, document removal and re-indexing, and local operation
with the Gemini library blocked and the network disabled.

The Streamlit page itself is tested with a **stand-in for Streamlit**, so it checks the app's
logic but not how real Streamlit draws the widgets.

### Manual checks (not covered by automated tests)

1. **Real Gemini answer:** set your key, run `python check_gemini.py`, then ask a question
   and confirm the label reads `Gemini (online)`.
2. **Local fallback:** remove the key (or set a wrong one) and ask again. The label should
   read `Local RAG (offline)` or `Local RAG (offline fallback)` with a short explanation.
3. **Network failure:** turn Wi-Fi off, ask a question, and confirm a local answer with a
   network message. Turn Wi-Fi back on; Gemini is tried again after the short pause.
4. **Documents:** upload a PDF and a PPTX, ask about each, check the page or slide numbers,
   then remove one and confirm it no longer appears in Sources.
