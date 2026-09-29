# 📄 RAG Document Q&A Chatbot

A Retrieval-Augmented Generation (RAG) chatbot that answers questions grounded
in your own documents. Built to be **fully functional offline** (no API key
needed) with an optional upgrade path to real generative answers via OpenAI
or Anthropic.

## Why this project is a strong portfolio piece
- Demonstrates the RAG pattern end-to-end: document ingestion → chunking →
  retrieval → grounded answer generation.
- Works out of the box for anyone reviewing your GitHub/demo — no API key
  required to try it.
- Shows you understand the difference between retrieval quality (embeddings/
  similarity search) and generation quality (LLM), and can swap either.

## Features
- Upload your own `.txt`/`.md` files, or use the bundled sample docs
  (a mock employee handbook + product FAQ).
- Overlapping chunking so answers aren't cut off mid-sentence.
- TF-IDF + cosine similarity retrieval by default (zero setup).
- Optional switch to `sentence-transformers` for stronger semantic search.
- Optional OpenAI / Anthropic key for natural-language generated answers,
  with source citations shown for every answer.
- Simple chat UI built with Streamlit.

## Run locally
```bash
pip install -r requirements.txt
streamlit run app.py
```
Open the URL Streamlit prints (usually http://localhost:8501).

## Deploy for free (Streamlit Community Cloud)
1. Push this folder to a public GitHub repo.
2. Go to https://share.streamlit.io → **New app** → select your repo →
   main file `app.py` → Deploy.
3. (Optional) In **App settings → Secrets**, add:
   ```
   OPENAI_API_KEY = "sk-..."
   ```
   to enable generative answers without users typing a key.

## Project structure
```
rag_chatbot/
├── app.py              # Streamlit UI
├── rag_engine.py        # Chunking, retrieval, and answer generation logic
├── sample_docs/          # Demo documents (handbook + product FAQ)
├── requirements.txt
└── README.md
```

## How it works (for your resume/interview talking points)
1. **Ingestion**: Documents are loaded and split into overlapping ~220-word
   chunks so context isn't lost at chunk boundaries.
2. **Indexing**: Each chunk is vectorized with TF-IDF (swappable for dense
   embeddings via `sentence-transformers`).
3. **Retrieval**: A user query is vectorized the same way; the top-k most
   similar chunks are retrieved via cosine similarity.
4. **Generation**: The retrieved chunks are passed as context to an LLM
   (or, in offline mode, the single best-matching passage is returned
   directly) so answers are always grounded in the source documents rather
   than the model's own knowledge — reducing hallucination.

## Ideas to extend further
- Add PDF/DOCX ingestion (`pypdf`, `python-docx`).
- Swap FAISS/Chroma in for the vector store at larger scale.
- Add conversation memory so follow-up questions use prior turns as context.
- Add a "confidence" flag when the best similarity score is low, so the bot
  can say "I'm not confident this is in the documents."
