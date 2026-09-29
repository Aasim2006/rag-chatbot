"""
rag_engine.py
--------------
Core Retrieval-Augmented Generation (RAG) logic.

Design goals for this portfolio project:
1. Works FULLY OFFLINE out of the box (no API key required) using TF-IDF +
   cosine similarity for retrieval and an extractive fallback for the answer.
2. If an OpenAI or Anthropic API key is supplied, it upgrades to a real
   generative answer written in natural language, grounded in the retrieved
   chunks (classic RAG pattern).
3. Swappable embedding backend: TF-IDF (default, zero setup) or
   sentence-transformers (better semantic search, needs `pip install
   sentence-transformers`) via USE_SENTENCE_EMBEDDINGS below.
"""

import os
import re
import glob
from dataclasses import dataclass

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Toggle to True if you `pip install sentence-transformers` and want
# stronger semantic retrieval instead of TF-IDF.
USE_SENTENCE_EMBEDDINGS = False


@dataclass
class Chunk:
    text: str
    source: str
    chunk_id: int


def load_documents(folder: str) -> list[tuple[str, str]]:
    """Load all .txt / .md files from a folder. Returns list of (filename, text)."""
    docs = []
    for path in sorted(glob.glob(os.path.join(folder, "*"))):
        if path.lower().endswith((".txt", ".md")):
            with open(path, "r", encoding="utf-8", errors="ignore") as f:
                docs.append((os.path.basename(path), f.read()))
    return docs


def chunk_text(text: str, chunk_size: int = 110, overlap: int = 20) -> list[str]:
    """
    Split text into chunks along natural paragraph/section boundaries first
    (so a policy section or FAQ answer stays intact), falling back to a
    word-count sliding window only for paragraphs longer than chunk_size.
    """
    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    chunks = []
    for para in paragraphs:
        para = re.sub(r"\s+", " ", para).strip()
        words = para.split(" ")
        if len(words) <= chunk_size:
            chunks.append(para)
            continue
        start = 0
        while start < len(words):
            end = start + chunk_size
            piece = " ".join(words[start:end]).strip()
            if piece:
                chunks.append(piece)
            start += chunk_size - overlap
    return chunks


class RAGEngine:
    def __init__(self):
        self.chunks: list[Chunk] = []
        self.vectorizer = None
        self.doc_matrix = None
        self._embedder = None

    def build_index(self, folder: str):
        """Load, chunk, and index all documents in `folder`."""
        self.chunks = []
        cid = 0
        for filename, text in load_documents(folder):
            for piece in chunk_text(text):
                self.chunks.append(Chunk(text=piece, source=filename, chunk_id=cid))
                cid += 1

        if not self.chunks:
            raise ValueError(f"No .txt/.md documents found in '{folder}'.")

        texts = [c.text for c in self.chunks]

        if USE_SENTENCE_EMBEDDINGS:
            from sentence_transformers import SentenceTransformer
            if self._embedder is None:
                self._embedder = SentenceTransformer("all-MiniLM-L6-v2")
            self.doc_matrix = self._embedder.encode(texts, normalize_embeddings=True)
        else:
            self.vectorizer = TfidfVectorizer(stop_words="english", max_features=5000)
            self.doc_matrix = self.vectorizer.fit_transform(texts)

    def retrieve(self, query: str, top_k: int = 3) -> list[tuple[Chunk, float]]:
        """Return the top_k most relevant chunks for the query, with similarity scores."""
        if USE_SENTENCE_EMBEDDINGS:
            q_vec = self._embedder.encode([query], normalize_embeddings=True)
            sims = cosine_similarity(q_vec, self.doc_matrix)[0]
        else:
            q_vec = self.vectorizer.transform([query])
            sims = cosine_similarity(q_vec, self.doc_matrix)[0]

        top_idx = np.argsort(sims)[::-1][:top_k]
        return [(self.chunks[i], float(sims[i])) for i in top_idx if sims[i] > 0]

    def answer(self, query: str, top_k: int = 3, llm_provider: str | None = None,
               api_key: str | None = None) -> dict:
        """
        Retrieve relevant chunks, then generate an answer.
        llm_provider: None (extractive fallback), "openai", or "anthropic".
        """
        results = self.retrieve(query, top_k=top_k)
        if not results:
            return {
                "answer": "I couldn't find anything relevant to that question in the "
                          "loaded documents. Try rephrasing, or upload more source files.",
                "sources": [],
            }

        context = "\n\n".join(
            f"[{c.source} | chunk {c.chunk_id}]\n{c.text}" for c, _ in results
        )
        sources = [{"source": c.source, "chunk_id": c.chunk_id, "score": round(s, 3)}
                   for c, s in results]

        if llm_provider == "openai" and api_key:
            answer_text = self._generate_openai(query, context, api_key)
        elif llm_provider == "anthropic" and api_key:
            answer_text = self._generate_anthropic(query, context, api_key)
        else:
            # Extractive fallback: no API key needed, works fully offline.
            best_chunk, best_score = results[0]
            answer_text = (
                "(Offline extractive mode — add an API key in the sidebar for a "
                "natural-language answer)\n\nMost relevant passage "
                f"(from {best_chunk.source}):\n\n\"{best_chunk.text.strip()}\""
            )

        return {"answer": answer_text, "sources": sources}

    @staticmethod
    def _generate_openai(query: str, context: str, api_key: str) -> str:
        from openai import OpenAI
        client = OpenAI(api_key=api_key)
        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": (
                    "You are a helpful assistant answering questions using ONLY the "
                    "provided context. If the answer isn't in the context, say you "
                    "don't know rather than guessing."
                )},
                {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {query}"},
            ],
            temperature=0.2,
        )
        return resp.choices[0].message.content

    @staticmethod
    def _generate_anthropic(query: str, context: str, api_key: str) -> str:
        import anthropic
        client = anthropic.Anthropic(api_key=api_key)
        resp = client.messages.create(
            model="claude-sonnet-4-5",
            max_tokens=500,
            system=(
                "You are a helpful assistant answering questions using ONLY the "
                "provided context. If the answer isn't in the context, say you "
                "don't know rather than guessing."
            ),
            messages=[{"role": "user", "content": f"Context:\n{context}\n\nQuestion: {query}"}],
        )
        return resp.content[0].text
