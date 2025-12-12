"""
Offline index builder: reads tolerated_decisions.jsonl, chunks text, embeds with OpenAI, writes decisions_index.npz.

Usage:
  OPENAI_API_KEY=... python index_builder.py \
    --input tolerated_decisions.jsonl \
    --output decisions_index.npz \
    --chunk-size 800 \
    --chunk-overlap 100 \
    --model text-embedding-3-small
"""

import argparse
import json
import os
from typing import Dict, List, Tuple

import numpy as np
from openai import OpenAI

DEFAULT_INPUT = os.getenv("DECISIONS_JSONL", "tolerated_decisions.jsonl")
DEFAULT_OUTPUT = os.getenv("INDEX_PATH", "decisions_index.npz")
DEFAULT_MODEL = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")
DEFAULT_CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "800"))
DEFAULT_CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "100"))


def env_flag(name: str) -> bool:
    return os.getenv(name, "").lower() in ("1", "true", "yes", "on")


def load_jsonl(path: str) -> List[Dict]:
    docs: List[Dict] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            docs.append(json.loads(line))
    return docs


def chunk_text(text: str, chunk_size: int, overlap: int) -> List[str]:
    chunks: List[str] = []
    start = 0
    while start < len(text):
        end = min(len(text), start + chunk_size)
        chunks.append(text[start:end])
        if end == len(text):
            break
        start = max(end - overlap, start + 1)
    return chunks


def embed_chunks(chunks: List[Dict], model: str, client: OpenAI) -> Tuple[np.ndarray, np.ndarray]:
    embeddings = []
    meta_entries = []
    for chunk in chunks:
        resp = client.embeddings.create(model=model, input=chunk["chunk"])
        vec = np.array(resp.data[0].embedding, dtype=float)
        embeddings.append(vec)
        meta_entries.append(chunk)
    return np.vstack(embeddings), np.array(meta_entries, dtype=object)


def build_index(input_path: str, output_path: str, model: str, chunk_size: int, chunk_overlap: int) -> None:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is required to build the index")

    client = OpenAI(api_key=api_key)
    print(f"[builder] Reading {input_path}")
    docs = load_jsonl(input_path)

    chunk_records: List[Dict] = []
    for doc in docs:
        doc_id = doc.get("id") or doc.get("decision_id") or doc.get("uuid") or "unknown-id"
        text = doc.get("text") or doc.get("content") or ""
        if not text:
            continue
        for idx, chunk in enumerate(chunk_text(text, chunk_size, chunk_overlap)):
            chunk_records.append(
                {
                    "id": doc_id,
                    "chunk_id": idx,
                    "chunk": chunk,
                    "title": doc.get("title"),
                    "source": doc.get("source"),
                }
            )

    if not chunk_records:
        raise RuntimeError("No chunks produced; check input content/fields")

    print(f"[builder] Created {len(chunk_records)} chunks; embedding with {model}")
    embeddings, meta = embed_chunks(chunk_records, model, client)

    print(f"[builder] Saving index to {output_path}")
    np.savez(output_path, embeddings=embeddings, meta=meta)
    print(f"[builder] Done. embeddings_shape={embeddings.shape}, meta_len={len(meta)}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build decisions_index.npz from a JSONL of decisions.")
    parser.add_argument("--input", default=DEFAULT_INPUT, help=f"Path to decisions JSONL (default: {DEFAULT_INPUT})")
    parser.add_argument("--output", default=DEFAULT_OUTPUT, help=f"Path to write NPZ (default: {DEFAULT_OUTPUT})")
    parser.add_argument("--model", default=DEFAULT_MODEL, help=f"Embedding model (default: {DEFAULT_MODEL})")
    parser.add_argument("--chunk-size", type=int, default=DEFAULT_CHUNK_SIZE, help="Chunk size in characters")
    parser.add_argument("--chunk-overlap", type=int, default=DEFAULT_CHUNK_OVERLAP, help="Chunk overlap in characters")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    build_index(
        input_path=args.input,
        output_path=args.output,
        model=args.model,
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
    )
