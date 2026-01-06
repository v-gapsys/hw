"""
Offline index builder: reads tolerated_decisions.jsonl, chunks text (or paragraphs), embeds with OpenAI, writes decisions_index.npz.

Usage:
  OPENAI_API_KEY=... python index_builder.py \
    --input tolerated_decisions.jsonl \
    --output decisions_index.npz \
    --chunk-size 800 \
    --chunk-overlap 100 \
    --model text-embedding-3-small \
    --use-paragraphs --paragraphs-per-chunk 1
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
DEFAULT_PARAS_PER_CHUNK = int(os.getenv("PARAS_PER_CHUNK", "1"))


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


def chunk_by_paragraphs(paragraphs: List[Dict], paragraphs_per_chunk: int = 1) -> List[List[Dict]]:
    """Group paragraphs into chunks; preserves paragraph metadata."""
    if paragraphs_per_chunk < 1:
        paragraphs_per_chunk = 1

    grouped: List[List[Dict]] = []
    current: List[Dict] = []

    for para in paragraphs:
        if not isinstance(para, dict):
            para = {"text": str(para)}
        text = para.get("text", "")
        if not text:
            continue
        current.append(para)
        if len(current) >= paragraphs_per_chunk:
            grouped.append(current)
            current = []

    if current:
        grouped.append(current)

    return grouped


def embed_chunks(chunks: List[Dict], model: str, client: OpenAI) -> Tuple[np.ndarray, np.ndarray]:
    embeddings = []
    meta_entries = []
    for chunk in chunks:
        resp = client.embeddings.create(model=model, input=chunk["chunk"])
        vec = np.array(resp.data[0].embedding, dtype=float)
        embeddings.append(vec)
        meta_entries.append(chunk)
    return np.vstack(embeddings), np.array(meta_entries, dtype=object)


def build_index(
    input_path: str,
    output_path: str,
    model: str,
    chunk_size: int,
    chunk_overlap: int,
    use_paragraphs: bool,
    paragraphs_per_chunk: int,
) -> None:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is required to build the index")

    client = OpenAI(api_key=api_key)
    print(f"[builder] Reading {input_path}")
    docs = load_jsonl(input_path)

    chunk_records: List[Dict] = []
    for doc in docs:
        doc_id = (
            doc.get("id")
            or doc.get("decision_id")
            or doc.get("uuid")
            or doc.get("case_number")
            or doc.get("link")
            or "unknown-id"
        )
        title = doc.get("title")
        source = doc.get("source") or doc.get("link")

        paragraphs = doc.get("paragraphs") if use_paragraphs else None
        if use_paragraphs and paragraphs and isinstance(paragraphs, list) and len(paragraphs) > 0:
            grouped = chunk_by_paragraphs(paragraphs, paragraphs_per_chunk)
            chunk_source = "paragraphs"
            for idx, group in enumerate(grouped):
                text_parts = [p.get("text", "") for p in group if isinstance(p, dict)]
                combined_text = " ".join(text_parts)
                meta_para = group[0] if group else {}
                chunk_records.append(
                    {
                        "id": doc_id,
                        "chunk_id": idx,
                        "chunk": combined_text,
                        "chunk_source": chunk_source,
                        "title": title,
                        "source": source,
                        "case_type": doc.get("case_type"),
                        "court": doc.get("court"),
                        "judges": doc.get("judges"),
                        "parties": doc.get("parties"),
                        "citations": doc.get("citations"),
                        "tolerated_reason": doc.get("tolerated_reason"),
                        "paragraph_count": doc.get("paragraph_count"),
                        "date_iso": doc.get("date_iso"),
                        "datetime_iso": doc.get("datetime_iso"),
                        "section": meta_para.get("section") if isinstance(meta_para, dict) else None,
                        "para_no": meta_para.get("para_no") if isinstance(meta_para, dict) else None,
                    }
                )
        else:
            text = doc.get("decision_section") or doc.get("text") or doc.get("content") or ""
            if not text:
                continue
            chunks = chunk_text(text, chunk_size, chunk_overlap)
            chunk_source = "text"
            for idx, chunk in enumerate(chunks):
                chunk_records.append(
                    {
                        "id": doc_id,
                        "chunk_id": idx,
                        "chunk": chunk,
                        "chunk_source": chunk_source,
                        "title": title,
                        "source": source,
                        "case_type": doc.get("case_type"),
                        "court": doc.get("court"),
                        "judges": doc.get("judges"),
                        "parties": doc.get("parties"),
                        "citations": doc.get("citations"),
                        "tolerated_reason": doc.get("tolerated_reason"),
                        "paragraph_count": doc.get("paragraph_count"),
                        "date_iso": doc.get("date_iso"),
                        "datetime_iso": doc.get("datetime_iso"),
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
    parser.add_argument(
        "--use-paragraphs",
        action="store_true",
        help="Use paragraphs field from JSONL when present (default: off)",
    )
    parser.add_argument(
        "--paragraphs-per-chunk",
        type=int,
        default=DEFAULT_PARAS_PER_CHUNK,
        help=f"How many paragraphs to combine per chunk when --use-paragraphs is set (default: {DEFAULT_PARAS_PER_CHUNK})",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    build_index(
        input_path=args.input,
        output_path=args.output,
        model=args.model,
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
        use_paragraphs=args.use_paragraphs,
        paragraphs_per_chunk=args.paragraphs_per_chunk,
    )
