from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from .config import ENABLE_SEARCH_TOOLS
from .core import mcp
from . import index_loader
from .openai_client import embed_query


def _ensure_search_enabled() -> None:
    if not ENABLE_SEARCH_TOOLS:
        raise RuntimeError("Search tools are disabled; set ENABLE_SEARCH_TOOLS=1 to use them")


def _metadata_search(query: str, meta_fields: List[str]) -> List[Tuple[int, float]]:
    """Search for exact matches in metadata fields. Returns (index, boost_score) pairs."""
    query_lower = query.lower()
    matches = []

    for idx, meta in enumerate(index_loader.META):
        if not isinstance(meta, dict):
            continue

        boost_score = 0.0
        for field in meta_fields:
            field_value = meta.get(field, "")

            # Handle motive_schema specially (it's a dict with labels)
            if field == "motive_schema" and isinstance(field_value, dict):
                labels = field_value.get("labels", [])
                for label in labels:
                    if isinstance(label, str) and query_lower in label.lower():
                        boost_score += 1.5  # Medium boost for motive matches
            # Handle regular string fields
            elif isinstance(field_value, str) and query_lower in field_value.lower():
                # Higher boost for exact matches vs partial matches
                if field_value.lower() == query_lower:
                    boost_score += 2.0
                else:
                    boost_score += 1.0

        if boost_score > 0:
            matches.append((idx, boost_score))

    return matches


@mcp.tool()
def search_decisions(query: str, top_k: int = 5, paragraph_type: Optional[str] = None) -> List[Dict[str, Any]]:
    """Hybrid search over decisions: combines metadata filtering with semantic search.
    Optionally filter by paragraph type (e.g., 'reasoning', 'facts', 'law').
    First searches structured metadata fields, then evaluates content semantically.
    Requires ENABLE_SEARCH_TOOLS and a loaded index."""
    _ensure_search_enabled()
    index_loader.ensure_index_ready()

    # Stage 1: Metadata search for exact matches in structured fields
    metadata_fields = ["parties", "judges", "court", "case_type", "title", "tolerated_reason", "motive_schema"]
    metadata_matches = _metadata_search(query, metadata_fields)

    # Stage 2: Semantic search across all content (or filtered by paragraph type)
    query_vec = embed_query(query)
    if index_loader.EMBEDDINGS.shape[1] != query_vec.shape[0]:
        raise RuntimeError(
            f"Embedding dimension mismatch: index has {index_loader.EMBEDDINGS.shape[1]}, query has {query_vec.shape[0]}"
        )

    semantic_scores = index_loader.EMBEDDINGS @ query_vec  # cosine-ish assuming normalized vectors

    # Filter by paragraph type if specified using text-based heuristics
    valid_indices = set(range(len(semantic_scores)))
    if paragraph_type:
        valid_indices = set()
        paragraph_type_lower = paragraph_type.lower()

        for idx, meta in enumerate(index_loader.META):
            if isinstance(meta, dict):
                chunk_text = meta.get("chunk", "").lower()

                # Apply paragraph type filtering based on content patterns
                should_include = False

                if paragraph_type_lower == "reasoning":
                    # Reasoning sections typically contain motivation, consideration, analysis
                    should_include = any(keyword in chunk_text for keyword in [
                        "motyv", "atsižvelg", "įvertinus", "consider", "reasoning",
                        "teismas", "konstatuoja", "nurodo"
                    ])
                elif paragraph_type_lower == "facts":
                    # Facts sections contain established circumstances
                    should_include = any(keyword in chunk_text for keyword in [
                        "nustatė", "nustatyta", "aplinky", "faktai", "facts",
                        "įrodyta", "patvirtinta"
                    ])
                elif paragraph_type_lower == "law":
                    # Law sections reference legal codes and provisions
                    should_include = any(keyword in chunk_text for keyword in [
                        "numato", " BK ", " CK ", " įstatym", " nuostat",
                        " Baudžiamojo kodekso ", " Civilinio kodekso "
                    ])
                elif paragraph_type_lower == "operative":
                    # Operative sections contain decisions and orders
                    should_include = any(keyword in chunk_text for keyword in [
                        "nutar", "nusprendė", "pripažįsta", "panaikina",
                        "patvirtina", "atmesti"
                    ])
                elif paragraph_type_lower == "header":
                    # Header sections contain case identification
                    should_include = any(keyword in chunk_text for keyword in [
                        "byla", "nr.", "teismas", "šalys", "case"
                    ])
                else:
                    # Unknown paragraph type - include chunk for broad search
                    should_include = True

                if should_include:
                    valid_indices.add(idx)

    # Stage 3: Combine results with hybrid scoring
    combined_scores = {}
    seen_decision_ids = set()

    # Process metadata matches first (higher priority)
    for idx, metadata_boost in metadata_matches:
        meta = index_loader.META[idx]
        decision_id = meta.get("id") if isinstance(meta, dict) else getattr(meta, "id", None)

        if decision_id and decision_id not in seen_decision_ids:
            # Boost semantic score with metadata match bonus
            base_semantic_score = float(semantic_scores[idx])
            combined_score = base_semantic_score + metadata_boost
            combined_scores[decision_id] = {
                "score": combined_score,
                "semantic_score": base_semantic_score,
                "metadata_boost": metadata_boost,
                "index": idx,
                "meta": meta
            }
            seen_decision_ids.add(decision_id)

    # Add remaining high-scoring semantic results (filtered by paragraph type)
    semantic_indices = np.argsort(semantic_scores)[::-1]  # Sort by semantic score descending

    for idx in semantic_indices:
        if idx not in valid_indices:
            continue

        meta = index_loader.META[idx]
        decision_id = meta.get("id") if isinstance(meta, dict) else getattr(meta, "id", None)

        if decision_id and decision_id not in seen_decision_ids:
            combined_scores[decision_id] = {
                "score": float(semantic_scores[idx]),
                "semantic_score": float(semantic_scores[idx]),
                "metadata_boost": 0.0,
                "index": idx,
                "meta": meta
            }
            seen_decision_ids.add(decision_id)

    # Sort by combined score and return top results
    sorted_results = sorted(combined_scores.items(), key=lambda x: x[1]["score"], reverse=True)
    top_results = sorted_results[:top_k]

    results: List[Dict[str, Any]] = []
    for decision_id, score_data in top_results:
        meta = score_data["meta"]
        results.append({
            "id": decision_id,
            "title": meta.get("title") if isinstance(meta, dict) else getattr(meta, "title", None),
            "chunk": meta.get("chunk") if isinstance(meta, dict) else getattr(meta, "chunk", None),
            "score": score_data["score"],
            "semantic_score": score_data["semantic_score"],
            "metadata_boost": score_data["metadata_boost"],
            "court": meta.get("court") if isinstance(meta, dict) else None,
            "judges": meta.get("judges") if isinstance(meta, dict) else None,
            "parties": meta.get("parties") if isinstance(meta, dict) else None,
        })

    return results


@mcp.tool()
def search_reasoning_paragraphs(query: str, top_k: int = 5) -> List[Dict[str, Any]]:
    """Search specifically within reasoning paragraphs for legal analysis and motivation.
    Equivalent to search_decisions(query, top_k=top_k, paragraph_type='reasoning').
    Requires ENABLE_SEARCH_TOOLS and a loaded index."""
    return search_decisions(query, top_k=top_k, paragraph_type="reasoning")


@mcp.tool()
def search_facts_paragraphs(query: str, top_k: int = 5) -> List[Dict[str, Any]]:
    """Search specifically within facts paragraphs for established circumstances.
    Equivalent to search_decisions(query, top_k=top_k, paragraph_type='facts').
    Requires ENABLE_SEARCH_TOOLS and a loaded index."""
    return search_decisions(query, top_k=top_k, paragraph_type="facts")


@mcp.tool()
def search_law_paragraphs(query: str, top_k: int = 5) -> List[Dict[str, Any]]:
    """Search specifically within law paragraphs for legal references and provisions.
    Equivalent to search_decisions(query, top_k=top_k, paragraph_type='law').
    Requires ENABLE_SEARCH_TOOLS and a loaded index."""
    return search_decisions(query, top_k=top_k, paragraph_type="law")


@mcp.tool()
def get_decision_chunks(decision_id: str, query: Optional[str] = None, top_k: int = 5, paragraph_type: Optional[str] = None) -> List[Dict[str, Any]]:
    """Return chunks for a decision; optionally rank by query and filter by paragraph type."""
    _ensure_search_enabled()
    index_loader.ensure_index_ready()

    chunks: List[Dict[str, Any]] = []
    for meta in index_loader.META:
        meta_id = meta.get("id") if isinstance(meta, dict) else getattr(meta, "id", None)
        if meta_id == decision_id:
            chunk_data = {"id": meta_id, "chunk": meta.get("chunk") if isinstance(meta, dict) else getattr(meta, "chunk", None)}

            # Filter by paragraph type if specified
            if paragraph_type:
                chunk_text = chunk_data["chunk"].lower() if chunk_data["chunk"] else ""
                paragraph_type_lower = paragraph_type.lower()

                should_include = False
                if paragraph_type_lower == "reasoning":
                    should_include = any(keyword in chunk_text for keyword in [
                        "motyv", "atsižvelg", "įvertinus", "consider", "reasoning",
                        "teismas", "konstatuoja", "nurodo"
                    ])
                elif paragraph_type_lower == "facts":
                    should_include = any(keyword in chunk_text for keyword in [
                        "nustatė", "nustatyta", "aplinky", "faktai", "facts",
                        "įrodyta", "patvirtinta"
                    ])
                elif paragraph_type_lower == "law":
                    should_include = any(keyword in chunk_text for keyword in [
                        "numato", " BK ", " CK ", " įstatym", " nuostat",
                        " Baudžiamojo kodekso ", " Civilinio kodekso "
                    ])
                else:
                    should_include = True  # Include if paragraph type not recognized

                if not should_include:
                    continue

            chunks.append(chunk_data)

    if not chunks:
        raise RuntimeError(f"No chunks found for decision_id={decision_id}")

    if query:
        query_vec = embed_query(query)
        query_lower = query.lower()

        # Hybrid scoring for chunks within this decision
        chunk_scores = []
        for i, chunk_data in enumerate(chunks):
            chunk_text = chunk_data.get("chunk", "").lower()

            # Base semantic score
            idx = None
            for meta_idx, meta in enumerate(index_loader.META):
                meta_id = meta.get("id") if isinstance(meta, dict) else getattr(meta, "id", None)
                if meta_id == decision_id:
                    chunk_meta = meta.get("chunk") if isinstance(meta, dict) else getattr(meta, "chunk", None)
                    if chunk_meta == chunk_data["chunk"]:
                        idx = meta_idx
                        break

            semantic_score = 0.0
            if idx is not None and index_loader.EMBEDDINGS.shape[0] > idx:
                semantic_score = float(index_loader.EMBEDDINGS[idx] @ query_vec)

            # Metadata boost for exact matches in chunk content
            metadata_boost = 0.0
            if query_lower in chunk_text:
                if chunk_text == query_lower:
                    metadata_boost += 2.0
                else:
                    metadata_boost += 1.0

            combined_score = semantic_score + metadata_boost
            chunk_scores.append((combined_score, chunk_data))

        # Sort by combined score
        chunk_scores.sort(key=lambda x: x[0], reverse=True)
        chunks = [chunk for _, chunk in chunk_scores[:max(1, min(top_k, len(chunk_scores)))] ]

    return chunks


@mcp.tool()
def get_decision_metadata(decision_id: str) -> Dict[str, Any]:
    """Get structured metadata for a specific decision."""
    _ensure_search_enabled()
    index_loader.ensure_index_ready()

    # Find metadata from any chunk of this decision
    for meta in index_loader.META:
        meta_id = meta.get("id") if isinstance(meta, dict) else getattr(meta, "id", None)
        if meta_id == decision_id:
            return {
                "id": meta_id,
                "title": meta.get("title"),
                "case_type": meta.get("case_type"),
                "court": meta.get("court"),
                "judges": meta.get("judges"),
                "parties": meta.get("parties"),
                "citations": meta.get("citations"),
                "tolerated_reason": meta.get("tolerated_reason"),
                "paragraph_count": meta.get("paragraph_count"),
                "date_iso": meta.get("date_iso"),
                "datetime_iso": meta.get("datetime_iso"),
            }

    raise RuntimeError(f"No metadata found for decision_id={decision_id}")


@mcp.tool()
def search_decisions_by_reason(tolerated_reason: str, top_k: int = 5) -> List[Dict[str, Any]]:
    """Search decisions by tolerated_reason classification."""
    _ensure_search_enabled()
    index_loader.ensure_index_ready()

    results: List[Dict[str, Any]] = []
    seen_ids = set()

    for meta in index_loader.META:
        if isinstance(meta, dict):
            meta_reason = meta.get("tolerated_reason")
            meta_id = meta.get("id")

            if meta_reason == tolerated_reason and meta_id and meta_id not in seen_ids:
                seen_ids.add(meta_id)
                results.append({
                    "id": meta_id,
                    "title": meta.get("title"),
                    "case_type": meta.get("case_type"),
                    "court": meta.get("court"),
                    "judges": meta.get("judges"),
                    "tolerated_reason": meta_reason,
                    "date_iso": meta.get("date_iso"),
                })

                if len(results) >= top_k:
                    break

    return results


@mcp.tool()
def search_decisions_by_court(court: str, top_k: int = 5) -> List[Dict[str, Any]]:
    """Search decisions by court name."""
    _ensure_search_enabled()
    index_loader.ensure_index_ready()

    results: List[Dict[str, Any]] = []
    seen_ids = set()

    for meta in index_loader.META:
        if isinstance(meta, dict):
            meta_court = meta.get("court", "").lower()
            meta_id = meta.get("id")

            if court.lower() in meta_court and meta_id and meta_id not in seen_ids:
                seen_ids.add(meta_id)
                results.append({
                    "id": meta_id,
                    "title": meta.get("title"),
                    "case_type": meta.get("case_type"),
                    "court": meta.get("court"),
                    "judges": meta.get("judges"),
                    "tolerated_reason": meta.get("tolerated_reason"),
                    "date_iso": meta.get("date_iso"),
                })

                if len(results) >= top_k:
                    break

    return results


@mcp.tool()
def search_decisions_by_case_type(case_type: str, top_k: int = 5) -> List[Dict[str, Any]]:
    """Search decisions by case type."""
    _ensure_search_enabled()
    index_loader.ensure_index_ready()

    results: List[Dict[str, Any]] = []
    seen_ids = set()

    for meta in index_loader.META:
        if isinstance(meta, dict):
            meta_case_type = meta.get("case_type", "").lower()
            meta_id = meta.get("id")

            if case_type.lower() in meta_case_type and meta_id and meta_id not in seen_ids:
                seen_ids.add(meta_id)
                results.append({
                    "id": meta_id,
                    "title": meta.get("title"),
                    "case_type": meta.get("case_type"),
                    "court": meta.get("court"),
                    "judges": meta.get("judges"),
                    "tolerated_reason": meta.get("tolerated_reason"),
                    "date_iso": meta.get("date_iso"),
                })

                if len(results) >= top_k:
                    break

    return results
