"""Combine FAISS semantic search with TF-IDF keyword search."""

import re
from collections.abc import Iterable

from langchain_core.documents import Document
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS, TfidfVectorizer

from src.vector_store import search_faiss


class HybridRetriever:
    """Search the PDF chunks with semantic and keyword matching."""

    def __init__(
        self,
        chunks: list[Document],
        vector_index,
        embedding_model: SentenceTransformer,
    ) -> None:
        if not chunks:
            raise ValueError("The PDF did not contain any searchable text.")

        self.chunks = chunks
        self.vector_index = vector_index
        self.embedding_model = embedding_model
        stop_words = ENGLISH_STOP_WORDS.union(
            {"according", "document", "information", "manual", "say", "says"}
        )
        self.keyword_vectorizer = TfidfVectorizer(
            stop_words=sorted(stop_words),
            ngram_range=(1, 2),
            sublinear_tf=True,
        )
        self.keyword_matrix = self.keyword_vectorizer.fit_transform(
            [chunk.page_content for chunk in chunks]
        )

    def retrieve(self, question: str, top_k: int = 3) -> list[Document]:
        """Return relevant chunks, joining adjacent chunks from the same page."""
        if not question.strip() or top_k <= 0:
            return []

        search_question = _prepare_search_question(question)
        search_count = min(max(top_k * 3, 10), len(self.chunks))
        semantic_results = search_faiss(
            self.vector_index,
            search_question,
            self.embedding_model,
            search_count,
        )
        semantic_scores = dict(semantic_results)

        query_vector = self.keyword_vectorizer.transform([search_question])
        if query_vector.nnz == 0:
            return []

        keyword_scores_array = (self.keyword_matrix @ query_vector.T).toarray().ravel()
        keyword_scores = {
            position: float(score)
            for position, score in enumerate(keyword_scores_array)
        }

        best_semantic_score = max(semantic_scores.values(), default=0.0)
        best_keyword_score = max(keyword_scores.values(), default=0.0)
        if best_semantic_score < 0.30 and best_keyword_score < 0.04:
            return []

        keyword_order = sorted(
            (position for position, score in keyword_scores.items() if score > 0),
            key=keyword_scores.__getitem__,
            reverse=True,
        )[:search_count]
        candidates = set(semantic_scores) | set(keyword_order)
        best_semantic_score = max(semantic_scores.values(), default=0.0)
        best_keyword_score = max(
            (keyword_scores[position] for position in keyword_order),
            default=0.0,
        )
        fused_scores = {
            position: 0.35
            * semantic_scores.get(position, 0.0)
            / best_semantic_score
            + 0.65
            * keyword_scores.get(position, 0.0)
            / best_keyword_score
            for position in candidates
        }
        anchors = sorted(candidates, key=fused_scores.__getitem__, reverse=True)[:top_k]

        selected_positions = set(anchors)
        for position in anchors:
            for neighbor in (position - 1, position + 1):
                if not 0 <= neighbor < len(self.chunks):
                    continue
                same_page = (
                    self.chunks[position].metadata.get("page_index")
                    == self.chunks[neighbor].metadata.get("page_index")
                )
                neighbor_is_relevant = (
                    semantic_scores.get(neighbor, 0.0) >= 0.18
                    or keyword_scores.get(neighbor, 0.0) >= 0.02
                )
                if same_page and neighbor_is_relevant:
                    selected_positions.add(neighbor)

        groups = self._join_adjacent_chunks(selected_positions, fused_scores)
        return sorted(
            groups,
            key=lambda document: document.metadata["relevance_score"],
            reverse=True,
        )

    def _join_adjacent_chunks(
        self,
        positions: Iterable[int],
        fused_scores: dict[int, float],
    ) -> list[Document]:
        ordered_positions = sorted(positions)
        groups: list[list[int]] = []
        for position in ordered_positions:
            if (
                groups
                and position == groups[-1][-1] + 1
                and self.chunks[position].metadata.get("page_index")
                == self.chunks[groups[-1][-1]].metadata.get("page_index")
            ):
                groups[-1].append(position)
            else:
                groups.append([position])

        results: list[Document] = []
        for group in groups:
            text = self.chunks[group[0]].page_content
            page_numbers: list[str] = []
            for position in group:
                chunk = self.chunks[position]
                if position != group[0]:
                    text = _append_without_overlap(text, chunk.page_content)
                for page_number in chunk.metadata.get("page_numbers", []):
                    if page_number not in page_numbers:
                        page_numbers.append(page_number)

            metadata = dict(self.chunks[group[0]].metadata)
            metadata["page_numbers"] = page_numbers
            metadata["chunk_ids"] = [
                self.chunks[position].metadata["chunk_id"] for position in group
            ]
            metadata["relevance_score"] = max(
                (fused_scores.get(position, 0.0) for position in group),
                default=0.0,
            )
            results.append(Document(page_content=text, metadata=metadata))
        return results


def _prepare_search_question(question: str) -> str:
    """Make small wording adjustments so related word forms can match the PDF."""
    search_question = re.sub(
        r"\bprevent(?:ed|ing|ion|ive)\b",
        "prevent",
        question,
        flags=re.IGNORECASE,
    )
    if re.search(r"(?i)\bhow\b.*\bprevent\b", search_question):
        search_question = f"{search_question} risk factors"
    if re.match(r"(?i)^\s*what\s+is\b", search_question):
        if re.search(r"(?i)\bhypertension\b", search_question):
            search_question = f"{search_question} diagnosed diagnostic criteria"
        elif re.search(r"(?i)\b(?:systolic|diastolic)\b", search_question):
            search_question = f"{search_question} measured"
        else:
            search_question = f"{search_question} measured"
    return search_question


def _append_without_overlap(existing_text: str, next_text: str) -> str:
    """Join overlapping splitter output without repeating the shared text."""
    maximum_overlap = min(len(existing_text), len(next_text), 250)
    for overlap_size in range(maximum_overlap, 0, -1):
        if (
            existing_text[-overlap_size:].casefold()
            == next_text[:overlap_size].casefold()
        ):
            return existing_text + next_text[overlap_size:]
    return existing_text + "\n" + next_text
