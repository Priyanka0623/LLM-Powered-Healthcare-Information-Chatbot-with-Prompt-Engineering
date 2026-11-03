"""Clean manual pages and split them into overlapping text chunks."""

import re

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter


def _remove_references(text: str) -> str:
    """Remove a reference list appended to an educational page."""
    match = re.search(r"(?im)^\s*REFERENCES?\s*:?", text)
    if match:
        text = text[: match.start()]

    text = re.sub(
        r"(?im)^\s*FOR\s+(?:PATIENTS|PHYSICIANS)\s+\d+\s*$",
        "",
        text,
    )
    text = re.sub(
        r"(?im)^\s*Diagnosis and management for patients with hypertension\s*$",
        "",
        text,
    )
    return text.strip()


def split_documents(
    documents: list[Document],
    chunk_size: int = 500,
    chunk_overlap: int = 100,
) -> list[Document]:
    """Split pages without losing printed page numbers or bullet text."""
    if chunk_size <= 0 or chunk_overlap < 0 or chunk_overlap >= chunk_size:
        raise ValueError("Use a positive chunk size and overlap smaller than the size.")

    cleaned_pages: list[Document] = []
    for document in documents:
        if re.search(r"(?im)^\s*Table of contents\s*$", document.page_content):
            continue

        page_label = str(document.metadata.get("page_label", "")).strip()
        if page_label and re.fullmatch(r"(?i)[ivxlcdm]+", page_label):
            continue

        text = _remove_references(document.page_content)
        if not text:
            continue
        if " ".join(text.casefold().split()) == (
            "managing hypertension healthy lifestyle medication"
        ):
            continue

        metadata = dict(document.metadata)
        printed_page = str(metadata.get("page_label", "")).strip()
        if not printed_page or printed_page.lower() == "none":
            printed_page = str(int(metadata.get("page", 0)) + 1)

        metadata["page_number"] = printed_page
        metadata["page_numbers"] = [printed_page]
        metadata["page_index"] = int(metadata.get("page", 0)) + 1
        cleaned_pages.append(Document(page_content=text, metadata=metadata))

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        add_start_index=True,
    )
    split_chunks = splitter.split_documents(cleaned_pages)

    unique_chunks: list[Document] = []
    known_chunks: dict[str, Document] = {}
    for chunk in split_chunks:
        normalized_text = " ".join(chunk.page_content.casefold().split())
        if not normalized_text:
            continue

        existing = known_chunks.get(normalized_text)
        if existing is not None:
            page_numbers = existing.metadata["page_numbers"]
            page_number = chunk.metadata["page_number"]
            if page_number not in page_numbers:
                page_numbers.append(page_number)
            continue

        chunk.metadata["chunk_id"] = len(unique_chunks)
        known_chunks[normalized_text] = chunk
        unique_chunks.append(chunk)

    return unique_chunks
