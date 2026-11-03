"""Retrieve document evidence and generate grounded educational answers."""

import os
from functools import lru_cache

os.environ.setdefault("USE_TF", "0")

import torch
from langchain_core.documents import Document
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

from src.embeddings import load_embedding_model
from src.pdf_loader import load_healthcare_pdf
from src.retrieval import HybridRetriever
from src.text_chunking import split_documents
from src.vector_store import create_faiss_index


GENERATION_MODEL_NAME = "google/flan-t5-base"
DISCLAIMER = (
    "Medical disclaimer: This is an educational student project based only on the "
    "provided WHO 2017 manual. It is not medical advice or a personal diagnosis, "
    "and the manual may not reflect current guidelines. Consult a qualified "
    "healthcare professional about personal medical concerns."
)
NOT_FOUND_MESSAGE = (
    "The uploaded document does not contain enough information to answer that "
    "question. Please consult a healthcare professional for personal medical concerns."
)
ANSWER_INSTRUCTIONS = (
    "You are a healthcare educational chatbot. Answer the question using only the "
    "given document context. Provide a simple and clear answer. Do not add information "
    "that is not present in the context. Include the main examples or steps listed "
    "in the context. For a how-to question, name the concrete actions from the "
    "context instead of giving only a general summary. If the answer is unavailable, say that the "
    "uploaded document does not contain the information. Do not diagnose the user or "
    "prescribe medication. For personal medical concerns, advise consulting a "
    "healthcare professional."
)
MAX_CONTEXT_CHARACTERS = 1600


@lru_cache(maxsize=1)
def get_retriever() -> HybridRetriever:
    """Load the PDF, embedding model, and search indexes once per application."""
    pages = load_healthcare_pdf()
    chunks = split_documents(pages, chunk_size=500, chunk_overlap=100)
    embedding_model = load_embedding_model()
    vector_index = create_faiss_index(
        [chunk.page_content for chunk in chunks],
        embedding_model,
    )
    return HybridRetriever(chunks, vector_index, embedding_model)


@lru_cache(maxsize=1)
def _load_generation_model():
    """Load FLAN-T5 once and keep it on CPU."""
    tokenizer = AutoTokenizer.from_pretrained(GENERATION_MODEL_NAME)
    model = AutoModelForSeq2SeqLM.from_pretrained(GENERATION_MODEL_NAME)
    model.to("cpu")
    model.eval()
    return tokenizer, model


def _prepare_context(
    documents: list[Document],
    max_documents: int = 2,
) -> tuple[str, list[Document]]:
    context_parts: list[str] = []
    used_documents: list[Document] = []
    remaining_characters = MAX_CONTEXT_CHARACTERS

    for document in documents[:max_documents]:
        if remaining_characters <= 0:
            break
        text = document.page_content[:remaining_characters]
        if text:
            context_parts.append(text)
            used_documents.append(document)
            remaining_characters -= len(text)

    return "\n\n".join(context_parts), used_documents


def generate_answer(question: str, context: str) -> str:
    """Generate an answer from a bounded context using the CPU FLAN-T5 model."""
    prompt_question = question
    if question.lstrip().casefold().startswith("how "):
        prompt_question = f"{question} Include the steps listed in the document."

    prompt = (
        f"{ANSWER_INSTRUCTIONS}\n\n"
        f"Document context:\n{context}\n\n"
        f"Question: {prompt_question}\n"
        "Include the specific information or list requested when it appears in the context.\n"
        "Answer:"
    )
    tokenizer, model = _load_generation_model()
    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=512,
    )
    with torch.inference_mode():
        output_tokens = model.generate(
            **inputs,
            max_new_tokens=160,
            num_beams=2,
        )

    answer = tokenizer.decode(output_tokens[0], skip_special_tokens=True).strip()
    return answer or NOT_FOUND_MESSAGE


def _format_source_pages(documents: list[Document]) -> str:
    page_numbers: list[str] = []
    for document in documents:
        for page_number in document.metadata.get("page_numbers", []):
            if page_number not in page_numbers:
                page_numbers.append(page_number)

    if not page_numbers:
        return "Source pages: not available"
    return "Source pages: " + ", ".join(page_numbers)


def answer_question(question: str) -> str:
    """Return a grounded answer with document pages and a safety disclaimer."""
    if not question.strip():
        return "Please enter a question."

    retrieved_documents = get_retriever().retrieve(question)
    if not retrieved_documents:
        answer = NOT_FOUND_MESSAGE
        source_line = "Source pages: no relevant passage found"
    else:
        question_lower = question.lstrip().casefold()
        max_documents = 2
        if question_lower.startswith("what is"):
            max_documents = 1
            if "hypertension" in question_lower:
                max_documents = 2
        elif "prevent" in question_lower:
            max_documents = 1
        context, used_documents = _prepare_context(
            retrieved_documents,
            max_documents=max_documents,
        )
        answer = generate_answer(question, context)
        source_line = _format_source_pages(used_documents)

    return f"{answer}\n\n{source_line}\n\n{DISCLAIMER}"
