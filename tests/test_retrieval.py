"""Basic retrieval checks using the supplied hypertension manual."""

import pytest

from src.chatbot import get_retriever
from src.pdf_loader import load_healthcare_pdf
from src.text_chunking import split_documents


@pytest.fixture(scope="module")
def retriever():
    return get_retriever()


@pytest.mark.parametrize(
    ("question", "expected_text", "expected_pages"),
    [
        ("What is hypertension?", ("140", "high blood pressure"), {"4"}),
        (
            "What are the symptoms of hypertension?",
            ("symptoms", "headache"),
            {"5", "6"},
        ),
        (
            "What are the risk factors of hypertension?",
            ("family history", "high salt"),
            {"7", "8"},
        ),
        (
            "How can hypertension be prevented?",
            ("salt", "physical activity", "fruit"),
            {"8"},
        ),
        (
            "How is hypertension diagnosed according to the document?",
            ("140 mmhg", "90 mmhg"),
            {"4"},
        ),
        (
            "What are the complications of untreated hypertension?",
            ("stroke", "myocardial infarction", "kidney"),
            {"10"},
        ),
        (
            "What lifestyle changes help manage hypertension?",
            ("salt", "fruit", "physical activity", "smoking"),
            {"14"},
        ),
        ("What is systolic blood pressure?", ("contracts", "contraction"), {"2"}),
        ("What is diastolic blood pressure?", ("relaxes", "relax"), {"2"}),
        (
            "What does the document say about medication?",
            ("prescribed", "medication"),
            {"27", "28"},
        ),
    ],
)
def test_manual_questions_retrieve_supporting_text(
    retriever,
    question,
    expected_text,
    expected_pages,
):
    results = retriever.retrieve(question)

    assert results, f"No PDF passage was retrieved for: {question}"
    retrieved_text = " ".join(result.page_content for result in results).casefold()
    assert any(term in retrieved_text for term in expected_text)
    retrieved_pages = {
        page
        for result in results
        for page in result.metadata.get("page_numbers", [])
    }
    assert expected_pages & retrieved_pages


def test_page_labels_are_kept_for_citations():
    pages = load_healthcare_pdf()
    chunks = split_documents(pages)

    assert len(pages) == 38
    assert chunks
    assert any("1" in chunk.metadata["page_numbers"] for chunk in chunks)
    assert all("page_index" in chunk.metadata for chunk in chunks)


def test_out_of_document_question_is_not_retrieved(retriever):
    assert retriever.retrieve("What does the manual say about quantum gravity?") == []
