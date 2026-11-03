"""Load the project PDF and keep its page metadata."""

from pathlib import Path

from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PDF_PATH = PROJECT_ROOT / "data" / "health.pdf"


def load_healthcare_pdf(pdf_path: str | Path = DEFAULT_PDF_PATH) -> list[Document]:
    """Load one PDF page per Document, including its original page metadata."""
    path = Path(pdf_path)
    if not path.is_file():
        raise FileNotFoundError(
            f"Healthcare PDF not found at {path}. Add the manual as data/health.pdf."
        )

    documents = PyPDFLoader(str(path)).load()
    if not documents:
        raise ValueError(f"No pages could be read from {path}.")
    return documents
