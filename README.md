# Healthcare Information Chatbot Using LLMs and RAG

A beginner-friendly Python project that answers basic questions about hypertension using passages retrieved from a healthcare PDF. It combines semantic search and keyword search, asks FLAN-T5-base to answer from the retrieved context, and shows the manual's printed page references.

The supplied manual is the WHO 2017 *Diagnosis and management for patients with hypertension* education manual. It is an educational source, not a source of current personal medical advice.

## Problem statement

Healthcare information can be difficult to find in a long PDF. This project demonstrates how retrieval-augmented generation (RAG) can locate relevant pages and produce a short, document-grounded answer instead of asking a model to answer from general knowledge.

## Objectives

- Load the local manual page by page with PyPDFLoader.
- Preserve the manual's printed page labels for citations.
- Split text into overlapping chunks and skip reference lists.
- Search chunks with Sentence Transformers, FAISS, and TF-IDF.
- Generate answers with `google/flan-t5-base` using only retrieved text.
- Clearly say when the PDF does not contain enough information.
- Provide a simple Gradio chat interface with a medical disclaimer.

## Technologies

Python, Jupyter Notebook, LangChain `PyPDFLoader`, `RecursiveCharacterTextSplitter`, Sentence Transformers (`all-MiniLM-L6-v2`), FAISS, scikit-learn TF-IDF, Hugging Face Transformers (`google/flan-t5-base`), and Gradio.

The models and FAISS index run locally on CPU. The first run may download model weights from Hugging Face if they are not already cached.

## Workflow

1. Read `data/health.pdf` one page at a time and retain page metadata.
2. Remove reference-list text, then split the page text into 500-character chunks with 100-character overlap.
3. Generate normalized sentence embeddings and build an in-memory FAISS index.
4. Fit a TF-IDF vectorizer over the same chunks.
5. Combine semantic and keyword ranks, include relevant adjacent chunks, and reject queries with no matching evidence.
6. Limit the context to fit FLAN-T5-base's 512-token input limit.
7. Generate an answer, append source page labels, and show the medical disclaimer.

## Project structure

```text
Healthcare-Chatbot-RAG/
├── data/
│   └── health.pdf
├── notebooks/
│   └── Healthcare_Chatbot_RAG.ipynb
├── src/
│   ├── __init__.py
│   ├── pdf_loader.py
│   ├── text_chunking.py
│   ├── embeddings.py
│   ├── vector_store.py
│   ├── retrieval.py
│   └── chatbot.py
├── tests/
│   └── test_retrieval.py
├── app.py
├── requirements.txt
├── README.md
└── .gitignore
```

### What each Python file does

- `pdf_loader.py`: opens the PDF and returns page documents with page metadata.
- `text_chunking.py`: removes reference lists and creates overlapping chunks while retaining printed page numbers.
- `embeddings.py`: loads the CPU sentence embedding model and encodes text.
- `vector_store.py`: creates and searches the FAISS index.
- `retrieval.py`: combines FAISS semantic search with TF-IDF keyword ranking.
- `chatbot.py`: limits context, loads FLAN-T5 once, creates answers, and formats sources and disclaimers.
- `app.py`: displays the multi-turn Gradio chat interface.
- `notebooks/Healthcare_Chatbot_RAG.ipynb`: demonstrates the pipeline in small, runnable stages.

## Setup in VS Code

Open the project folder in VS Code. Use the integrated terminal and select Python 3.10 or later (Python 3.12 is supported).

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Place the manual at `data/health.pdf`. The original `health_info.pdf` found in the workspace was preserved; a copy was placed at `data/health.pdf`.

## Run the application

From the project root:

```powershell
python app.py
```

Open the local URL printed by Gradio. Use the examples or enter your own questions. The same chat session accepts multiple questions; use **Clear conversation** to start over.

## Run the notebook

Open `notebooks/Healthcare_Chatbot_RAG.ipynb` in VS Code and select the project's Python kernel. Run the cells from top to bottom. The notebook displays the PDF page count, chunk text and page metadata, vector and keyword setup, retrieved evidence, and generated answers.

## Test the project

Run the retrieval tests from the project root:

```powershell
python -m pytest -q
```

The tests cover all ten requested hypertension questions, page metadata, and a question that is outside the manual's subject. They check retrieved PDF evidence and source pages; generated prose can vary slightly with model versions.

Validation in the supplied workspace: PyPDFLoader loaded 38 PDF pages, the Gradio application imported successfully, and all 12 retrieval tests passed. All ten requested questions were also run through FLAN-T5-base with CPU inference and checked against the cited passages.

## Sample questions and answers

These are examples produced from the supplied manual. The wording can vary slightly between model versions.

| Question | Example document-based answer | Printed source pages |
|---|---|---|
| What is hypertension? | The manual describes diagnosis using an average systolic pressure of at least 140 mmHg or an average diastolic pressure of at least 90 mmHg on two or more separate days. | 3, 4 |
| What are the symptoms of hypertension? | Many people have no symptoms. Possible symptoms listed include headache, palpitation, and excess sweating. | 5, 6 |
| How can hypertension be prevented? | Manage modifiable risk factors through weight control, physical activity, eating less salt, quitting smoking, consuming less alcohol, and managing emotional stress. | 8 |
| How is hypertension diagnosed according to the document? | The manual says an average systolic pressure of 140 mmHg or higher, or an average diastolic pressure of 90 mmHg or higher, on two or more separate days. | 4 |
| What does the document say about medication? | Take medication as prescribed and regularly; consult a physician before starting or stopping it. | 27, 28 |

The bot cites the printed page or pages from which it retrieved its answer. If no relevant passage is found, it says that the uploaded document does not contain enough information.

## Medical safety and limitations

- This is a student educational project, not a medical advice or diagnostic application.
- Answers are restricted to the provided WHO 2017 manual. Its historical recommendations may not match current guidelines.
- The bot must not provide personal diagnoses or prescribe medication. Users should consult healthcare professionals for personal medical concerns.
- PDF text extraction may miss information in image-only pages or complex layouts.
- Retrieval and generated text can still be incomplete or incorrect. Check the cited passages in the PDF.
- The FAISS index and TF-IDF matrix are rebuilt in memory when the app starts; this simple project does not use a database.
- CPU inference for FLAN-T5-base can be slow, and model weights require disk space.

## Future enhancements

- Add a small evaluation set with human-checked expected evidence and citation pages.
- Improve PDF table and scanned-page extraction.
- Add user-selectable sources and a way to open the cited page.
- Compare retrieval settings and answer quality without changing the source document.
