"""
Multi-Format Document Loaders for Demo 4.
Extracts text and metadata from PDF, DOCX, CSV, XLSX, TXT, and Markdown files.
"""

from pathlib import Path
from typing import List, Dict, Any
import csv
import io
import uuid
from langchain_core.documents import Document


def load_pdf(file_path: Path, doc_id: str) -> List[Document]:
    """Extracts text page-by-page from a PDF file."""
    import pypdf

    docs = []
    reader = pypdf.PdfReader(str(file_path))
    for page_num, page in enumerate(reader.pages, start=1):
        text = page.extract_text() or ""
        if text.strip():
            docs.append(
                Document(
                    page_content=text.strip(),
                    metadata={
                        "doc_id": doc_id,
                        "source": file_path.name,
                        "page": page_num,
                        "format": "pdf",
                    },
                )
            )
    return docs


def load_docx(file_path: Path, doc_id: str) -> List[Document]:
    """Extracts text paragraphs and tables from a Word (.docx) file."""
    import docx

    doc = docx.Document(str(file_path))
    paragraphs = []

    # Extract paragraphs
    for p in doc.paragraphs:
        if p.text.strip():
            paragraphs.append(p.text.strip())

    # Extract tables
    for table in doc.tables:
        table_rows = []
        for row in table.rows:
            row_cells = [cell.text.strip() for cell in row.cells if cell.text.strip()]
            if row_cells:
                table_rows.append(" | ".join(row_cells))
        if table_rows:
            paragraphs.append("\n".join(table_rows))

    content = "\n\n".join(paragraphs)
    if not content.strip():
        return []

    return [
        Document(
            page_content=content,
            metadata={
                "doc_id": doc_id,
                "source": file_path.name,
                "page": 1,
                "format": "docx",
            },
        )
    ]


def load_csv(file_path: Path, doc_id: str) -> List[Document]:
    """Extracts structured text from a CSV file."""
    docs = []
    with open(file_path, mode="r", encoding="utf-8", errors="replace") as f:
        reader = csv.reader(f)
        header = next(reader, None)
        header_str = ", ".join(header) if header else ""

        for row_num, row in enumerate(reader, start=1):
            if any(cell.strip() for cell in row):
                row_text = f"Row {row_num}: " + ", ".join(
                    f"{header[i]}: {val.strip()}" if header and i < len(header) else val.strip()
                    for i, val in enumerate(row)
                )
                docs.append(
                    Document(
                        page_content=row_text,
                        metadata={
                            "doc_id": doc_id,
                            "source": file_path.name,
                            "page": row_num,
                            "format": "csv",
                        },
                    )
                )
    return docs


def load_text(file_path: Path, doc_id: str) -> List[Document]:
    """Extracts text from plain text or Markdown files."""
    text = file_path.read_text(encoding="utf-8", errors="replace")
    if not text.strip():
        return []

    return [
        Document(
            page_content=text.strip(),
            metadata={
                "doc_id": doc_id,
                "source": file_path.name,
                "page": 1,
                "format": file_path.suffix.lstrip(".").lower() or "txt",
            },
        )
    ]


def extract_document(file_path: Path, doc_id: str = None) -> List[Document]:
    """
    Dispatcher: routes to the appropriate parser based on file extension.
    """
    if not doc_id:
        doc_id = str(uuid.uuid4())

    suffix = file_path.suffix.lower()

    if suffix == ".pdf":
        return load_pdf(file_path, doc_id)
    elif suffix in [".docx", ".doc"]:
        return load_docx(file_path, doc_id)
    elif suffix == ".csv":
        return load_csv(file_path, doc_id)
    elif suffix in [".txt", ".md", ".markdown"]:
        return load_text(file_path, doc_id)
    else:
        # Fallback to plain text read
        return load_text(file_path, doc_id)
