import pdfplumber
import pytesseract
from pdf2image import convert_from_path
from langchain_core.documents import Document
from pathlib import Path


def load_pdf(path: Path):
    documents = []

    # 1️⃣ Try normal text extraction
    with pdfplumber.open(path) as pdf:
        for page_num, page in enumerate(pdf.pages):
            text = page.extract_text()

            if text and text.strip():
                documents.append(
                    Document(
                        page_content=text,
                        metadata={
                            "source": path.name,
                            "page": page_num,
                            "method": "text"
                        }
                    )
                )

    # 2️⃣ If no text found → OCR
    if not documents:
        images = convert_from_path(path)

        for page_num, image in enumerate(images):
            text = pytesseract.image_to_string(image)

            if text.strip():
                documents.append(
                    Document(
                        page_content=text,
                        metadata={
                            "source": path.name,
                            "page": page_num,
                            "method": "ocr"
                        }
                    )
                )

    return documents
