"""
AI Data Analyst Agent - File Ingestion Layer
Handles: CSV, Excel, Image (OCR), Audio (Whisper), PDF
"""

import pandas as pd
from pathlib import Path
import pytesseract
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'


def load_csv(file) -> pd.DataFrame:
    return pd.read_csv(file)


def load_excel(file) -> pd.DataFrame:
    xl = pd.ExcelFile(file)
    sheet = xl.sheet_names[0]
    return pd.read_excel(file, sheet_name=sheet)


def load_image_as_text(file) -> str:
    try:
        from PIL import Image
        import pytesseract
        img = Image.open(file)
        return pytesseract.image_to_string(img)
    except ImportError:
        return "[pytesseract not installed]"
    except Exception as e:
        return f"[OCR error: {e}]"


def load_audio_as_text(file) -> str:
    try:
        import whisper
        import tempfile
        import os

        # Save uploaded file to a temp file
        suffix = ".mp3"
        if hasattr(file, 'name'):
            suffix = os.path.splitext(file.name)[-1]

        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(file.read())
            tmp_path = tmp.name

        # Transcribe with Whisper
        model = whisper.load_model("base")
        result = model.transcribe(tmp_path)

        # Cleanup temp file
        os.unlink(tmp_path)

        return result["text"]

    except ImportError:
        return "[Whisper not installed — run: pip install openai-whisper]"
    except Exception as e:
        return f"[Audio transcription error: {e}]"


def load_pdf_as_text(file) -> str:
    try:
        import pdfplumber
        text_parts = []
        with pdfplumber.open(file) as pdf:
            for page in pdf.pages:
                text_parts.append(page.extract_text() or "")
        return "\n".join(text_parts)
    except ImportError:
        return "[pdfplumber not installed]"
    except Exception as e:
        return f"[PDF error: {e}]"


def detect_and_load(file, filename: str):
    ext = Path(filename).suffix.lower()
    if ext == ".csv":
        return load_csv(file), "csv"
    elif ext in [".xlsx", ".xls"]:
        return load_excel(file), "excel"
    elif ext in [".png", ".jpg", ".jpeg", ".bmp", ".tiff"]:
        return load_image_as_text(file), "image"
    elif ext in [".mp3", ".wav", ".m4a", ".ogg"]:
        return load_audio_as_text(file), "audio"
    elif ext == ".pdf":
        return load_pdf_as_text(file), "pdf"
    else:
        return None, "unsupported"