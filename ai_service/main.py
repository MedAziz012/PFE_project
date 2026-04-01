import os
import uuid
import logging
import tempfile
from pathlib import Path

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse

from .extractor import OrangeExtractor  # absolute import — works with uvicorn

MAX_UPLOAD_SIZE_BYTES = 10 * 1024 * 1024

# ── Logging setup ─────────────────────────────────────────────────────────────
# In Docker: docker logs fastapi shows all INFO+ messages
# Set LOG_LEVEL=DEBUG env var to see debug messages during development
logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO"),
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger(__name__)

# ── App + extractor ───────────────────────────────────────────────────────────
app = FastAPI(title="Orange PF Extractor API", version="1.0.0")

# One extractor instance shared across all requests
# (patterns are compiled at class level — safe to share)
extractor = OrangeExtractor()

# ── Endpoint ──────────────────────────────────────────────────────────────────

@app.post("/analyze")
async def analyze_document(file: UploadFile = File(...)):
    """
    Upload a PDF → extract structured fields → return JSON.

    Accepts:
    - Fiche de renseignement  → extracts ref_urbanisme, dlpi, adresse, logements
    - Mandat                  → extracts syndic/promoteur fields
    """

    filename = file.filename or ""
    safe_filename = Path(filename).name

    # Guard 1: file type
    if not safe_filename.lower().endswith(".pdf"):
        raise HTTPException(
            status_code=400,
            detail=f"Only PDF files accepted. Got: {safe_filename or 'unknown'}"
        )

    # Guard 2: file size (10 MB max)
    content = await file.read()
    if len(content) > MAX_UPLOAD_SIZE_BYTES:
        raise HTTPException(
            status_code=413,
            detail="File too large. Maximum size is 10 MB."
        )

    # Use a unique temp filename — prevents collision if two requests
    # arrive simultaneously with the same filename
    suffix = f"_{uuid.uuid4().hex[:8]}.pdf"
    stem = Path(safe_filename).stem or "upload"
    tmp_path = os.path.join(tempfile.gettempdir(), f"{stem}{suffix}")

    try:
        with open(tmp_path, "wb") as f:
            f.write(content)

        logger.info("Analyzing: %s (%d bytes)", safe_filename, len(content))
        result = extractor.extract(tmp_path, safe_filename)
        logger.info(
            "Done: %s | type=%s | llm=%s",
            safe_filename,
            result.get("document_type"),
            result.get("llm_used"),
        )

        # Return 200 even for partial results — let the caller decide
        # what to do with missing fields (None values)
        return JSONResponse(status_code=200, content=result)

    except Exception as e:
        logger.exception("Unexpected error processing %s", safe_filename)
        raise HTTPException(status_code=500, detail=str(e))

    finally:
        # Always clean up — even if extraction raised an exception
        if os.path.exists(tmp_path):
            os.remove(tmp_path)


@app.get("/health")
def health():
    """Simple health check — Spring Boot calls this to verify FastAPI is up."""
    return {"status": "ok"}