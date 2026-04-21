"""FastAPI extraction and recommendation API."""
import asyncio
import os
import logging
import tempfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Optional

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse, RedirectResponse, Response
from extractor import OrangeExtractor
from recommendation_engine import RecommendationEngine, FolderDocuments

logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO"),
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger(__name__)

app        = FastAPI(title="Orange PF Extractor API", version="3.0.0")
extractor  = OrangeExtractor()
rec_engine = RecommendationEngine()

MAX_FILE_SIZE = 10 * 1024 * 1024
ALLOWED_SUFFIXES = {".pdf", ".png", ".jpg", ".jpeg", ".webp"}
_executor = ThreadPoolExecutor(max_workers=6, thread_name_prefix="extractor")

PRESENCE_MAP = {
    "fiche":          "fiche_present",
    "autorisation":   "autorisation_present",
    "mandat":         "mandat_present",
    "plan_situation": "plan_situation_present",
    "plan_masse":     "plan_masse_present",
    "certificat":     None,
}

_EXPECTED_FIELDS = {
    "fiche":        ["ref_urbanisme", "dlpi", "adresse", "nb_logements_residentiels", "nb_locaux_pros","nb_lots","nb_macrolots"],
    "autorisation": ["ref_urbanisme", "adresse"],
    "mandat":       ["orange_representant_nom", "orange_representant_mobile", "orange_representant_email"],
}


@app.get("/")
def root():
    return RedirectResponse(url="/docs", status_code=307)


@app.get("/favicon.ico", include_in_schema=False)
def favicon():
    return Response(status_code=204)


async def _read_and_validate(slot_name: str, upload: UploadFile) -> tuple[str, str, bytes]:
    filename = Path(upload.filename or slot_name).name
    if Path(filename).suffix.lower() not in ALLOWED_SUFFIXES:
        raise HTTPException(400, f"'{slot_name}': only PDF/PNG accepted; got '{filename}'.")
    content = await upload.read()
    size = len(content)
    if not size:
        raise HTTPException(400, f"'{slot_name}' ({filename}): file is empty.")
    if size > MAX_FILE_SIZE:
        raise HTTPException(413, f"'{slot_name}': exceeds 10 MB ({size // (1024 *1024)} MB).")
    logger.info("Validated [%s]: %s (%d KB)", slot_name, filename, size // 1024)
    return slot_name, filename, content


def _extract_from_bytes(filename: str, content: bytes) -> dict[str, Any]:
    suffix = Path(filename).suffix.lower() or ".pdf"
    fd, tmp = tempfile.mkstemp(suffix=suffix, prefix=f"{Path(filename).stem[:24]}_")
    os.close(fd)
    try:
        with open(tmp, "wb") as f:
            f.write(content)
        return extractor.extract(tmp, filename)
    finally:
        if os.path.exists(tmp):
            os.remove(tmp)


async def _extract_async(slot_name: str, filename: str, content: bytes) -> tuple[str, dict[str, Any]]:
    result = await asyncio.get_running_loop().run_in_executor(_executor, _extract_from_bytes, filename, content)
    logger.info("Extracted [%s]: %s -> type=%s", slot_name, filename, result.get("document_type"))
    return filename, result

@app.post("/analyze-batch")
async def analyze_batch(
    fiche:          Optional[UploadFile] = File(None),
    autorisation:   Optional[UploadFile] = File(None),
    plan_situation: Optional[UploadFile] = File(None),
    plan_masse:     Optional[UploadFile] = File(None),
    mandat:         Optional[UploadFile] = File(None),
    certificat:     Optional[UploadFile] = File(None),
):
    named = [(k, v) for k, v in [("fiche", fiche), ("autorisation", autorisation), 
                                   ("plan_situation", plan_situation), ("plan_masse", plan_masse),
                                   ("mandat", mandat), ("certificat", certificat)] if v]
    if not named:
        raise HTTPException(400, "At least one file must be uploaded.")
    validated = await asyncio.gather(*[_read_and_validate(sn, up) for sn, up in named])
    pairs = await asyncio.gather(*[_extract_async(sn, fn, ct) for sn, fn, ct in validated])
    folder = _build_folder_documents(dict(pairs))
    return JSONResponse(content={
        "documents": dict(pairs),
        "recommendation": rec_engine.to_dict(rec_engine.analyze(folder)),
    }, status_code=200)

# Fields expected per document type — used to detect extraction failures.
# A failure = document is present but the extractor returned None for that field.
_EXPECTED_FIELDS: dict[str, list[str]] = {
    "fiche":        ["ref_urbanisme", "dlpi", "adresse",
                     "nb_logements_residentiels", "nb_locaux_pros", "nb_lots","nb_macrolots"],
    "autorisation": ["ref_urbanisme", "adresse"],
    "mandat":       ["orange_representant_nom",
                     "orange_representant_mobile", "orange_representant_email"],
}


def _build_folder_documents(results: dict[str, dict]) -> FolderDocuments:
    """Aggregate extraction results and track failures."""
    folder = FolderDocuments()
    failures_map = getattr(folder, "extraction_failures", None)
    if not isinstance(failures_map, dict):
        failures_map = {}
        setattr(folder, "extraction_failures", failures_map)

    for _, data in results.items():
        doc_type = data.get("document_type")
        is_present = bool(doc_type) and not data.get("error")
        
        if PRESENCE_MAP.get(doc_type):
            setattr(folder, PRESENCE_MAP[doc_type], is_present)

        if not is_present:
            continue

        # Track extraction failures (None values for expected fields)
        failures = [f for f in _EXPECTED_FIELDS.get(doc_type, []) if data.get(f) is None]
        if failures:
            failures_map[doc_type] = failures

        if doc_type == "fiche":
            folder.fiche_ref_urbanisme = data.get("ref_urbanisme")
            folder.fiche_dlpi = data.get("dlpi")
            folder.fiche_adresse = data.get("adresse")
            try:
                folder.fiche_nb_logements_res = int(data["nb_logements_residentiels"])
            except (TypeError, ValueError, KeyError):
                pass
            try:
                folder.fiche_nb_locaux_pro = int(data["nb_locaux_pros"])
            except (TypeError, ValueError, KeyError):
                pass
        elif doc_type == "autorisation":
            folder.autorisation_ref_urbanisme = data.get("ref_urbanisme")
            folder.autorisation_adresse = data.get("adresse")
        elif doc_type == "mandat":
            folder.mandat_orange_rep_nom = data.get("orange_representant_nom")
            folder.mandat_orange_rep_mobile = data.get("orange_representant_mobile")
            folder.mandat_orange_rep_email = data.get("orange_representant_email")

    return folder
@app.get("/health")
def health():
    return {"status": "ok", "version": "3.0.0"}