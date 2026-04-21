"""FastAPI extraction and recommendation API — async job queue edition."""
import asyncio
import os
import logging
import tempfile
import uuid
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from fastapi import FastAPI, BackgroundTasks, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse, RedirectResponse, Response
from extractor import OrangeExtractor
from recommendation_engine import RecommendationEngine, FolderDocuments

logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO"),
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger(__name__)

app        = FastAPI(title="Orange PF Extractor API", version="4.0.0")
extractor  = OrangeExtractor()
rec_engine = RecommendationEngine()

MAX_FILE_SIZE   = 10 * 1024 * 1024
ALLOWED_SUFFIXES = {".pdf", ".png", ".jpg", ".jpeg", ".webp"}
_executor = ThreadPoolExecutor(max_workers=6, thread_name_prefix="extractor")

# ---------------------------------------------------------------------------
# In-memory job store  {job_id: JobRecord}
# For production, replace with Redis / a small DB.
# ---------------------------------------------------------------------------
# JobRecord shape:
#   status : "pending" | "processing" | "done" | "error"
#   created_at : ISO-8601 string
#   result  : dict | None   (populated on "done")
#   error   : str  | None   (populated on "error")
_jobs: dict[str, dict[str, Any]] = {}

# ---------------------------------------------------------------------------
# Document / folder constants
# ---------------------------------------------------------------------------
PRESENCE_MAP = {
    "fiche":          "fiche_present",
    "autorisation":   "autorisation_present",
    "mandat":         "mandat_present",
    "plan_situation": "plan_situation_present",
    "plan_masse":     "plan_masse_present",
    "certificat":     None,
}

_EXPECTED_FIELDS: dict[str, list[str]] = {
    "fiche":        ["ref_urbanisme", "dlpi", "adresse",
                     "nb_logements_residentiels", "nb_locaux_pros", "nb_lots", "nb_macrolots"],
    "autorisation": ["ref_urbanisme", "adresse"],
    "mandat":       ["orange_representant_nom",
                     "orange_representant_mobile", "orange_representant_email"],
}

# ---------------------------------------------------------------------------
# Utility routes
# ---------------------------------------------------------------------------

@app.get("/")
def root():
    return RedirectResponse(url="/docs", status_code=307)


@app.get("/favicon.ico", include_in_schema=False)
def favicon():
    return Response(status_code=204)


@app.get("/health")
def health():
    return {"status": "ok", "version": "4.0.0"}


# ---------------------------------------------------------------------------
# Validation + extraction helpers  (unchanged logic, same as v3)
# ---------------------------------------------------------------------------

async def _read_and_validate(slot_name: str, upload: UploadFile) -> tuple[str, str, bytes]:
    filename = Path(upload.filename or slot_name).name
    if Path(filename).suffix.lower() not in ALLOWED_SUFFIXES:
        raise HTTPException(400, f"'{slot_name}': only PDF/PNG accepted; got '{filename}'.")
    content = await upload.read()
    size = len(content)
    if not size:
        raise HTTPException(400, f"'{slot_name}' ({filename}): file is empty.")
    if size > MAX_FILE_SIZE:
        raise HTTPException(413, f"'{slot_name}': exceeds 10 MB ({size // (1024 * 1024)} MB).")
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
    result = await asyncio.get_running_loop().run_in_executor(
        _executor, _extract_from_bytes, filename, content
    )
    logger.info("Extracted [%s]: %s -> type=%s", slot_name, filename, result.get("document_type"))
    return filename, result


def _build_folder_documents(results: dict[str, dict]) -> FolderDocuments:
    """Aggregate extraction results and track failures."""
    folder = FolderDocuments()
    failures_map: dict[str, list[str]] = {}
    setattr(folder, "extraction_failures", failures_map)

    for _, data in results.items():
        doc_type   = data.get("document_type")
        is_present = bool(doc_type) and not data.get("error")

        if PRESENCE_MAP.get(doc_type):
            setattr(folder, PRESENCE_MAP[doc_type], is_present)

        if not is_present:
            continue

        failures = [f for f in _EXPECTED_FIELDS.get(doc_type, []) if data.get(f) is None]
        if failures:
            failures_map[doc_type] = failures

        if doc_type == "fiche":
            folder.fiche_ref_urbanisme = data.get("ref_urbanisme")
            folder.fiche_dlpi          = data.get("dlpi")
            folder.fiche_adresse       = data.get("adresse")
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
            folder.autorisation_adresse       = data.get("adresse")
        elif doc_type == "mandat":
            folder.mandat_orange_rep_nom    = data.get("orange_representant_nom")
            folder.mandat_orange_rep_mobile = data.get("orange_representant_mobile")
            folder.mandat_orange_rep_email  = data.get("orange_representant_email")

    return folder


# ---------------------------------------------------------------------------
# Background worker — runs after the HTTP response is already sent
# ---------------------------------------------------------------------------

async def _run_job(job_id: str, validated: list[tuple[str, str, bytes]]) -> None:
    """Perform OCR + LLM extraction in the background and store the result."""
    _jobs[job_id]["status"] = "processing"
    logger.info("Job %s — starting extraction for %d file(s)", job_id, len(validated))

    try:
        pairs = await asyncio.gather(
            *[_extract_async(sn, fn, ct) for sn, fn, ct in validated]
        )
        results: dict[str, dict] = dict(pairs)
        folder = _build_folder_documents(results)

        _jobs[job_id].update(
            status="done",
            result={
                "documents":      results,
                "recommendation": rec_engine.to_dict(rec_engine.analyze(folder)),
            },
        )
        logger.info("Job %s — done", job_id)

    except Exception as exc:  # noqa: BLE001
        logger.exception("Job %s — failed: %s", job_id, exc)
        _jobs[job_id].update(status="error", error=str(exc))


# ---------------------------------------------------------------------------
# POST /analyze-batch  →  returns a job_id immediately (no waiting)
# ---------------------------------------------------------------------------

@app.post("/analyze-batch", status_code=202)
async def analyze_batch(
    background_tasks: BackgroundTasks,
    fiche:            Optional[UploadFile] = File(None),
    autorisation:     Optional[UploadFile] = File(None),
    plan_situation:   Optional[UploadFile] = File(None),
    plan_masse:       Optional[UploadFile] = File(None),
    mandat:           Optional[UploadFile] = File(None),
    certificat:       Optional[UploadFile] = File(None),
):
    """
    Accept files, validate them synchronously, then kick off extraction in
    the background.  Returns a `job_id` immediately (HTTP 202 Accepted).

    Poll **GET /status/{job_id}** until `status == "done"` or `"error"`.
    """
    named = [
        (k, v) for k, v in [
            ("fiche",          fiche),
            ("autorisation",   autorisation),
            ("plan_situation", plan_situation),
            ("plan_masse",     plan_masse),
            ("mandat",         mandat),
            ("certificat",     certificat),
        ] if v
    ]
    if not named:
        raise HTTPException(400, "At least one file must be uploaded.")

    # Validation is fast (just reads bytes + checks size/ext) — do it now so
    # the caller gets an error immediately instead of discovering it later.
    validated: list[tuple[str, str, bytes]] = await asyncio.gather(
        *[_read_and_validate(sn, up) for sn, up in named]
    )

    # Create the job record BEFORE scheduling the task so the status endpoint
    # can never return 404 for a job that has already started.
    job_id = str(uuid.uuid4())
    _jobs[job_id] = {
        "status":     "pending",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "result":     None,
        "error":      None,
    }

    # Schedule the heavy work to run after this response is sent.
    background_tasks.add_task(_run_job, job_id, validated)

    logger.info("Job %s — queued (%d file(s))", job_id, len(validated))
    return JSONResponse(
        status_code=202,
        content={
            "job_id":    job_id,
            "status":    "pending",
            "poll_url":  f"/status/{job_id}",
        },
    )


# ---------------------------------------------------------------------------
# GET /status/{job_id}  →  poll for results
# ---------------------------------------------------------------------------

@app.get("/status/{job_id}")
def job_status(job_id: str):
    """
    Poll this endpoint every few seconds after calling POST /analyze-batch.

    Possible `status` values:
    - **pending**    — queued, not yet started
    - **processing** — OCR / LLM extraction in progress
    - **done**       — `result` contains `documents` + `recommendation`
    - **error**      — `error` field explains what went wrong
    """
    job = _jobs.get(job_id)
    if job is None:
        raise HTTPException(404, f"Job '{job_id}' not found.")

    # Only include heavyweight fields when work is finished
    response: dict[str, Any] = {
        "job_id":     job_id,
        "status":     job["status"],
        "created_at": job["created_at"],
    }
    if job["status"] == "done":
        response["result"] = job["result"]
    elif job["status"] == "error":
        response["error"] = job["error"]

    return JSONResponse(content=response)