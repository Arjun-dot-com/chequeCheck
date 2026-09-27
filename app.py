"""FastAPI entry point for the ChequeCheck CV/ML service."""

from __future__ import annotations

import json
import logging
import shutil
from pathlib import Path
from threading import Lock
from typing import Final
from uuid import uuid4

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.concurrency import run_in_threadpool
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from cv_core.pipeline import ChequeProcessingPipeline


LOGGER = logging.getLogger(__name__)
PROJECT_ROOT: Final[Path] = Path(__file__).resolve().parent
UPLOAD_DIR: Final[Path] = PROJECT_ROOT / "temp_uploads"
AUDIT_FILE: Final[Path] = PROJECT_ROOT / "bank_data" / "audit_log.json"
CHEQUE_EXTENSIONS: Final[frozenset[str]] = frozenset(
    {".png", ".jpg", ".jpeg", ".pdf"}
)
REFERENCE_EXTENSIONS: Final[frozenset[str]] = frozenset(
    {".png", ".jpg", ".jpeg"}
)

UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

app = FastAPI(
    title="ChequeCheck CV API",
    description="API for processing cheque images with YOLO, TrOCR, and SNNs.",
    version="2.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Construct the model-owning pipeline once at module startup. The route only
# reuses this instance and never allocates new YOLO, TrOCR, or SNN models.
try:
    pipeline: ChequeProcessingPipeline | None = ChequeProcessingPipeline()
except Exception:
    LOGGER.exception("Could not initialize the cheque-processing pipeline")
    pipeline = None

# Shared inference models are read-only, but serializing whole-pipeline calls
# prevents concurrent CPU requests from causing avoidable peak-memory spikes.
PIPELINE_LOCK = Lock()


def _validate_upload(
    upload: UploadFile,
    allowed_extensions: frozenset[str],
    label: str,
) -> None:
    """Validate an uploaded filename without trusting it as a disk path."""
    suffix = Path(upload.filename or "").suffix.lower()
    if not upload.filename or suffix not in allowed_extensions:
        supported = ", ".join(sorted(allowed_extensions))
        raise HTTPException(
            status_code=400,
            detail=f"Invalid {label} file type. Supported extensions: {supported}.",
        )


def _save_upload(upload: UploadFile) -> Path:
    """Persist an upload under a random server-generated temporary filename."""
    suffix = Path(upload.filename or "").suffix.lower()
    destination = UPLOAD_DIR / f"{uuid4().hex}{suffix}"
    try:
        with destination.open("wb") as buffer:
            shutil.copyfileobj(upload.file, buffer)
    except Exception:
        destination.unlink(missing_ok=True)
        raise
    return destination


def _run_pipeline(
    cheque_path: Path,
    signature_path: Path | None,
    amount_path: Path | None,
) -> dict[str, object]:
    """Run inference under the process-wide model lock."""
    if pipeline is None:  # Guard retained for static type narrowing.
        raise RuntimeError("Pipeline is not initialized")
    with PIPELINE_LOCK:
        return pipeline.process_cheque(
            image_path=str(cheque_path),
            reference_signature_path=(
                str(signature_path) if signature_path is not None else None
            ),
            reference_amount_path=(
                str(amount_path) if amount_path is not None else None
            ),
        )


@app.get("/audit")
async def get_audit_log() -> list[object]:
    """Return the existing audit log when it contains valid JSON."""
    if not AUDIT_FILE.exists():
        return []
    try:
        with AUDIT_FILE.open("r", encoding="utf-8") as audit_file:
            content = json.load(audit_file)
        return content if isinstance(content, list) else []
    except (json.JSONDecodeError, OSError):
        return []


@app.get("/health")
async def health_check() -> dict[str, str | bool]:
    """Report process liveness and whether every ML model loaded successfully."""
    return {"status": "ok", "pipeline_ready": pipeline is not None}


@app.post("/scan")
async def scan_cheque(
    file: UploadFile = File(...),
    reference_signature: UploadFile | None = File(default=None),
    reference_amount: UploadFile | None = File(default=None),
) -> JSONResponse:
    """Process a cheque with optional signature and cursive-amount references."""
    if pipeline is None:
        raise HTTPException(
            status_code=503,
            detail="Pipeline not initialized. Check all model dependencies and weights.",
        )

    _validate_upload(file, CHEQUE_EXTENSIONS, "cheque")
    if reference_signature is not None:
        _validate_upload(
            reference_signature, REFERENCE_EXTENSIONS, "reference signature"
        )
    if reference_amount is not None:
        _validate_upload(reference_amount, REFERENCE_EXTENSIONS, "reference amount")

    saved_paths: list[Path] = []
    uploads = [
        upload
        for upload in (file, reference_signature, reference_amount)
        if upload is not None
    ]

    try:
        cheque_path = await run_in_threadpool(_save_upload, file)
        saved_paths.append(cheque_path)

        signature_path: Path | None = None
        if reference_signature is not None:
            signature_path = await run_in_threadpool(
                _save_upload, reference_signature
            )
            saved_paths.append(signature_path)

        amount_path: Path | None = None
        if reference_amount is not None:
            amount_path = await run_in_threadpool(_save_upload, reference_amount)
            saved_paths.append(amount_path)

        result = await run_in_threadpool(
            _run_pipeline, cheque_path, signature_path, amount_path
        )
        return JSONResponse(content=result)
    except HTTPException:
        raise
    except Exception as exc:
        LOGGER.exception("Cheque processing failed")
        raise HTTPException(
            status_code=500,
            detail=f"Processing failed: {exc}",
        ) from exc
    finally:
        for path in saved_paths:
            try:
                path.unlink(missing_ok=True)
            except OSError:
                LOGGER.exception("Could not delete temporary upload %s", path)
        for upload in uploads:
            await upload.close()
