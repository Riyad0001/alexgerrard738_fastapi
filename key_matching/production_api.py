from __future__ import annotations

import logging
import re
import secrets
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from time import perf_counter

import cv2
from fastapi import (
    Depends,
    FastAPI,
    File,
    Form,
    HTTPException,
    Query,
    Request,
    Security,
    UploadFile,
)
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from fastapi.security import APIKeyHeader
from fastapi.staticfiles import StaticFiles

from .config import settings
from .database import DuplicateKeyError, FeatureStore
from .io import SUPPORTED_EXTENSIONS
from .matcher import MatchWeights, rank_candidates, score_pair
from .pipeline import KeyPipeline, PipelineError
from .schemas import (
    Candidate,
    DeleteResponse,
    KeyInfo,
    KeyRecord,
    MatchCandidateDjango,
    MatchResponse,
    MatchResponseDjango,
    ReferencesResponse,
    RegisterResponse,
    TrainedKeyGroup,
    TrainedKeyItem,
)

LOGGER = logging.getLogger("key_matching.api")
WEIGHTS = MatchWeights()
API_KEY_HEADER = APIKeyHeader(name="X-API-Key", auto_error=False)


def require_api_key(provided: str | None = Security(API_KEY_HEADER)) -> None:
    if settings.api_key and (
        provided is None or not secrets.compare_digest(provided, settings.api_key)
    ):
        raise HTTPException(401, detail={"error": "invalid_api_key"})


PROTECTED = [Depends(require_api_key)]


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings.validate()
    for directory in (settings.storage_dir, settings.images_dir, settings.artifacts_dir):
        directory.mkdir(parents=True, exist_ok=True)
    pipeline = KeyPipeline(
        str(settings.detector_model),
        str(settings.embedding_model),
        allow_segmentation_fallback=settings.allow_segmentation_fallback,
        max_image_pixels=settings.max_image_pixels,
    )
    pipeline.load_models()
    app.state.pipeline = pipeline
    app.state.store = FeatureStore(
        settings.database_url, timeout=settings.sqlite_timeout_seconds
    )
    app.state.ready = True
    LOGGER.info("service_ready database=%s", settings.database_url)
    try:
        yield
    finally:
        app.state.ready = False
        app.state.store.close()


app = FastAPI(
    title="Physical Key Matching Service",
    version="2.1.0",
    lifespan=lifespan,
    docs_url="/docs" if settings.enable_docs else None,
    redoc_url="/redoc" if settings.enable_docs else None,
)

if settings.allowed_origins:
    app.add_middleware(
        CORSMiddleware,
        allow_origins=list(settings.allowed_origins),
        allow_credentials=False,
        allow_methods=["GET", "POST", "PATCH", "PUT", "DELETE"],
        allow_headers=["Content-Type", "X-API-Key", "X-Request-ID"],
    )
if settings.app_env == "production":
    app.add_middleware(TrustedHostMiddleware, allowed_hosts=list(settings.allowed_hosts))

# Serve images and artifacts stored on disk so URLs returned to clients work.
app.mount("/storage", StaticFiles(directory=str(settings.storage_dir)), name="storage")


@app.middleware("http")
async def request_context(request: Request, call_next):
    request_id = request.headers.get("X-Request-ID") or str(uuid.uuid4())
    started = perf_counter()
    response = await call_next(request)
    response.headers["X-Request-ID"] = request_id
    LOGGER.info(
        "request_complete method=%s path=%s status=%s duration_ms=%.1f request_id=%s",
        request.method,
        request.url.path,
        response.status_code,
        (perf_counter() - started) * 1000,
        request_id,
    )
    return response


def _safe_unlink(paths: list[Path | str]) -> None:
    storage_root = settings.storage_dir.resolve()
    for value in paths:
        path = Path(value).resolve()
        if path.is_relative_to(storage_root):
            path.unlink(missing_ok=True)
        else:
            LOGGER.error("refused_file_cleanup path=%s", path)


def _save_upload(upload: UploadFile) -> Path:
    suffix = Path(upload.filename or "").suffix.lower()
    if suffix not in SUPPORTED_EXTENSIONS:
        raise HTTPException(415, detail={"error": "unsupported_image_type"})
    if upload.content_type and not (
        upload.content_type.startswith("image/")
        or upload.content_type == "application/octet-stream"
    ):
        raise HTTPException(415, detail={"error": "invalid_content_type"})
    path = settings.images_dir / f"{uuid.uuid4()}{suffix}"
    size = 0
    with path.open("xb") as output:
        while chunk := upload.file.read(1024 * 1024):
            size += len(chunk)
            if size > settings.max_upload_bytes:
                output.close()
                path.unlink(missing_ok=True)
                raise HTTPException(413, detail={"error": "image_too_large"})
            output.write(chunk)
    if size == 0:
        path.unlink(missing_ok=True)
        raise HTTPException(422, detail={"error": "empty_image"})
    return path


def _process(request: Request, upload: UploadFile):
    path = _save_upload(upload)
    normalized_path = settings.artifacts_dir / f"{path.stem}_normalized.png"
    mask_path = settings.artifacts_dir / f"{path.stem}_mask.png"
    try:
        features, normalized = request.app.state.pipeline.process(path)
        if not cv2.imwrite(str(normalized_path), normalized.image):
            raise OSError("Could not save normalized image")
        if not cv2.imwrite(str(mask_path), normalized.mask):
            raise OSError("Could not save normalized mask")
        return features, path, normalized_path, mask_path
    except (PipelineError, OSError) as exc:
        _safe_unlink([path, normalized_path, mask_path])
        LOGGER.info("rejected_key_image reason=%s", exc)
        raise HTTPException(
            422,
            detail={"error": "invalid_key_image", "message": str(exc)},
        ) from exc


def _normalize_bitting(value: str, num_pins: int) -> str:
    parts = [part for part in re.split(r"[-,\s]+", value.strip()) if part]
    if len(parts) != num_pins or not all(part.isdigit() for part in parts):
        raise HTTPException(
            422,
            detail={
                "error": "invalid_key_bitting",
                "message": "Key bitting must contain one numeric value per pin",
            },
        )
    return "-".join(parts)


def _time_ago(created_at: str) -> str:
    """Return a human-readable 'X days ago' string from an ISO timestamp string."""
    from datetime import datetime, timezone
    try:
        dt = datetime.fromisoformat(created_at.replace(" ", "T"))
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        now = datetime.now(timezone.utc)
        diff = now - dt
        seconds = int(diff.total_seconds())
        if seconds < 60:
            return f"{seconds} seconds ago"
        minutes = seconds // 60
        if minutes < 60:
            return f"{minutes} minute{'s' if minutes != 1 else ''} ago"
        hours = minutes // 60
        if hours < 24:
            return f"{hours} hour{'s' if hours != 1 else ''} ago"
        days = hours // 24
        return f"{days} day{'s' if days != 1 else ''} ago"
    except Exception:
        return ""


def _record_to_key_info(record: dict, request: Request | None = None) -> KeyInfo:
    """Convert a FeatureStore registration record to a Django-compatible KeyInfo."""
    meta = record.get("metadata", {})
    
    # Build absolute image URLs if possible
    def _abs_url(path_str: str | None) -> str | None:
        if not path_str:
            return None
        if request is not None:
            try:
                rel = Path(path_str).relative_to(settings.storage_dir)
                return str(request.base_url).rstrip("/") + "/storage/" + str(rel).replace("\\", "/")
            except ValueError:
                pass
        return path_str

    return KeyInfo(
        id=record["key_id"],
        key_type=meta.get("key_type"),
        manufacturer=meta.get("manufacturer"),
        lock_brand=meta.get("lock_brand"),
        key_code=meta.get("key_code"),
        key_bitting=meta.get("key_bitting"),
        no_pins=meta.get("num_pins"),
        brand=meta.get("brand"),
        code=meta.get("code"),
        x=meta.get("x"),
        cross_refs=meta.get("cross_refs", []),
        front_image=_abs_url(meta.get("front_image")),
        back_image=_abs_url(meta.get("back_image")),
        created_at=record.get("created_at"),
        updated_at=record.get("updated_at"),
    )


def _parse_references(value) -> list:
    """Safely parse references from a JSON string, list, or empty value."""
    import json as _json
    if not value:
        return []
    if isinstance(value, list):
        return value
    if isinstance(value, str):
        value = value.strip()
        if not value:
            return []
        try:
            parsed = _json.loads(value)
            if isinstance(parsed, list):
                return parsed
            return []
        except Exception:
            return []
    return []


@app.get("/")
def root():
    return {
        "service": "physical-key-matcher",
        "version": app.version,
        "docs": app.docs_url,
    }


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/ready")
def ready(request: Request):
    if not getattr(request.app.state, "ready", False) or not request.app.state.store.ping():
        raise HTTPException(503, detail={"error": "service_not_ready"})
    return {
        "status": "ready",
        "pipeline_version": "geometry-bitting-v2",
        "catalog_sides": len(request.app.state.store.all()),
    }


@app.post(
    "/v1/keys/register",
    response_model=RegisterResponse,
    status_code=201,
    dependencies=PROTECTED,
)
def register_key(
    request: Request,
    key_id: str = Form(default="", max_length=128),
    key_type: str = Form(..., min_length=1, max_length=100),
    manufacturer: str = Form(..., min_length=1, max_length=100),
    lock_brand: str = Form(..., min_length=1, max_length=100),
    key_code: str = Form(..., min_length=1, max_length=100),
    key_bitting: str = Form(..., min_length=1, max_length=200),
    num_pins: int = Form(..., ge=1, le=20),
    brand: str = Form(..., min_length=1, max_length=100),
    code: str = Form(..., min_length=1, max_length=100),
    x: str = Form(..., min_length=1, max_length=100),
    front_image: UploadFile = File(...),
    back_image: UploadFile | None = File(default=None),
):
    key_id = key_id.strip() or str(uuid.uuid4())
    registration = {
        "key_type": key_type.strip(),
        "manufacturer": manufacturer.strip(),
        "lock_brand": lock_brand.strip(),
        "key_code": key_code.strip(),
        "key_bitting": _normalize_bitting(key_bitting, num_pins),
        "num_pins": num_pins,
        "brand": brand.strip(),
        "code": code.strip(),
        "x": x.strip(),
    }
    blank_fields = [
        name for name, value in registration.items()
        if isinstance(value, str) and not value
    ]
    if blank_fields:
        raise HTTPException(
            422,
            detail={"error": "required_registration_fields_blank", "fields": blank_fields},
        )
    store = request.app.state.store
    if store.exists(key_id):
        raise HTTPException(409, detail={"error": "key_id_already_exists"})

    processed = []
    consistency = None
    try:
        front = _process(request, front_image)
        processed.append(("front", *front))
        if back_image is not None and back_image.filename:
            back = _process(request, back_image)
            processed.append(("back", *back))
            consistency, _ = score_pair(front[0], back[0], WEIGHTS)
            if consistency < settings.consistency_threshold:
                raise HTTPException(
                    409,
                    detail={"error": "inconsistent_sides", "score": consistency},
                )

        sides = []
        for side, features, image_path, normalized_path, mask_path in processed:
            features.metadata.update(registration)
            sides.append(
                (
                    side,
                    features,
                    {
                        "front_image": str(image_path) if side == "front" else None,
                        "back_image": str(image_path) if side == "back" else None,
                        "normalized_image": str(normalized_path),
                        "mask_image": str(mask_path),
                    },
                )
            )
        store.register(key_id, registration, sides)
    except DuplicateKeyError as exc:
        _safe_unlink([path for item in processed for path in item[2:]])
        raise HTTPException(409, detail={"error": "key_id_already_exists"}) from exc
    except Exception:
        _safe_unlink([path for item in processed for path in item[2:]])
        raise

    return RegisterResponse(
        key_id=key_id,
        status="registered",
        sides=[item[0] for item in processed],
        registration=registration,
        consistency_score=consistency,
    )


@app.get("/v1/keys", response_model=list[KeyRecord], dependencies=PROTECTED)
def list_keys(
    request: Request,
    limit: int = Query(50, ge=1, le=10000),
    offset: int = Query(0, ge=0),
):
    return request.app.state.store.list_registrations(limit, offset)


@app.get("/v1/keys/{key_id}", response_model=KeyRecord, dependencies=PROTECTED)
def get_key(request: Request, key_id: str):
    record = request.app.state.store.get_registration(key_id)
    if record is None:
        raise HTTPException(404, detail={"error": "key_not_found"})
    return record


@app.patch("/v1/keys/{key_id}", response_model=KeyRecord, dependencies=PROTECTED)
def update_key(
    request: Request,
    key_id: str,
    key_type: str | None = Form(default=None),
    manufacturer: str | None = Form(default=None),
    lock_brand: str | None = Form(default=None),
    key_code: str | None = Form(default=None),
    key_bitting: str | None = Form(default=None),
    num_pins: int | None = Form(default=None),
    brand: str | None = Form(default=None),
    code: str | None = Form(default=None),
    x: str | None = Form(default=None),
    front_image: UploadFile | None = File(default=None),
    back_image: UploadFile | None = File(default=None),
):
    update_data = {
        k: v for k, v in {
            "key_type": key_type,
            "manufacturer": manufacturer,
            "lock_brand": lock_brand,
            "key_code": key_code,
            "key_bitting": key_bitting,
            "num_pins": num_pins,
            "brand": brand,
            "code": code,
            "x": x,
        }.items() if v is not None
    }
    
    if not update_data and not (front_image and front_image.filename) and not (back_image and back_image.filename):
        raise HTTPException(422, detail={"error": "no_updates_provided"})

    store = request.app.state.store
    current_record = store.get_registration(key_id)
    if not current_record:
        raise HTTPException(404, detail={"error": "key_not_found"})

    if "key_bitting" in update_data:
        pins = update_data.get("num_pins", current_record["metadata"].get("num_pins", 5))
        update_data["key_bitting"] = _normalize_bitting(update_data["key_bitting"], pins)

    processed = []
    try:
        for side, image in [("front", front_image), ("back", back_image)]:
            if image is not None and image.filename:
                res = _process(request, image)
                processed.append((side, *res))
                
        old_paths = []
        for side, features, image_path, normalized_path, mask_path in processed:
            paths = {
                "front_image": str(image_path) if side == "front" else None,
                "back_image": str(image_path) if side == "back" else None,
                "normalized_image": str(normalized_path),
                "mask_image": str(mask_path),
            }
            old_paths.extend(store.update_side(key_id, side, features, paths))
        
        _safe_unlink(old_paths)
    except Exception:
        _safe_unlink([path for item in processed for path in item[2:]])
        raise

    if update_data:
        updated_record = store.update_registration(key_id, update_data)
    else:
        updated_record = store.get_registration(key_id)
        
    if updated_record is None:
        raise HTTPException(404, detail={"error": "key_not_found"})
    return updated_record


@app.delete(
    "/v1/keys/{key_id}",
    response_model=DeleteResponse,
    dependencies=PROTECTED,
)
def delete_key(request: Request, key_id: str):
    paths = request.app.state.store.delete(key_id)
    if not paths:
        raise HTTPException(404, detail={"error": "key_not_found"})
    _safe_unlink(paths)
    return DeleteResponse(key_id=key_id, status="deleted")


@app.post("/v1/keys/match", response_model=MatchResponse, dependencies=PROTECTED)
def match_key(
    request: Request,
    image: UploadFile = File(...),
    limit: int = Query(1, ge=1, le=20),
):
    query, query_path, normalized_path, mask_path = _process(request, image)
    query_id = query_path.stem
    try:
        rows = request.app.state.store.all()
        references = [
            (f"{key_id}:{side}", features) for key_id, side, features in rows
        ]
        ranked = rank_candidates(
            query,
            references,
            limit=len(references) or 1,
            shortlist=settings.bitting_shortlist,
            weights=WEIGHTS,
        )
        matches, seen = [], set()
        for row in ranked:
            key_id, side = row["reference_id"].rsplit(":", 1)
            if key_id in seen:
                continue
            seen.add(key_id)
            detail = row["breakdown"]
            record = request.app.state.store.get_registration(key_id)
            matches.append(
                Candidate(
                    key_id=key_id,
                    side=side,
                    score=row["score"],
                    bitting_score=detail["bitting"],
                    blade_score=detail["blade_contour"],
                    embedding_score=detail["embedding"],
                    registration=record["metadata"] if record else {},
                )
            )
            if len(matches) == limit:
                break
        return MatchResponse(
            query_id=query_id,
            matches=matches,
            confident_match=bool(
                matches and matches[0].score >= settings.match_threshold
            ),
            threshold=settings.match_threshold,
        )
    finally:
        _safe_unlink([query_path, normalized_path, mask_path])


# ═══════════════════════════════════════════════════════════════════════════════
#  Django-compatible microservice endpoints
#  These mirror the exact request/response shapes of the Django DRF views so
#  the Django app can treat this service as a drop-in backend.
# ═══════════════════════════════════════════════════════════════════════════════


def _build_key_info_with_paths(record: dict, request: Request) -> KeyInfo:
    """Build KeyInfo, pulling front/back image paths from key_features table."""
    store: FeatureStore = request.app.state.store
    key_id = record["key_id"]
    meta = record.get("metadata", {})

    front_path, back_path = None, None
    with store._lock:
        row = store._execute(
            "SELECT front_image, back_image FROM key_features WHERE key_id=? AND side='front'",
            (key_id,)
        ).fetchone()
        if row:
            front_path = row["front_image"]
        row = store._execute(
            "SELECT front_image, back_image FROM key_features WHERE key_id=? AND side='back'",
            (key_id,)
        ).fetchone()
        if row:
            back_path = row["back_image"]

    def _abs(p: str | None) -> str | None:
        if not p:
            return None
        path = Path(p)
        base = str(request.base_url).rstrip("/")

        # Case 1: absolute path — make it relative to storage_dir
        if path.is_absolute():
            try:
                rel = path.relative_to(settings.storage_dir.resolve())
                return base + "/storage/" + str(rel).replace("\\", "/")
            except ValueError:
                pass

        # Case 2: relative path starting with "storage/" or "storage\"
        norm = str(path).replace("\\", "/")
        storage_prefix = str(settings.storage_dir).replace("\\", "/").rstrip("/") + "/"
        storage_name = Path(settings.storage_dir).name  # e.g. "storage"

        if norm.startswith(storage_name + "/"):
            # e.g. "storage/images/abc.jpg" → serve from /storage/images/abc.jpg
            rel = norm[len(storage_name) + 1:]
            return base + "/storage/" + rel

        # Case 3: bare relative path with no "storage/" prefix
        return base + "/storage/" + norm

    return KeyInfo(
        id=key_id,
        key_type=meta.get("key_type"),
        manufacturer=meta.get("manufacturer"),
        lock_brand=meta.get("lock_brand"),
        key_code=meta.get("key_code"),
        key_bitting=meta.get("key_bitting"),
        no_pins=meta.get("num_pins"),
        brand=meta.get("brand"),
        code=meta.get("code"),
        x=meta.get("x"),
        cross_refs=meta.get("cross_refs", []),
        front_image=_abs(front_path),
        back_image=_abs(back_path),
        created_at=record.get("created_at"),
        updated_at=record.get("updated_at"),
    )


# ── Admin: Add Key ─────────────────────────────────────────────────────────────

@app.post("/v1/admin/keys/add", status_code=201, dependencies=PROTECTED)
def admin_add_key(
    request: Request,
    key_id: str = Form(default=""),
    key_type: str = Form(default=""),
    manufacturer: str = Form(default=""),
    lock_brand: str = Form(default=""),
    key_code: str = Form(default=""),
    key_bitting: str = Form(default=""),
    num_pins: int = Form(default=0),
    brand: str = Form(default=""),
    code: str = Form(default=""),
    x: str = Form(default=""),
    references: str = Form(default=""),  # JSON string: [{reference_name, reference_value}]
    front_image: UploadFile = File(...),
    back_image: UploadFile | None = File(default=None),
):
    """Add a key to the catalog. Django-compatible response: {success, message, data}."""
    key_id = key_id.strip() or str(uuid.uuid4())

    # Normalize bitting only when num_pins is provided and valid
    if key_bitting and num_pins and num_pins > 0:
        try:
            key_bitting = _normalize_bitting(key_bitting, num_pins)
        except HTTPException:
            pass  # Accept bitting as-is if it doesn't match pin count

    registration = {
        "key_type":    key_type.strip(),
        "manufacturer": manufacturer.strip(),
        "lock_brand":  lock_brand.strip(),
        "key_code":    key_code.strip(),
        "key_bitting": key_bitting.strip(),
        "num_pins":    num_pins,
        "brand":       brand.strip(),
        "code":        code.strip(),
        "x":           x.strip(),
        "cross_refs":  _parse_references(references),
    }

    store = request.app.state.store
    if store.exists(key_id):
        raise HTTPException(409, detail={"error": "key_id_already_exists"})

    processed = []
    consistency = None
    try:
        front = _process(request, front_image)
        processed.append(("front", *front))
        if back_image is not None and back_image.filename:
            back = _process(request, back_image)
            processed.append(("back", *back))
            consistency, _ = score_pair(front[0], back[0], WEIGHTS)
            if consistency < settings.consistency_threshold:
                raise HTTPException(
                    409, detail={"error": "inconsistent_sides", "score": consistency}
                )
        sides = []
        for side, features, image_path, normalized_path, mask_path in processed:
            features.metadata.update(registration)
            if side == "front":
                registration["front_image"] = str(image_path)
            else:
                registration["back_image"] = str(image_path)
            sides.append((
                side, features,
                {
                    "front_image": str(image_path) if side == "front" else None,
                    "back_image": str(image_path) if side == "back" else None,
                    "normalized_image": str(normalized_path),
                    "mask_image": str(mask_path),
                },
            ))
        store.register(key_id, registration, sides)
    except DuplicateKeyError as exc:
        _safe_unlink([path for item in processed for path in item[2:]])
        raise HTTPException(409, detail={"error": "key_id_already_exists"}) from exc
    except Exception:
        _safe_unlink([path for item in processed for path in item[2:]])
        raise

    record = store.get_registration(key_id)
    key_info = _build_key_info_with_paths(record, request) if record else None
    return {
        "success": True,
        "message": "Key registered successfully",
        "data": key_info,
    }


# ── Admin: Update Key (GET + PATCH) ───────────────────────────────────────────

@app.get("/v1/admin/keys/{key_id}/update", dependencies=PROTECTED)
def admin_get_key(request: Request, key_id: str):
    """Get key detail — mirrors Django UpdateKeyView.get()."""
    record = request.app.state.store.get_registration(key_id)
    if record is None:
        raise HTTPException(404, detail={"error": "key_not_found"})
    return {"success": True, "data": _build_key_info_with_paths(record, request)}


@app.patch("/v1/admin/keys/{key_id}/update", dependencies=PROTECTED)
def admin_update_key(
    request: Request,
    key_id: str,
    key_type: str | None = Form(default=None),
    manufacturer: str | None = Form(default=None),
    lock_brand: str | None = Form(default=None),
    key_code: str | None = Form(default=None),
    key_bitting: str | None = Form(default=None),
    num_pins: int | None = Form(default=None),
    brand: str | None = Form(default=None),
    code: str | None = Form(default=None),
    x: str | None = Form(default=None),
    references: str | None = Form(default=None),  # JSON string
    front_image: UploadFile | None = File(default=None),
    back_image: UploadFile | None = File(default=None),
):
    """Update key metadata and/or images — mirrors Django UpdateKeyView.patch()."""
    update_data = {
        k: v for k, v in {
            "key_type": key_type, "manufacturer": manufacturer,
            "lock_brand": lock_brand, "key_code": key_code,
            "key_bitting": key_bitting, "num_pins": num_pins,
            "brand": brand, "code": code, "x": x,
        }.items() if v is not None
    }

    # Parse and include references if provided
    if references is not None:
        update_data["cross_refs"] = _parse_references(references)

    has_images = (front_image and front_image.filename) or (back_image and back_image.filename)

    store = request.app.state.store
    current_record = store.get_registration(key_id)
    if not current_record:
        raise HTTPException(404, detail={"error": "key_not_found"})

    if "key_bitting" in update_data:
        try:
            pins = update_data.get("num_pins", current_record["metadata"].get("num_pins", 5))
            update_data["key_bitting"] = _normalize_bitting(update_data["key_bitting"], pins)
        except HTTPException:
            pass  # Accept bitting as-is

    processed = []
    try:
        for side, image in [("front", front_image), ("back", back_image)]:
            if image is not None and image.filename:
                res = _process(request, image)
                processed.append((side, *res))
                if side == "front":
                    update_data["front_image"] = str(res[1])
                else:
                    update_data["back_image"] = str(res[1])

        old_paths = []
        for side, features, image_path, normalized_path, mask_path in processed:
            paths = {
                "front_image": str(image_path) if side == "front" else None,
                "back_image": str(image_path) if side == "back" else None,
                "normalized_image": str(normalized_path),
                "mask_image": str(mask_path),
            }
            old_paths.extend(store.update_side(key_id, side, features, paths))
        _safe_unlink(old_paths)
    except Exception:
        _safe_unlink([path for item in processed for path in item[2:]])
        raise

    if update_data:
        updated_record = store.update_registration(key_id, update_data)
    else:
        updated_record = store.get_registration(key_id)

    if updated_record is None:
        raise HTTPException(404, detail={"error": "key_not_found"})
    return {"success": True, "data": _build_key_info_with_paths(updated_record, request)}


# ── Admin: Delete Key ──────────────────────────────────────────────────────────

@app.delete("/v1/admin/keys/{key_id}", dependencies=PROTECTED)
def admin_delete_key(request: Request, key_id: str):
    """Delete a key — mirrors Django DeleteKeyView. Returns {message: 'deleted'}."""
    paths = request.app.state.store.delete(key_id)
    if not paths:
        raise HTTPException(404, detail={"error": "key_not_found"})
    _safe_unlink(paths)
    return {"message": "deleted"}


# ── Admin: Cross-References ────────────────────────────────────────────────────

@app.get(
    "/v1/admin/keys/{key_id}/references",
    response_model=ReferencesResponse,
    dependencies=PROTECTED,
)
def admin_get_references(request: Request, key_id: str):
    """Get cross-references for a key — mirrors Django KeyReferencesUpsertView.get()."""
    record = request.app.state.store.get_registration(key_id)
    if record is None:
        raise HTTPException(404, detail={"error": "key_not_found"})
    return ReferencesResponse(
        key_id=key_id,
        references=record["metadata"].get("cross_refs", []),
    )


def _upsert_references(request: Request, key_id: str, references: list):
    record = request.app.state.store.update_cross_refs(key_id, references)
    if record is None:
        raise HTTPException(404, detail={"error": "key_not_found"})
    return {
        "message": "References saved successfully",
        "key_id": key_id,
        "cross_refs": record["metadata"].get("cross_refs", []),
    }


@app.post("/v1/admin/keys/{key_id}/references", dependencies=PROTECTED)
def admin_post_references(request: Request, key_id: str, references: list = Form(...)):
    return _upsert_references(request, key_id, references)


@app.put("/v1/admin/keys/{key_id}/references", dependencies=PROTECTED)
def admin_put_references(request: Request, key_id: str, references: list = Form(...)):
    return _upsert_references(request, key_id, references)


@app.patch("/v1/admin/keys/{key_id}/references", dependencies=PROTECTED)
def admin_patch_references(request: Request, key_id: str, references: list = Form(...)):
    return _upsert_references(request, key_id, references)


# ── Match Key (Django-compatible) ──────────────────────────────────────────────

@app.post("/v1/keys/match-key", response_model=MatchResponseDjango, dependencies=PROTECTED)
def match_key_django(
    request: Request,
    query_image: UploadFile = File(...),
    top_k: int = Form(default=3),
):
    """Match a query image — mirrors Django MatchKeyView.
    
    Fields: query_image (file), top_k (int, 1-20).
    Returns: {message, best_match, all_matches} with confidence_percentage.
    """
    top_k = max(1, min(top_k, 20))

    query, query_path, normalized_path, mask_path = _process(request, query_image)
    try:
        rows = request.app.state.store.all()
        if not rows:
            return MatchResponseDjango(
                message="No trained keys available",
                best_match=None,
                all_matches=[],
            )

        references = [(f"{kid}:{side}", feat) for kid, side, feat in rows]
        ranked = rank_candidates(
            query,
            references,
            limit=len(references) or 1,
            shortlist=settings.bitting_shortlist,
            weights=WEIGHTS,
        )

        best_per_key: dict[str, dict] = {}
        for row in ranked:
            key_id, side = row["reference_id"].rsplit(":", 1)
            score = row["score"]
            existing = best_per_key.get(key_id)
            if existing is None or score > existing["score"]:
                best_per_key[key_id] = {"key_id": key_id, "side": side, "score": score}

        candidates: list[MatchCandidateDjango] = []
        for item in sorted(best_per_key.values(), key=lambda x: x["score"], reverse=True):
            record = request.app.state.store.get_registration(item["key_id"])
            if record is None:
                continue
            score = item["score"]
            confidence = round(max(0.0, min(100.0, ((score - 0.55) / 0.45) * 100.0)), 2)
            candidates.append(MatchCandidateDjango(
                key_info=_build_key_info_with_paths(record, request),
                matched_side=item["side"],
                similarity_score=score,
                confidence_percentage=confidence,
            ))
            if len(candidates) == top_k:
                break

        if not candidates:
            return MatchResponseDjango(
                message="No matching keys found",
                best_match=None,
                all_matches=[],
            )

        best = candidates[0]
        is_reliable = best.similarity_score >= 0.72
        return MatchResponseDjango(
            message="Match found!" if is_reliable else "Possible match found",
            best_match=best,
            all_matches=candidates,
        )
    finally:
        _safe_unlink([query_path, normalized_path, mask_path])


# ── Keys List (flat) ───────────────────────────────────────────────────────────

@app.get("/v1/keys/list", dependencies=PROTECTED)
def list_keys_django(
    request: Request,
    limit: int = Query(50, ge=1, le=200),
    offset: int = Query(0, ge=0),
):
    """List all keys — mirrors Django KeyListView. Returns flat list of KeyInfo."""
    records = request.app.state.store.list_registrations(limit, offset)
    return [_build_key_info_with_paths(r, request) for r in records]


# ── Trained Keys (grouped + paginated) ────────────────────────────────────────

@app.get("/v1/keys/trained", dependencies=PROTECTED)
def trained_keys_list(
    request: Request,
    search: str | None = Query(default=None),
    manufacturer: str | None = Query(default=None),
    page: int = Query(default=1, ge=1),
    page_size: int = Query(default=20, ge=1, le=100),
):
    """Grouped + paginated key list — mirrors Django TrainedKeyListAPIView."""
    from datetime import datetime, timezone, timedelta

    records = request.app.state.store.list_registrations(10000, 0)

    # Filter
    filtered = []
    for r in records:
        meta = r["metadata"]
        if search:
            search_lower = search.lower()
            if not any(
                search_lower in str(meta.get(f, "")).lower()
                for f in ("key_code", "key_bitting", "manufacturer")
            ):
                continue
        if manufacturer and manufacturer.lower() != "all":
            if str(meta.get("manufacturer", "")).lower() != manufacturer.lower():
                continue
        filtered.append(r)

    # Paginate
    total = len(filtered)
    start = (page - 1) * page_size
    page_records = filtered[start: start + page_size]

    # Group by date
    from collections import OrderedDict
    today = datetime.now(timezone.utc).date()
    yesterday = today - timedelta(days=1)
    grouped: dict[str, list] = OrderedDict()

    for r in page_records:
        try:
            dt = datetime.fromisoformat(r["created_at"].replace(" ", "T"))
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            key_date = dt.date()
        except Exception:
            key_date = today

        if key_date == today:
            label = "Today"
        elif key_date == yesterday:
            label = "Yesterday"
        else:
            label = key_date.strftime("%b %d, %Y")

        meta = r["metadata"]
        grouped.setdefault(label, []).append(TrainedKeyItem(
            id=r["key_id"],
            title=f"{meta.get('key_bitting', '')} Cylinder Key Blank",
            manufacturer=meta.get("manufacturer"),
            key_code=meta.get("key_code"),
            time_ago=_time_ago(r["created_at"]),
        ))

    response_data = [TrainedKeyGroup(date=d, items=items) for d, items in grouped.items()]

    base = str(request.base_url).rstrip("/") + str(request.url.path)
    next_url = f"{base}?page={page+1}&page_size={page_size}" if start + page_size < total else None
    prev_url = f"{base}?page={page-1}&page_size={page_size}" if page > 1 else None

    return {
        "count": total,
        "next": next_url,
        "previous": prev_url,
        "success": True,
        "data": response_data,
    }


# ── Trained Key Detail ─────────────────────────────────────────────────────────

@app.get("/v1/keys/trained/{key_id}", dependencies=PROTECTED)
def trained_key_detail(request: Request, key_id: str):
    """Full key detail — mirrors Django TrainedKeyDetailAPIView."""
    record = request.app.state.store.get_registration(key_id)
    if record is None:
        raise HTTPException(404, detail={"success": False, "message": "Key not found"})
    key_info = _build_key_info_with_paths(record, request)
    return {"success": True, "data": key_info}


# ── Latest Scan History (trained keys ordered by date) ────────────────────────

@app.get("/v1/keys/history/latest", dependencies=PROTECTED)
def latest_scan_history(
    request: Request,
    page: int = Query(default=1, ge=1),
    page_size: int = Query(default=10, ge=1, le=100),
):
    """Latest trained keys ordered by date — mirrors Django LatestScanHistoryAPIView."""
    all_records = request.app.state.store.list_registrations(10000, 0)
    total = len(all_records)
    start = (page - 1) * page_size
    page_records = all_records[start: start + page_size]

    results = []
    for r in page_records:
        meta = r["metadata"]
        key_info = _build_key_info_with_paths(r, request)
        try:
            from datetime import datetime, timezone
            dt = datetime.fromisoformat(r["created_at"].replace(" ", "T"))
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            time_str = dt.strftime("%b %d, %I:%M %p")
        except Exception:
            time_str = r["created_at"]

        results.append({
            "id": r["key_id"],
            "manufacturer": meta.get("manufacturer"),
            "bitting": meta.get("key_bitting"),
            "key_code": meta.get("key_code"),
            "image": key_info.front_image,
            "time": time_str,
        })

    base = str(request.base_url).rstrip("/") + str(request.url.path)
    next_url = f"{base}?page={page+1}&page_size={page_size}" if start + page_size < total else None
    prev_url = f"{base}?page={page-1}&page_size={page_size}" if page > 1 else None

    return {
        "count": total,
        "next": next_url,
        "previous": prev_url,
        "success": True,
        "total_trained_keys": total,
        "data": results,
    }
