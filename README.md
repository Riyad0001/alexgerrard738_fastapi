# Physical Key Matching Service

FastAPI service for registering physical keys from segmented images and matching
new images against the registered catalog. The production pipeline uses the
packaged YOLO segmentation model, normalized blade geometry, bitting profiles,
blade contours, and a TorchScript embedding model.

## Production guarantees

- Startup fails if configuration is invalid or either model cannot load.
- Classical threshold segmentation is disabled by default and forbidden in production.
- Registration metadata and both key sides commit in one SQLite transaction.
- Duplicate key IDs return HTTP 409 and never overwrite data.
- Failed registrations remove their uploaded and generated files.
- Match uploads and artifacts are deleted after the response is produced.
- SQLite uses WAL, foreign keys, busy timeouts, and process-local locking.
- Production requires an X-API-Key value of at least 24 characters.
- Upload byte and decoded-pixel limits are enforced.

## Required models

- models/key_detector.pt
- models/key_embedding_model_traced.pt

Override these locations with DETECTOR_MODEL and EMBEDDING_MODEL.

## Local development

    py -3.11 -m venv .venv
    & ".\.venv\Scripts\python.exe" -m pip install -r requirements.txt
    & ".\.venv\Scripts\python.exe" -m uvicorn main:app --host 127.0.0.1 --port 8010

- Swagger: http://127.0.0.1:8010/docs
- Liveness: http://127.0.0.1:8010/health
- Readiness: http://127.0.0.1:8010/ready

## Production configuration

| Variable | Required | Default | Purpose |
|---|---:|---|---|
| APP_ENV | yes | development | Use production in deployments |
| API_KEY | production | unset | Secret accepted through X-API-Key |
| STORAGE_DIR | no | storage | Persistent database/image root |
| DETECTOR_MODEL | no | models/key_detector.pt | YOLO model path |
| EMBEDDING_MODEL | no | models/key_embedding_model_traced.pt | TorchScript model path |
| MATCH_THRESHOLD | no | 0.65 | Confident-match threshold |
| CONSISTENCY_THRESHOLD | no | 0 | Front/back rejection threshold |
| BITTING_SHORTLIST | no | 10 | Geometry shortlist size |
| MAX_UPLOAD_BYTES | no | 20971520 | Per-image upload limit |
| MAX_IMAGE_PIXELS | no | 40000000 | Decoded image pixel limit |
| SQLITE_TIMEOUT_SECONDS | no | 30 | SQLite lock timeout |
| ALLOW_SEGMENTATION_FALLBACK | no | false | Development-only CV fallback |
| ALLOWED_ORIGINS | no | empty | Comma-separated browser origins |
| ALLOWED_HOSTS | production | localhost,127.0.0.1 | Comma-separated HTTP hosts |
| ENABLE_DOCS | no | true | Enable Swagger and ReDoc |

Generate API_KEY with the platform secret generator. Do not store it in Git.

## Docker

    docker build -t key-matcher:2.1.0 .
    docker run --rm -p 8000:8000 \
      -e APP_ENV=production \
      -e API_KEY="$KEY_MATCHER_API_KEY" \
      -e ALLOWED_HOSTS="localhost,127.0.0.1,your-api-host.example" \
      -v key-matcher-storage:/app/storage \
      key-matcher:2.1.0

SQLite deployments must use one application worker and one writable persistent
volume. Never run multiple containers against the same SQLite file. A
multi-instance deployment requires replacing FeatureStore with a network
database implementation.

## API

Public operational endpoints:

- GET /health
- GET /ready

All /v1 endpoints require X-API-Key when API_KEY is configured:

- POST /v1/keys/register
- GET /v1/keys?limit=50&offset=0
- GET /v1/keys/{key_id}
- DELETE /v1/keys/{key_id}
- POST /v1/keys/match?limit=1

Registration is multipart and requires key_type, manufacturer, lock_brand,
key_code, key_bitting, num_pins, brand, code, x, and front_image. back_image
and key_id are optional. Omitting key_id generates a UUID. key_bitting must
contain one numeric value per pin, for example 5-2-4-7-6.

## Data and migrations

The default database is storage/features.sqlite3. Startup creates and migrates
tables without deleting existing feature rows. Back up the complete STORAGE_DIR
while the service is stopped, or use SQLite's online backup API. Never copy only
the main database file while WAL writes are active.

Deleting a key removes its database rows and files owned under STORAGE_DIR.
Match query images are not retained.

## Verification and release gate

    & ".\.venv\Scripts\python.exe" -m pytest -q

Before release, calibrate MATCH_THRESHOLD and CONSISTENCY_THRESHOLD against a
held-out dataset containing different keys, lighting, backgrounds, rotations,
glare, blur, and partial occlusion. Matching scores are identification evidence;
they must not directly unlock physical security hardware without a separate
authorization control.
# alexgerrard738_fastapi
