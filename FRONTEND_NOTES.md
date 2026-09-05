# Notes for the Frontend Developer

This document describes the CV/ML backend's API contract so the frontend
can integrate against it without needing to read the Python code. If
anything here looks wrong once you actually integrate, treat this file as
the thing that's out of date — ping the CV/ML side to fix it or the code.

## What this service does (and doesn't do)

It's a single-purpose image-processing API: send it one cheque image, get
back extracted fields + a couple of fraud signals. It does **not**:
- store anything (uploaded files are processed and deleted immediately),
- know about users, sessions, or auth,
- validate against real bank records,
- make an approve/review/reject decision — that business logic (and any
  thresholds on the confidence/fraud scores below) belongs in your layer
  or a service in between.

## Base URL & running it locally

By default: `http://127.0.0.1:8000` (via `uvicorn app:app --reload`).
CORS is currently wide open (`allow_origins=["*"]`) so you can call it
directly from a browser dev server on any port during development. This
**should be locked down to your actual frontend origin(s)** before any
real deployment — flag that to whoever owns deployment config.

## Endpoints

### `GET /health`

Quick liveness/readiness check — call this on app load if you want to
show a "backend unavailable" state instead of letting a scan silently
fail.

Response:
```json
{ "status": "ok", "pipeline_ready": true }
```

`pipeline_ready: false` means the YOLO model failed to load (most likely
it hasn't been trained yet on that environment) — `/scan` will return
`503` until that's fixed on the backend side. There's nothing the
frontend can do about this except show a friendly "processing is
temporarily unavailable" message.

### `POST /scan`

Multipart form upload, field name **`file`**.

- Accepted types: `.png`, `.jpg`, `.jpeg`, `.pdf` (by extension — checked
  client-side too if you want to fail fast before uploading).
- One file per request. No batch endpoint currently.

Example (fetch):
```js
const form = new FormData();
form.append("file", fileInput.files[0]);

const res = await fetch("http://127.0.0.1:8000/scan", {
  method: "POST",
  body: form,
});
const data = await res.json();
```

#### Success response — `200`

```json
{
  "status": "success",
  "extracted_data": {
    "micr_code":      { "value": "123456789", "confidence": 91.2 },
    "date":           { "value": "12/05/2026", "confidence": 88.4 },
    "payee":          { "value": "JOHN DOE", "confidence": 76.0 },
    "amount":         { "value": "1,250.00", "confidence": 94.5 },
    "cheque_number":  { "value": "000452", "confidence": 90.1 },
    "bank_name":      { "value": "", "confidence": 0.0 }
  },
  "validation": {
    "is_signed": true,
    "amount_tamper_flag": false,
    "payee_tamper_flag": false
  }
}
```

Field notes:
- Every field under `extracted_data` always has the shape
  `{ "value": string, "confidence": number }`. `value` is `""` and
  `confidence` is `0.0` when that field wasn't detected on the cheque, or
  OCR found nothing readable in it — **this is not itself an error**,
  just "no data for this field". Render it as "not detected" rather than
  blank/crashing.
- `confidence` is Tesseract's average word-confidence, **0–100** (not
  0–1).
- `is_signed`: boolean heuristic based on ink coverage in the detected
  signature region. `false` can mean either "genuinely unsigned" or
  "signature region wasn't detected at all" — the two aren't currently
  distinguished in the response.
- `amount_tamper_flag` / `payee_tamper_flag`: boolean heuristics (edge
  density in the region is abnormally high). These are **not**
  forensic-grade fraud detection — treat `true` as "worth a human
  glance", not "definitely fraudulent". Do not word any UI around these
  as a hard fraud verdict.
- Field list may grow over time (e.g. routing number split out
  separately) — don't assume `extracted_data` is a fixed/closed set of
  keys; read what you need by key name and ignore unknown ones.

#### Error responses

All errors come back as FastAPI's standard shape:
```json
{ "detail": "human-readable message" }
```

| Status | Meaning | Suggested UI handling |
|---|---|---|
| 400 | Missing file, or extension not in png/jpg/jpeg/pdf | Show inline validation error before/instead of a generic failure |
| 503 | Backend's YOLO model isn't loaded | "Scanning temporarily unavailable" — not user-fixable, don't ask them to retry the same file |
| 500 | Unexpected processing failure (corrupt image, decoding failure, etc.) | Generic "couldn't process this image, try another" — safe to offer retry |

There is currently no request timeout enforced server-side. A single
scan is CPU-bound (YOLO inference + OCR) and typically takes low single
digit seconds on CPU; budget your own client-side timeout/spinner
accordingly (e.g. 15–30s before showing a "still working…" or timeout
state).

## Practical integration preferences

- **Preview before upload.** Since there's no batch endpoint and each
  scan takes a few seconds, let the user preview/crop/rotate the image
  client-side before sending it — resending a bad photo is expensive.
- **Client-side extension/size check first.** The API only checks
  extension, not file size — enforce a sane max upload size (e.g. 10MB)
  on the frontend to avoid slow uploads of huge camera photos.
- **Don't poll `/health` continuously.** Check once on load / before
  first scan; there's no push mechanism for "model just became ready".
- **PDF support** renders only the **first page** to an image before
  processing (via PyMuPDF) — if a user uploads a multi-page PDF, only
  page 1 is scanned. Word the UI/upload prompt accordingly ("upload a
  single-page scan of the cheque"), or trim multi-page PDFs client-side.
- **Don't build UI around exact `confidence` thresholds yet** — no
  calibration/threshold has been agreed for "trust this value
  automatically" vs. "flag for manual review". That decision should be
  made jointly once real accuracy numbers exist from a properly trained
  model, not hardcoded in the frontend independently.

## Tech stack (for context, not something you need to run)

Python, FastAPI + Uvicorn, OpenCV, Ultralytics YOLOv8, Tesseract OCR
(via `pytesseract`). See `TECHNICAL_NOTES.md` if you're curious about the
internals.
