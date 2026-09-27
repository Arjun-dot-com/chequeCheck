# Frontend Integration Notes

This file is the source of truth for the browser-facing API contract. The CV
service processes uploads in memory/local temporary storage and returns field
extraction plus visual comparison flags. It does not authenticate users,
settle payments, or return an approval/rejection decision.

## Base URL

Local development uses:

```text
http://127.0.0.1:8000
```

CORS currently allows every origin for development. Restrict `allow_origins`
to the deployed frontend origins before production.

## `GET /health`

Response:

```json
{
  "status": "ok",
  "pipeline_ready": true
}
```

`pipeline_ready: false` means at least one startup dependency failed. Common
causes are missing YOLO/SNN checkpoints, missing Python dependencies, or a
TrOCR download/cache problem. The frontend should disable scanning and show a
temporary service-unavailable message.

## `POST /scan`

Send `multipart/form-data` with these fields:

| Field | Required | Accepted extensions | Purpose |
|---|---:|---|---|
| `file` | Yes | `.png`, `.jpg`, `.jpeg`, `.pdf` | Cheque image; only page one of a PDF is processed |
| `reference_signature` | No | `.png`, `.jpg`, `.jpeg` | Genuine account-holder signature reference |
| `reference_amount` | No | `.png`, `.jpg`, `.jpeg` | Visual reference of the expected cursive amount |

Only one cheque is accepted per request. Uploaded files are assigned random
server-side names and removed after processing.

### Browser example

```js
const form = new FormData();
form.append("file", chequeInput.files[0]);

if (signatureInput.files[0]) {
  form.append("reference_signature", signatureInput.files[0]);
}

if (amountReferenceInput.files[0]) {
  form.append("reference_amount", amountReferenceInput.files[0]);
}

const response = await fetch("http://127.0.0.1:8000/scan", {
  method: "POST",
  body: form,
});

const payload = await response.json();
if (!response.ok) {
  throw new Error(payload.detail ?? "Cheque processing failed");
}
```

Do not set `Content-Type` manually when sending `FormData`; the browser must add
the multipart boundary.

## Success response (`200`)

The top-level object contains exactly `status`, `extracted_data`, and
`validation`:

```json
{
  "status": "success",
  "extracted_data": {
    "micr_code": {
      "value": "073902766",
      "confidence": 100.0
    },
    "date": {
      "value": "Feb. 25, 2015",
      "confidence": 100.0
    },
    "payee": {
      "value": "Cathy Johnson",
      "confidence": 100.0
    },
    "amount": {
      "value": "One hundred and 00/100",
      "confidence": 100.0
    },
    "cheque_number": {
      "value": "001001",
      "confidence": 100.0
    },
    "bank_name": {
      "value": "FNB",
      "confidence": 100.0
    }
  },
  "validation": {
    "is_signed": true,
    "amount_tamper_flag": false,
    "payee_tamper_flag": false
  }
}
```

### Extracted-field behavior

- Every field always has `{ "value": string, "confidence": number }`.
- A missing YOLO region is returned as `{ "value": "", "confidence": 0.0 }`.
- A crop processed by the current TrOCR implementation returns `100.0` because
  calibrated token/word confidence is not implemented yet. Do not display this
  as a guarantee of correctness or build automated decisions around it.
- TrOCR is strongest on a tight, single handwritten line. MICR digits and bank
  logos may be inaccurate until field-specific recognizers are added.

### Validation behavior

- `is_signed` means the extracted signature matched the uploaded reference
  according to the SNN threshold. It defaults to `false` if either image is
  absent or unusable.
- `amount_tamper_flag` is `true` when the extracted amount and reference amount
  embeddings are farther apart than the configured threshold. It defaults to
  `false` when comparison is unavailable.
- `payee_tamper_flag` is currently always `false`; the API has no payee-reference
  upload yet.
- These are visual similarity signals, not legal or financial fraud verdicts.

## Error responses

Errors use FastAPI's standard shape:

```json
{
  "detail": "Human-readable error message"
}
```

| Status | Meaning | Suggested UI behavior |
|---:|---|---|
| `400` | Unsupported filename extension or missing required upload | Show an inline upload error |
| `422` | Multipart field is missing or malformed | Check the form field names |
| `500` | Corrupt image or unexpected inference failure | Show a retry/change-image message |
| `503` | One or more models failed during application startup | Disable scanning and show service unavailable |

## UX recommendations

- Allow preview, rotation, and cropping before upload.
- Apply a client-side size limit; the backend currently validates extensions,
  not upload byte size.
- Use a generous timeout. CPU inference runs YOLO, several TrOCR generations,
  and optional SNN comparisons, and is serialized server-side to limit peak
  memory use.
- Do not poll `/health` continuously. Check on application load and before a
  retry after a `503`.
- Treat a blank field as "not detected," not as an API failure.
- Do not describe a visual mismatch as confirmed fraud.

## Current backend stack

FastAPI, OpenCV, Ultralytics YOLOv8, PyTorch, Hugging Face Transformers with
Microsoft TrOCR, and a shared Siamese neural network.
