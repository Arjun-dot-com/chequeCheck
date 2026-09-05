# Technical Notes (CV/ML Backend)

Internal reference for whoever works on the CV/ML side of this repo.
Covers the tech stack and why each piece was picked, how a request flows
through the system, known bugs that were fixed, and remaining gaps.

## Tech stack

| Piece | Choice | Why |
|---|---|---|
| Web framework | **FastAPI** + Uvicorn | Async, automatic `multipart/form-data` file upload handling, free interactive docs at `/docs` (handy for manual testing without writing a client), minimal boilerplate compared to Flask/Django for a small single-purpose API. |
| Image ops / preprocessing | **OpenCV (`opencv-python`)** | Industry-standard for classic CV (grayscale, blur, thresholding, affine warps for deskew). Fast, well-documented, no training required for these steps. |
| Region detection | **Ultralytics YOLOv8 (nano)** | Needed to locate 7 different field regions (payee, amount, date, account no., cheque no., bank name, signature) on cheques that vary in layout. A single trainable object detector generalizes across cheque templates far better than hand-tuned fixed-coordinate cropping would. "Nano" size was chosen for fast CPU inference/training given no GPU is currently used (`device="cpu"` in `train_yolo.py`) — swap to a bigger variant (`yolov8s`/`m`) if accuracy needs outgrow nano's capacity once more labeled data exists. |
| Text extraction | **Tesseract OCR (via `pytesseract`)** | Free, offline, no per-call cost or network dependency (unlike Google Vision / Azure AI Vision, which the original brief also suggested) — appropriate for a system that may process PII and shouldn't have to ship every cheque image to a third party. Per-field Tesseract configs (`--psm 7` = "treat as single line", plus character whitelists) are used because each field is a small single-line crop, not a full-page document — this measurably improves accuracy over default settings on small crops. |
| PDF handling | **PyMuPDF (`pymupdf`, import name historically `fitz`)** | Renders the first page of an uploaded PDF to an image before the rest of the pipeline (which is OpenCV-based and can't read PDFs directly) sees it. Chosen over `pdf2image` because it has no external system dependency (`pdf2image` needs the Poppler binary installed separately, another Tesseract-style setup headache); PyMuPDF ships everything in the wheel. |
| Fraud heuristics | Plain OpenCV (adaptive threshold, morphology, Canny edges, Laplacian variance) | Deliberately simple, explainable placeholders (see "Known gaps" below) rather than a trained anomaly-detection model, since there's no labeled fraud dataset yet. |

## Request flow

```
POST /scan (multipart file)
        |
        v
 app.py: save upload to temp_uploads/<uuid>.<ext>
        |
        v
 ChequeProcessingPipeline.process_cheque(path)   [cv_core/pipeline.py]
        |
        |-- 1. ChequePreprocessor.preprocess(path)      [preprocessor.py]
        |        load_image (PDF -> PyMuPDF render, else cv2.imread)
        |        -> grayscale -> Gaussian blur -> Otsu binarize -> deskew
        |        returns (deskewed_binary, original_color_image)
        |
        |-- 2. ROIExtractor.extract_all(original_color_image)  [roi_extractor.py]
        |        runs YOLOv8 inference, crops every detected box (conf > 0.3)
        |        -> {class_name: cropped_ndarray} e.g. {"Amt": <crop>, "Sign": <crop>, ...}
        |
        |-- 3. field-name mapping (YOLO class names -> OCR config keys)
        |        e.g. "DateIss" -> "date", "ReceiverName" -> "payee", "AcNo" -> "micr"
        |
        |-- 4. OCREngine.extract_all_text(mapped_rois)   [ocr_engine.py]
        |        Tesseract per crop with a field-specific --psm/whitelist config
        |        -> {field: {"value": str, "confidence": float 0-100}}
        |
        |-- 5. FraudChecker                                [fraud_checker.py]
        |        isolate_signature(Sign crop)   -> is_signed: bool (ink-ratio heuristic)
        |        detect_tampering(Amt crop)     -> amount_tamper_flag: bool (edge-density heuristic)
        |        detect_tampering(ReceiverName crop) -> payee_tamper_flag: bool
        |
        v
 JSON response: { status, extracted_data: {...}, validation: {...} }
        |
        v
 app.py: delete temp upload, return JSONResponse
```

The YOLO model and the two heavy engines (`ROIExtractor`, `OCREngine`,
`FraudChecker`) are instantiated **once**, at module import time in
`app.py` (`pipeline = ChequeProcessingPipeline()`), not per-request — YOLO
weight loading is the expensive part, so it's paid once at process
startup, not on every scan.

## Dataset & training

- Labeled samples live in `cv_core/data/samples/Images/` — one `.jpg` +
  one YOLO-format `.txt` label file per cheque (`class x_center y_center
  width height`, normalized 0–1). Currently ~112 images.
- Classes (from `dataset.yaml`): `0 IssueBank, 1 ReceiverName, 2 AcNo,
  3 Amt, 4 ChqNo, 5 DateIss, 6 Sign`.
- `train.txt` / `test.txt` list the train/val image paths and are
  currently **absolute Windows paths** tied to this machine
  (`C:\Users\arjun\...`). If the project moves machines or the images get
  relocated, regenerate these two files (one absolute or repo-root-relative
  path per line) rather than hand-editing 112 lines.
- `train_yolo.py` fine-tunes `yolov8n.pt` (auto-downloaded by Ultralytics
  the first time it's needed if not already present at the repo root) and
  writes results under `cv_core/models/cheque_roi_extractor/`.

### Bug fixed: trained weights were landing in the wrong directory

The original `train_yolo.py` passed a **relative** path,
`project="cv_core/models"`, to `model.train()`. The installed Ultralytics
version resolves a relative `project` against its *own* internal
`runs/` directory rather than the current working directory — so training
actually wrote to `runs/detect/cv_core/models/cheque_roi_extractor/`
instead of `cv_core/models/cheque_roi_extractor/`. `ROIExtractor` looks for
weights at the latter path, so it always raised `FileNotFoundError: YOLO
model not found`, even after a training run had "succeeded." This is also
why an incomplete `cheque_roi_extractor2/` directory (just an `args.yaml`
and a plot, no weights) existed — a second training attempt that hit the
same issue.

Fix: `train_yolo.py` now builds `project` as an **absolute path**
(`os.path.join(current_dir, "cv_core", "models")`), which Ultralytics
uses literally regardless of its internal runs-dir settings. Also added
`exist_ok=True` so re-running training overwrites the same run directory
in place instead of accumulating `cheque_roi_extractor2`,
`cheque_roi_extractor3`, etc. Verified by running a real training pass and
confirming `best.pt`/`last.pt` land at
`cv_core/models/cheque_roi_extractor/weights/`.

**Important:** the checkpoint currently sitting in that `weights/` folder
was only trained for **1 epoch** as a smoke test to prove this fix — it
detects nothing useful (mAP ≈ 0). Retrain with a realistic epoch count
(and ideally more labeled data) before treating detections as meaningful.

## Other fixes made while reviewing this code

- `fraud_checker.py`: `isolate_signature` and `detect_tampering` would
  divide by zero / crash on an empty (`0` pixels) ROI crop. Added
  early-return guards. This mattered because `roi_extractor.py` could hand
  back a degenerate zero-size crop when a YOLO box's coordinates rounded
  to `x1==x2` or `y1==y2`.
- `roi_extractor.py`: now skips degenerate boxes (`x2 <= x1 or y2 <= y1`)
  entirely rather than storing an empty crop.
- `pipeline.py`: added `.size > 0` checks before calling fraud-check
  functions on signature/amount/payee crops, and added `chq_no` /
  `bank_name` to the OCR field mapping — these were being detected by
  YOLO but silently discarded before (never OCR'd, never returned), even
  though "cheque number" extraction is explicitly called out in the
  project brief.
- `ocr_engine.py`: collapsed four near-identical `read_*` methods into one
  loop driven by the `configs` dict, and added `chq_no`/`bank_name`
  configs to match the above.
- `app.py`:
  - Used the raw client-supplied filename directly in
    `os.path.join(UPLOAD_DIR, file.filename)` — a filename like
    `../../evil.py` would let a client write outside `temp_uploads/`
    (path traversal). Fixed by generating a random filename
    (`uuid4().hex`) and keeping only the original extension.
  - `file.filename` could be `None` and would crash `.lower()` before
    reaching the try/except. Added a null check.
  - The generic `except Exception` handler was also catching
    `HTTPException`s raised earlier in the same `try` block and
    re-wrapping them into a less useful `500 Processing failed: 500:
    <original detail>`. Added an explicit `except HTTPException: raise`
    before the generic handler.
  - Added `GET /health` and CORS middleware (`allow_origins=["*"]` for
    now) — the frontend runs on a different origin/port and needs both to
    integrate at all; see `FRONTEND_NOTES.md`.
- Added PDF support: `.pdf` was accepted by `app.py`'s extension check but
  `ChequePreprocessor.load_image` used `cv2.imread`, which cannot read
  PDFs and would raise a confusing `FileNotFoundError` for any PDF upload.
  Added a PyMuPDF-based render-first-page-to-image step, verified against
  a generated test PDF.
- Removed `cv_core/models/cheque_roi_extractor2/` (dead directory from
  the failed second training attempt — no weights, just leftover
  `args.yaml`/plot) and stray `__pycache__/` directories.

## Known gaps / things not implemented here

These are explicitly **out of scope for the CV/ML slice** per the
project brief, or deliberately left simple pending more data:
- No validation against real/mock banking records, no duplicate-cheque
  detection, no approve/review/reject decision logic, no dashboard,
  reporting, or audit trail — the brief expects these elsewhere in the
  system (likely a separate backend service sitting between this API and
  the frontend).
- Fraud checks are simple, explainable heuristics (ink coverage, edge
  density, Laplacian variance), not a trained anomaly/forgery-detection
  model. They're a reasonable v1 given no labeled fraud dataset exists;
  revisit once (if) one does.
- `is_signed=false` doesn't distinguish "cheque genuinely unsigned" from
  "signature region wasn't detected by YOLO at all."
- No authentication/authorization on `/scan` — assumed to sit behind
  another service or gateway that handles that.
- No automated test suite yet (see "How to test" below for manual
  verification steps).

## How to test

All commands assume the project virtualenv is active
(`venv\Scripts\activate` on Windows) with `pip install -r
requirements.txt` already run, and the Tesseract binary installed
separately (see README's Setup section) — without Tesseract, OCR fields
will just come back empty (`value: "", confidence: 0.0`) rather than
crashing, so you can still test the detection half of the pipeline
without it installed.

### 1. Sanity-check imports and the trained model load

```bash
python -c "from cv_core.pipeline import ChequeProcessingPipeline; p = ChequeProcessingPipeline(); print('OK')"
```
If this raises `FileNotFoundError: YOLO model not found`, you need to run
`python train_yolo.py` first.

### 2. Run the pipeline directly on a sample image (no server needed)

```bash
python -c "
import json
from cv_core.pipeline import ChequeProcessingPipeline
p = ChequeProcessingPipeline()
result = p.process_cheque('cv_core/data/samples/Images/Cheque083654.jpg')
print(json.dumps(result, indent=2))
"
```
Swap in any file under `cv_core/data/samples/Images/` (or your own cheque
photo). Check that `extracted_data` fields are populated with plausible
values and reasonable `confidence` scores once you've trained for more
than the smoke-test epoch count.

### 3. Retrain the model for real

```bash
python -c "from train_yolo import train_model; train_model(epochs=50)"
```
Watch the per-class `mAP50` in the validation table at the end — with
only ~112 images, don't expect production-grade numbers immediately, but
you should see it climb well above the near-zero smoke-test baseline.
Confirm weights land at
`cv_core/models/cheque_roi_extractor/weights/best.pt` (they will, given
the path fix above, but worth eyeballing once).

### 4. Run the API and hit it manually

```bash
uvicorn app:app --reload
```
Then either:
- Open `http://127.0.0.1:8000/docs` (FastAPI's auto-generated Swagger UI)
  and use the "Try it out" button on `POST /scan` to upload a file from
  the browser — no extra tooling needed.
- Or from another terminal:
  ```bash
  curl -F "file=@cv_core/data/samples/Images/Cheque083654.jpg" http://127.0.0.1:8000/scan
  curl http://127.0.0.1:8000/health
  ```

### 5. Exercise the error paths

- Upload a non-image file (e.g. a `.txt`) → expect `400`.
- Stop the server, delete/rename
  `cv_core/models/cheque_roi_extractor/weights/best.pt`, restart, then
  call `/scan` → expect `503` and `/health` to report
  `"pipeline_ready": false`.
- Upload a corrupted/truncated image file → expect a clean `500` with a
  message, not an unhandled server crash.
