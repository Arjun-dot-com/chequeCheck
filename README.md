# ChequeCheck — CV/ML Backend

An AI-powered cheque scanning, field-extraction, and basic fraud-flagging
service. This repo currently contains the **computer vision / OCR backend
only** — a FastAPI service that takes a cheque image and returns extracted
fields (payee, amount, date, MICR/account number, cheque number, bank name)
plus a couple of heuristic fraud signals (signature presence, tampering
indicators). The web frontend is built and maintained separately; see
[`FRONTEND_NOTES.md`](FRONTEND_NOTES.md) for the API contract it should
integrate against.

For a deeper explanation of the tech stack, why each piece was chosen, and
how data flows through the pipeline, see
[`TECHNICAL_NOTES.md`](TECHNICAL_NOTES.md).

## Original brief

> **Category:** Fraud Detection / OCR & AI
>
> Design an intelligent cheque processing system that scans cheque images,
> extracts key details via OCR (cheque number, account number,
> routing/transit number, payee, amount, date, signature area), validates
> authenticity against banking records, and applies fraud detection rules
> to decide whether to approve, flag for manual review, or reject the
> cheque.
>
> **Expected outcome:** OCR extraction accuracy ≥ 95%, fraud detection
> accuracy ≥ 90%, processing time < 30s/cheque, manual review reduced by
> ≥ 50%, full audit trail.
>
> **In scope:** image upload, OCR extraction, validation against
> mock/actual banking records, fraud logic, approval workflow, dashboard &
> reporting.
> **Out of scope:** real-time payment settlement, core banking system
> replacement, customer-facing mobile app.

This repo implements the CV/ML slice of that brief (image → structured
data → fraud signals). The validation-against-banking-records, approval
workflow, dashboard, and audit trail are expected to live in the backend
service / frontend that consumes this API.

## Project status

- Preprocessing (grayscale, denoise, Otsu binarization, deskew) — working.
- YOLOv8-based region-of-interest detection for 7 cheque fields — working,
  **but the model needs real training**. The checkpoint currently in
  `cv_core/models/cheque_roi_extractor/weights/` was only trained for a
  single smoke-test epoch to prove the training → inference path works
  end-to-end; it will not detect fields reliably. Re-run
  `train_yolo.py` with a realistic epoch count before relying on it.
- OCR field reading via Tesseract — working, **requires the Tesseract
  binary to be installed separately** (it's not a pip package — see
  Setup below).
- Fraud checks (signature-ink presence, basic tampering heuristics via
  edge density / Laplacian variance) — working, intentionally simple
  placeholders, not the ≥90%-accuracy fraud model described in the brief.
- Validation against banking records, approval/review/reject workflow,
  dashboard, reporting, audit trail — **not implemented here**; out of
  scope for the CV/ML slice.

## Repository layout

```
app.py                      FastAPI app exposing POST /scan and GET /health
train_yolo.py                Trains the YOLOv8 ROI-detector model
dataset.yaml                 YOLO dataset config (classes + paths)
requirements.txt             Python dependencies
yolov8n.pt                   Pretrained YOLOv8-nano base checkpoint (auto-downloads if missing)

cv_core/
  pipeline.py                 Orchestrates preprocess -> detect -> OCR -> fraud checks
  preprocessor.py              Grayscale / denoise / binarize / deskew
  roi_extractor.py             YOLOv8 wrapper -> cropped regions per field
  ocr_engine.py                Tesseract wrapper -> {value, confidence} per field
  fraud_checker.py             Signature isolation + tampering heuristics
  models/cheque_roi_extractor/ Trained YOLO weights + training run artifacts
  data/samples/                Labeled sample cheque images (YOLO format) used for training
```

## Setup

1. Create and activate a virtual environment, then install dependencies:
   ```bash
   python -m venv venv
   venv\Scripts\activate        # Windows
   pip install -r requirements.txt
   ```
2. Install the Tesseract OCR binary (separate from the `pytesseract` pip
   package):
   - Windows: https://github.com/UB-Mannheim/tesseract/wiki, then ensure
     the install directory is on your `PATH` (or set
     `pytesseract.pytesseract.tesseract_cmd` explicitly).
   - macOS: `brew install tesseract`
   - Linux: `apt-get install tesseract-ocr`
3. Train the ROI-detection model (see below) — required before the API
   can serve real predictions.
4. Run the API:
   ```bash
   uvicorn app:app --reload
   ```

## Training the ROI detector

```bash
python train_yolo.py
```

This fine-tunes a YOLOv8-nano model on the labeled samples in
`cv_core/data/samples/` to detect 7 regions: `IssueBank`, `ReceiverName`,
`AcNo`, `Amt`, `ChqNo`, `DateIss`, `Sign`. By default it runs only 2
epochs (a quick smoke test) — increase `epochs` in `train_model()` (or
pass it explicitly) for usable accuracy; with only ~112 labeled images,
expect to also need more labeled data over time.

The trained weights are written to
`cv_core/models/cheque_roi_extractor/weights/best.pt`, which is exactly
where `ROIExtractor` (in `cv_core/roi_extractor.py`) looks for them by
default. Re-running training overwrites this run in place (`exist_ok=True`)
rather than creating `cheque_roi_extractor2`, `cheque_roi_extractor3`, etc.

## Testing it yourself

See the **"How to test"** section at the bottom of `TECHNICAL_NOTES.md`
for step-by-step instructions (unit-level pipeline test, running the API,
and hitting it with a sample cheque image via curl/Swagger UI).
