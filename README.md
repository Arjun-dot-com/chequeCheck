# ChequeCheck CV/ML Backend

ChequeCheck is a FastAPI service for extracting fields from cheque images and
comparing handwritten regions with reference samples. Its processing path is:

```text
OpenCV preprocessing -> YOLOv8 ROI detection -> Microsoft TrOCR -> shared SNN
```

The API accepts a cheque plus optional reference signature and handwritten
amount images. It returns the stable response contract documented in
[`FRONTEND_NOTES.md`](FRONTEND_NOTES.md). Implementation details and model
limitations are in [`TECHNICAL_NOTES.md`](TECHNICAL_NOTES.md).

## Current capabilities

- Loads PNG, JPG, JPEG, and single-page PDF cheque inputs.
- Detects bank, payee, account/MICR, amount, cheque number, date, and signature
  regions with YOLOv8.
- Recognizes individual handwritten text crops with
  `microsoft/trocr-base-handwritten`.
- Uses one 128-dimensional Siamese feature extractor for both signature
  verification and cursive amount matching.
- Selects CUDA automatically when available and otherwise runs on CPU.
- Loads YOLO, TrOCR, and SNN once during application startup, not per request.
- Saves uploads under random UUID filenames and deletes them after processing.

## Important limitations

- The included YOLO training artifacts came from a smoke-test training run and
  are not production quality. Train with more labeled data before evaluating
  detection accuracy.
- TrOCR is a single-line handwriting recognizer. It must receive tight YOLO
  crops; passing an entire cheque produces unreliable text.
- The current OCR response reports `confidence: 100.0` for recognized crops as
  a placeholder. This is not a calibrated confidence score.
- Handwritten TrOCR is not a MICR reader. Routing/account/cheque digits need a
  dedicated MICR model for production-grade extraction.
- No trained SNN checkpoint or SNN training dataset is included. Until weights
  are placed at `cv_core/models/snn/best.pt`, `/health` reports the pipeline as
  unavailable.
- The API performs visual comparisons only. It does not make payment decisions
  or validate against a core banking system.

## Repository layout

```text
app.py                         FastAPI application and multipart upload handling
train_yolo.py                  YOLOv8 ROI-detector training entry point
test_trocr.py                  Single handwritten-line TrOCR smoke test
test_cheque_fields.py          Multi-field demo for the supplied cheque layout
requirements.txt               Python dependencies

cv_core/
  pipeline.py                  End-to-end orchestration
  preprocessor.py              Image/PDF loading, denoising, binarization, deskew
  roi_extractor.py             YOLOv8 inference and field crops
  ocr_engine.py                Microsoft TrOCR wrapper
  snn_model.py                 Siamese CNN, contrastive loss, inference wrapper
  fraud_checker.py             Signature and cursive-amount SNN checks
  data/samples/                YOLO images and labels
  models/cheque_roi_extractor/ Local YOLO training output

validation/                    Existing business-rule/audit utilities
frontend/                      Frontend assets
tests/                         Validation tests
```

## Setup

Python 3.10 or newer is required.

### Windows PowerShell

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

The project pins `transformers` to the 4.x API because the TrOCR checkpoint's
tokenizer is not compatible with the Transformers 5.x loading behavior observed
in this project. No Tesseract installation is required.

The first TrOCR run downloads `microsoft/trocr-base-handwritten` from Hugging
Face and caches it in the current user's Hugging Face cache. The unauthenticated
request warning is harmless for this public model; an `HF_TOKEN` is only needed
for higher Hub rate limits.

## Required model assets

### YOLO detector

`ROIExtractor` expects:

```text
cv_core/models/cheque_roi_extractor/weights/best.pt
```

Train it with:

```powershell
python train_yolo.py
```

For a non-smoke-test run:

```powershell
python -c "from train_yolo import train_model; train_model(epochs=50)"
```

The current `train.txt` and `test.txt` contain machine-specific absolute paths.
Regenerate them after cloning or moving the repository.

### Siamese network

`FraudChecker` expects trained weights at:

```text
cv_core/models/snn/best.pt
```

The checkpoint must match `SiameseNetwork` in `cv_core/snn_model.py`. Model
weights are ignored by Git, so provision this file separately in each runtime.

## Test TrOCR by itself

Use a tightly cropped handwritten line:

```powershell
python test_trocr.py "C:\path\to\image.jpeg" --crop X1 Y1 X2 Y2
```

Example for the provided `X_017.jpeg` layout:

```powershell
python test_trocr.py "C:\path\to\X_017.jpeg" --crop 380 240 1210 405
python test_cheque_fields.py "C:\path\to\X_017.jpeg"
```

`test_cheque_fields.py` uses normalized demonstration regions for that cheque
layout. It is not a replacement for YOLO and will not generalize to arbitrary
layouts.

## Run the API

After both local checkpoints are available:

```powershell
uvicorn app:app --reload
```

Open `http://127.0.0.1:8000/docs`, or use PowerShell's `curl.exe`:

```powershell
curl.exe -F "file=@C:\path\cheque.jpg" `
  -F "reference_signature=@C:\path\signature.jpg" `
  -F "reference_amount=@C:\path\amount-reference.jpg" `
  http://127.0.0.1:8000/scan
```

The two reference files are optional. Without them, signature and amount checks
remain unverified and return their default `false` flags.

## Response contract

```json
{
  "status": "success",
  "extracted_data": {
    "micr_code": {"value": "", "confidence": 0.0},
    "date": {"value": "", "confidence": 0.0},
    "payee": {"value": "", "confidence": 0.0},
    "amount": {"value": "", "confidence": 0.0},
    "cheque_number": {"value": "", "confidence": 0.0},
    "bank_name": {"value": "", "confidence": 0.0}
  },
  "validation": {
    "is_signed": false,
    "amount_tamper_flag": false,
    "payee_tamper_flag": false
  }
}
```

## Tests

```powershell
python -m pytest -q
python -m compileall -q app.py cv_core test_trocr.py test_cheque_fields.py
```

Before publishing, review `git status` carefully. Local checkpoints, Hugging
Face caches, virtual environments, temporary uploads, and runtime audit logs
must not be committed.
