# Technical Notes: CV/ML Backend

This document describes the implemented architecture, model lifecycle, data
flow, and known limitations. It reflects the current code rather than the
original Tesseract/OpenCV-heuristic prototype.

## Architecture

| Layer | Implementation | Responsibility |
|---|---|---|
| API | FastAPI + Uvicorn | Multipart uploads, health reporting, response schema, cleanup |
| Image processing | OpenCV + PyMuPDF | Decode images/PDF page one, grayscale, denoise, binarize, deskew |
| ROI detection | Ultralytics YOLOv8-nano | Locate bank, payee, account, amount, cheque number, date, and signature |
| Handwriting OCR | Microsoft `trocr-base-handwritten` | Generate text from each tightly cropped ROI |
| Similarity model | PyTorch Siamese CNN | Shared 128-D features for signature verification and cursive word spotting |
| PDF rendering | PyMuPDF | Convert the first PDF page into a BGR OpenCV image |

Tesseract, ink-density rules, Canny edge density, and Laplacian-variance fraud
rules are no longer used by the CV pipeline.

## Device selection

TrOCR and the SNN use:

```python
torch.device("cuda" if torch.cuda.is_available() else "cpu")
```

The loaded models are moved to that device and put into evaluation mode. SNN
inference uses `torch.no_grad()` and TrOCR generation uses
`torch.inference_mode()`.

## Model lifecycle

`app.py` constructs one global `ChequeProcessingPipeline`. Its constructor
creates one instance each of:

- `ChequePreprocessor`
- `ROIExtractor` and its YOLO model
- `OCREngine`, `TrOCRProcessor`, and `VisionEncoderDecoderModel`
- `FraudChecker` and its single shared `SNNVerifier`

Route handlers reuse these instances. They never load models per request. A
process-wide lock serializes complete inference calls to constrain CPU and RAM
spikes. With multiple Uvicorn workers, each worker is a separate process and
therefore owns a separate model copy; use worker counts deliberately.

If startup fails, `app.py` keeps the service alive with `pipeline = None`.
`GET /health` then returns `pipeline_ready: false`, and `POST /scan` returns
`503`.

## Request flow

```text
POST /scan
  file=<cheque>
  reference_signature=<optional image>
  reference_amount=<optional image>
        |
        v
app.py
  validate extensions
  save each upload as temp_uploads/<uuid>.<allowed-extension>
        |
        v
ChequeProcessingPipeline.process_cheque(...)
        |
        +-- 1. ChequePreprocessor.preprocess
        |      image: cv2.imread
        |      PDF: render first page at 300 DPI
        |      grayscale -> Gaussian blur -> Otsu -> deskew
        |
        +-- 2. ROIExtractor.extract_all(original BGR image)
        |      YOLO inference -> non-degenerate crops above 0.3 confidence
        |
        +-- 3. Map detector names to API names
        |      IssueBank    -> bank_name
        |      ReceiverName -> payee
        |      AcNo         -> micr_code
        |      Amt          -> amount
        |      ChqNo        -> cheque_number
        |      DateIss      -> date
        |
        +-- 4. OCREngine.extract_all_text
        |      BGR crop -> RGB PIL image -> processor -> model.generate
        |      -> {value: decoded text, confidence: 100.0}
        |
        +-- 5. FraudChecker
               Sign + reference signature -> shared SNN -> is_signed
               Amt + reference amount     -> shared SNN -> tamper flag
        |
        v
{status, extracted_data, validation}
        |
        v
app.py finally block
  delete all temporary files and close UploadFile handles
```

Missing detector regions are represented as empty values with `0.0`
confidence. Missing references skip SNN inference and retain unverified default
flags.

## TrOCR implementation

`OCREngine` loads:

```text
microsoft/trocr-base-handwritten
```

Supported OpenCV inputs are grayscale, BGR, and BGRA arrays. Each crop is
converted to an RGB PIL image and passed to `TrOCRProcessor`. Token IDs from
`model.generate()` are decoded with special tokens removed.

### Dependency compatibility

`requirements.txt` constrains Transformers to `>=4.40,<5.0`. Transformers
5.17 produced a backend-tokenizer construction failure for this checkpoint in
the development environment. A clean install of the pinned requirements loads
the expected `RobertaTokenizerFast`.

### OCR limitations

- TrOCR is a line recognizer, not a document-layout engine. Whole-cheque input
  can collapse to meaningless output; YOLO crops are required.
- `confidence: 100.0` is a contract placeholder, not model certainty. Real
  confidence needs generation scores/logits and calibration.
- The handwritten checkpoint performs poorly on MICR fonts, dense printed
  labels, and multi-line bank logos. A MICR-specific recognizer should handle
  routing, account, and cheque numbers.
- The first use downloads model files from Hugging Face. Later loads use the
  user cache.

## Siamese network

`cv_core/snn_model.py` contains a SigNet-style convolutional feature extractor:

- Four convolution stages with ReLU activations
- Local response normalization in the early stages
- Max pooling
- Fully connected projection to a normalized 128-D embedding

The dual-branch `forward(img1, img2)` calls the same `forward_once` method for
both inputs, guaranteeing shared parameters. `SNNVerifier` converts OpenCV
crops to grayscale `1 x 1 x 105 x 105` tensors and compares embeddings with
Euclidean distance.

The contrastive training loss is:

```text
mean((1 - label) * D^2 + label * max(0, margin - D)^2)
```

`label = 0` means similar/genuine, `label = 1` means dissimilar/forged, and the
default margin is `1.0`.

### Dual use

One `SNNVerifier` instance is shared by both operations:

1. Signature crop versus an enrolled genuine signature.
2. Cursive amount crop versus a visual exemplar of the expected amount.

The default match threshold is `0.5`. This threshold must be calibrated on
held-out data for each task; a convenient default is not evidence of accuracy.

## Checkpoints and training data

### YOLO

Default checkpoint:

```text
cv_core/models/cheque_roi_extractor/weights/best.pt
```

Dataset classes from `dataset.yaml`:

```text
0 IssueBank
1 ReceiverName
2 AcNo
3 Amt
4 ChqNo
5 DateIss
6 Sign
```

The repository's current YOLO result artifacts came from smoke testing and do
not establish usable mAP. Retrain with more labeled images and a realistic
epoch count. `train.txt` and `test.txt` currently contain absolute local paths
and must be regenerated when the repository moves.

### SNN

Default checkpoint:

```text
cv_core/models/snn/best.pt
```

No production SNN checkpoint, training pipeline, signature enrollment store,
or reference-word gallery is included yet. The API cannot become ready until a
compatible state dictionary is provisioned.

Model binaries (`*.pt`, `*.pth`, `*.ckpt`, `*.safetensors`, and model `.bin`
files) are intentionally ignored by Git.

## API stability and security

- Client filenames are never used as destination paths. Only an allowlisted
  suffix is retained, and `uuid4().hex` supplies the filename.
- Cheque uploads accept PNG/JPG/JPEG/PDF. Reference uploads accept PNG/JPG/JPEG.
- All saved paths are removed in a `finally` block on success or failure.
- The `/scan` response contains exactly `status`, `extracted_data`, and
  `validation` at the top level.
- CORS is open for local development and must be restricted in production.
- There is no authentication, request-size limit, persistent reference store,
  rate limit, or server-side request timeout yet.
- Exception text is currently included in `500` responses; production should
  log internal details and return a generic public message.

## Legacy validation package

The `validation/` package and `GET /audit` endpoint remain in the repository,
but `/scan` no longer appends `validation_result` because that violated the
frontend's three-key response contract. The current scan route does not invoke
bank-record decision logic or add new audit entries.

## Testing

### Syntax and unit tests

```powershell
python -m compileall -q app.py cv_core test_trocr.py test_cheque_fields.py
python -m pytest -q
```

### Isolated TrOCR line

```powershell
python test_trocr.py "C:\path\to\cheque.jpeg" --crop X1 Y1 X2 Y2
```

### Demonstration multi-field extraction

```powershell
python test_cheque_fields.py "C:\path\to\X_017.jpeg"
```

The multi-field helper contains normalized crop regions tailored to the sample
layout. It demonstrates why ROI extraction is necessary but does not replace
YOLO.

### API

```powershell
uvicorn app:app --reload
```

Open `http://127.0.0.1:8000/docs`, or run:

```powershell
curl.exe -F "file=@C:\path\cheque.jpg" `
  -F "reference_signature=@C:\path\signature.jpg" `
  -F "reference_amount=@C:\path\amount-reference.jpg" `
  http://127.0.0.1:8000/scan
```

### Error paths

- Unsupported extension -> `400`
- Missing required multipart `file` -> `422`
- Missing/corrupt local model during startup -> `/health` is not ready and
  `/scan` returns `503`
- Corrupt image after upload -> `500`

## Remaining production work

- Train and calibrate YOLO and SNN with representative data.
- Add a dedicated MICR recognizer.
- Add persistent, access-controlled reference enrollment.
- Return calibrated OCR confidence rather than `100.0`.
- Add request-size/auth/rate controls and production CORS settings.
- Add tests for TrOCR preprocessing, SNN checkpoint loading, pipeline mapping,
  multipart cleanup, and the exact JSON contract.
- Establish measured accuracy, latency, bias, and false-match rates before any
  automated financial decision is allowed.
