import os
import shutil
import uuid
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from cv_core.pipeline import ChequeProcessingPipeline
from validation.service import process_validation

app = FastAPI(
    title="ChequeCheck CV API",
    description="API for processing cheque images to extract fields and detect fraud.",
    version="1.0.0"
)

# Allow the frontend (running on a different origin/port during development)
# to call this API directly from the browser. Restrict this to the real
# frontend origin(s) before deploying to production.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize the pipeline globally so the model is only loaded once
try:
    pipeline = ChequeProcessingPipeline()
except Exception as e:
    print(f"Warning: Could not initialize pipeline completely. Ensure YOLO model is trained. Error: {e}")
    pipeline = None

UPLOAD_DIR = "temp_uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

@app.get("/health")
async def health_check():
    """Simple liveness/readiness probe for the frontend to check before scanning."""
    return {"status": "ok", "pipeline_ready": pipeline is not None}

@app.post("/scan")
async def scan_cheque(file: UploadFile = File(...)):
    if pipeline is None:
        raise HTTPException(status_code=503, detail="Pipeline not initialized. Check if the YOLO model is trained.")

    if not file.filename or not file.filename.lower().endswith(('.png', '.jpg', '.jpeg', '.pdf')):
        raise HTTPException(status_code=400, detail="Invalid file type. Only PNG, JPG, JPEG, or PDF are supported.")

    # Use a generated filename (keeping only the original extension) so a
    # malicious or unlucky client-supplied filename can't traverse out of
    # UPLOAD_DIR or collide with another concurrent upload.
    ext = os.path.splitext(file.filename)[1].lower()
    file_path = os.path.join(UPLOAD_DIR, f"{uuid.uuid4().hex}{ext}")
    
    try:
        # Save the uploaded file temporarily
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
            
        # Process the image
        results = pipeline.process_cheque(file_path)
        validation_result = process_validation(results)
        results["validation_result"] = validation_result
        
        # Clean up the temporary file
        if os.path.exists(file_path):
            os.remove(file_path)
            
        if "error" in results:
            raise HTTPException(status_code=500, detail=results["error"])

        return JSONResponse(content=results)

    except HTTPException:
        raise
    except Exception as e:
        # Ensure cleanup on failure
        if os.path.exists(file_path):
            os.remove(file_path)
        raise HTTPException(status_code=500, detail=f"Processing failed: {str(e)}")

# To run the API locally:
# uvicorn app:app --reload
