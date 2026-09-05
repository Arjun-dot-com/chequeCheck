import os
from ultralytics import YOLO

def train_model(epochs=2):
    """
    Trains a YOLOv8 nano model on the cheque dataset.
    Args:
        epochs (int): Number of epochs to train. Defaults to 2 for testing.
    """
    print(f"Starting YOLOv8 training for {epochs} epochs...")
    
    # Load a pretrained YOLOv8 nano model
    model = YOLO("yolov8n.pt")
    
    # Get the absolute path to the dataset.yaml and output directory.
    # NOTE: `project` must be absolute. Ultralytics resolves a *relative* project
    # path against its own internal runs/ directory (not the current working
    # directory), which silently wrote past trainings to runs/detect/cv_core/...
    # instead of cv_core/models/..., leaving ROIExtractor unable to find best.pt.
    current_dir = os.path.dirname(os.path.abspath(__file__))
    yaml_path = os.path.join(current_dir, "dataset.yaml")
    project_dir = os.path.join(current_dir, "cv_core", "models")

    # Train the model
    # device='0' ensures it uses the dedicated GPU (RTX 4050)
    results = model.train(
        data=yaml_path,
        epochs=epochs,
        imgsz=640,
        batch=16,
        device="cpu", # Using CPU since CUDA torch is not installed
        project=project_dir,
        name="cheque_roi_extractor",
        exist_ok=True,
    )

    print("Training complete! The best model is saved in: cv_core/models/cheque_roi_extractor/weights/best.pt")

if __name__ == "__main__":
    # We will run a quick 2-epoch test to ensure the pipeline works
    train_model(epochs=2)
