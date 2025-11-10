from ultralytics import YOLO
import os

# ============================================================
# YOLOv11 Model Size Selection
# ============================================================
# Uncomment ONE of the following lines to select model size:
# - nano (n): Fastest, smallest, lowest accuracy
# - small (s): Good balance for embedded devices
# - medium (m): Balanced speed and accuracy
# - large (l): Higher accuracy, slower
# - extra large (x): Best accuracy, slowest

# MODEL_SIZE = "yolo11n.pt"      # Nano - fastest
MODEL_SIZE = "yolo11s.pt"    # Small
# MODEL_SIZE = "yolo11m.pt"    # Medium
# MODEL_SIZE = "yolo11l.pt"    # Large
# MODEL_SIZE = "yolo11x.pt"    # Extra Large

# ============================================================
# Training Configuration
# ============================================================
DATA_YAML = "datasets/king-queen-pawn-1/data.yaml"  # Path to dataset config
EPOCHS = 100                               # Number of training epochs
IMGSZ = 640                                # Image size for training
BATCH = 16                                 # Batch size (adjust based on GPU memory)
PROJECT = "runs/detect_med"                    # Project folder for results
NAME = "yolo11_digits"                     # Experiment name

# ============================================================
# Additional Training Parameters (Optional)
# ============================================================
# Device selection: auto-detect GPU or use CPU
import torch
DEVICE = 0 if torch.cuda.is_available() else 'cpu'  # Auto-detect GPU, fallback to CPU
PATIENCE = 50                 # Early stopping patience (epochs without improvement)
SAVE_PERIOD = -1              # Save checkpoint every x epochs (-1 to disable)
WORKERS = 8                   # Number of dataloader workers
OPTIMIZER = 'auto'            # Optimizer: 'SGD', 'Adam', 'AdamW', 'NAdam', 'RAdam', 'auto'
LR0 = 0.01                    # Initial learning rate
MOMENTUM = 0.937              # SGD momentum/Adam beta1
WEIGHT_DECAY = 0.0005         # Optimizer weight decay
RESUME = False                # Resume training from last checkpoint

def main():
    """
    Train YOLOv11 model on digital numbers dataset from Roboflow
    """
    
    # Verify data.yaml exists
    if not os.path.exists(DATA_YAML):
        raise FileNotFoundError(
            f"Dataset configuration not found at: {DATA_YAML}\n"
            f"Please run download_model.py first to download the dataset."
        )
    
    print("=" * 60)
    print(f"Starting YOLOv11 Training")
    print(f"Model: {MODEL_SIZE}")
    print(f"Dataset: {DATA_YAML}")
    print(f"Device: {DEVICE}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"CUDA Version: {torch.version.cuda}")
    print(f"Epochs: {EPOCHS}")
    print(f"Image Size: {IMGSZ}")
    print(f"Batch Size: {BATCH}")
    print("=" * 60)
    
    # Load YOLO model
    model = YOLO(MODEL_SIZE)
    
    # Train the model
    results = model.train(
        data=DATA_YAML,
        epochs=EPOCHS,
        imgsz=IMGSZ,
        batch=BATCH,
        device=DEVICE,
        project=PROJECT,
        name=NAME,
        patience=PATIENCE,
        save_period=SAVE_PERIOD,
        workers=WORKERS,
        optimizer=OPTIMIZER,
        lr0=LR0,
        momentum=MOMENTUM,
        weight_decay=WEIGHT_DECAY,
        resume=RESUME,
        verbose=True,
        plots=True,              # Generate training plots
        save=True,               # Save checkpoints
        save_json=True,          # Save results as JSON
        val=True,                # Validate during training
    )
    
    print("\n" + "=" * 60)
    print("Training completed!")
    print(f"Results saved to: {results.save_dir}")
    print("=" * 60)
    
    # Evaluate on validation set
    print("\nRunning validation...")
    metrics = model.val()
    
    print(f"\nValidation Results:")
    print(f"  mAP50: {metrics.box.map50:.4f}")
    print(f"  mAP50-95: {metrics.box.map:.4f}")
    print(f"  Precision: {metrics.box.mp:.4f}")
    print(f"  Recall: {metrics.box.mr:.4f}")
    
    # Export the model (optional)
    print("\nExporting model...")
    model.export(format='onnx')  # Export to ONNX format
    print("Model exported successfully!")
    
    return results, metrics

if __name__ == "__main__":
    main()
