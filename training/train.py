"""
Standalone YOLOv8 Training Script for Person A (CV & Detection Lead)

Trains YOLOv8x on football dataset classes:
  0: ball
  1: goalkeeper
  2: player
  3: referee

Configuration is loaded automatically from .env if present.

Usage:
  # With explicit dataset path:
  python training/train.py --data /path/to/data.yaml --epochs 100 --batch 16 --device 0

  # Or auto-download from Roboflow using .env credentials:
  python training/train.py --download --epochs 100
"""

import argparse
import os
import shutil
from dotenv import load_dotenv
from ultralytics import YOLO

# Load configuration from .env file
load_dotenv()


def parse_args():
    parser = argparse.ArgumentParser(
        description="Fine-tune YOLOv8x on Football Player & Ball Dataset"
    )
    parser.add_argument(
        "--data",
        type=str,
        default=os.getenv("DATA_YAML", None),
        help="Path to data.yaml file from Roboflow dataset",
    )
    parser.add_argument(
        "--download",
        action="store_true",
        help="Download dataset from Roboflow using credentials in .env",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=os.getenv("MODEL_NAME", "yolov8x.pt"),
        help="Base pretrained YOLO model (default: yolov8x.pt or from .env)",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=int(os.getenv("EPOCHS", "100")),
        help="Number of training epochs (default: 100 or from .env)",
    )
    parser.add_argument(
        "--batch",
        type=int,
        default=int(os.getenv("BATCH_SIZE", "16")),
        help="Batch size (default: 16 or from .env)",
    )
    parser.add_argument(
        "--imgsz",
        type=int,
        default=int(os.getenv("IMAGE_SIZE", "640")),
        help="Image size in pixels (default: 640 or from .env)",
    )
    parser.add_argument(
        "--device",
        default=os.getenv("DEVICE", "0"),
        help="Device to train on (e.g. 0 for GPU, 'cpu' for CPU)",
    )
    parser.add_argument(
        "--output-weights",
        type=str,
        default="models/best.pt",
        help="Destination path to export fine-tuned weights",
    )
    return parser.parse_args()


def download_roboflow_dataset():
    """Download dataset from Roboflow using environment credentials."""
    from roboflow import Roboflow

    api_key = os.getenv("ROBOFLOW_API_KEY")
    workspace = os.getenv("ROBOFLOW_WORKSPACE", "shreys-workspace-rkusb")
    project_name = os.getenv("ROBOFLOW_PROJECT", "football-players-detection-3zvbc")
    version_num = int(os.getenv("ROBOFLOW_VERSION", "1"))

    if not api_key:
        raise ValueError(
            "ROBOFLOW_API_KEY not found in environment or .env file. "
            "Please add it to .env or pass --data /path/to/data.yaml."
        )

    print(f"Connecting to Roboflow workspace: {workspace}, project: {project_name} (v{version_num})...")
    rf = Roboflow(api_key=api_key)
    project = rf.workspace(workspace).project(project_name)
    dataset = project.version(version_num).download("yolov8")
    data_yaml = os.path.join(dataset.location, "data.yaml")
    print(f"Dataset downloaded. data.yaml: {data_yaml}")
    return data_yaml


def main():
    args = parse_args()

    data_path = args.data
    if args.download or (data_path is None and not os.path.exists("data.yaml")):
        data_path = download_roboflow_dataset()
    elif data_path is None:
        raise ValueError("Must provide --data /path/to/data.yaml or specify --download")

    print("=" * 60)
    print("  Football Detection — YOLO Fine-Tuning")
    print(f"  Base Model: {args.model}")
    print(f"  Dataset:    {data_path}")
    print(f"  Epochs:     {args.epochs}")
    print(f"  Batch:      {args.batch}")
    print(f"  Device:     {args.device}")
    print("=" * 60)

    # Initialize pretrained YOLO
    model = YOLO(args.model)

    # Train model
    results = model.train(
        data=data_path,
        epochs=args.epochs,
        batch=args.batch,
        imgsz=args.imgsz,
        device=args.device,
        project="runs_football",
        name="yolov8x_football",
        patience=20,
        save=True,
        save_period=10,
        plots=True,
        verbose=True,
    )

    # Validate
    print("\nValidating model performance...")
    metrics = model.val()
    print(f"mAP50:    {metrics.box.map50:.4f}")
    print(f"mAP50-95: {metrics.box.map:.4f}")

    # Export best weights to models/best.pt
    best_weights_path = os.path.join(
        "runs_football", "yolov8x_football", "weights", "best.pt"
    )

    if os.path.exists(best_weights_path):
        os.makedirs(os.path.dirname(args.output_weights), exist_ok=True)
        shutil.copy2(best_weights_path, args.output_weights)
        print(f"\n[SUCCESS] Best weights copied to: {args.output_weights}")
    else:
        print(f"\n[WARNING] Could not find {best_weights_path}")


if __name__ == "__main__":
    main()
