"""
Standalone YOLOv8 Training Script for Person A (CV & Detection Lead)

Trains YOLOv8x on football dataset classes:
  0: ball
  1: goalkeeper
  2: player
  3: referee

Usage:
  python training/train.py --data /path/to/data.yaml --epochs 100 --batch 16 --device 0
"""

import argparse
import os
import shutil
from ultralytics import YOLO


def parse_args():
    parser = argparse.ArgumentParser(
        description="Fine-tune YOLOv8x on Football Player & Ball Dataset"
    )
    parser.add_argument(
        "--data",
        type=str,
        required=True,
        help="Path to data.yaml file from Roboflow dataset",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="yolov8x.pt",
        help="Base pretrained YOLO model (default: yolov8x.pt)",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=100,
        help="Number of training epochs (default: 100)",
    )
    parser.add_argument(
        "--batch",
        type=int,
        default=16,
        help="Batch size (default: 16)",
    )
    parser.add_argument(
        "--imgsz",
        type=int,
        default=640,
        help="Image size in pixels (default: 640)",
    )
    parser.add_argument(
        "--device",
        default="0",
        help="Device to train on (e.g. 0 for GPU, 'cpu' for CPU)",
    )
    parser.add_argument(
        "--output-weights",
        type=str,
        default="models/best.pt",
        help="Destination path to export fine-tuned weights",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    print("=" * 60)
    print("  Football Detection — YOLO Fine-Tuning")
    print(f"  Base Model: {args.model}")
    print(f"  Dataset:    {args.data}")
    print(f"  Epochs:     {args.epochs}")
    print(f"  Batch:      {args.batch}")
    print(f"  Device:     {args.device}")
    print("=" * 60)

    # Initialize pretrained YOLO
    model = YOLO(args.model)

    # Train model
    results = model.train(
        data=args.data,
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
