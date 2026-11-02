# Football Match Analysis — Computer Vision Pipeline

AI-powered football match analysis using YOLO object detection, ByteTrack tracking, optical flow camera estimation, and K-Means jersey clustering.

## Project Structure

```
SPD Project/
├── main.py                          # Main pipeline entry point
├── requirements.txt                 # Python dependencies
├── generate_dummy_stub.py           # Mock data generator for testing
│
├── models/
│   └── best.pt                      # Fine-tuned YOLO weights (from Colab)
│
├── training/
│   └── football_yolov8_training.ipynb  # Colab notebook for YOLO training
│
├── input_videos/
│   └── input_video.mp4              # Source match footage
│
├── output_videos/
│   └── output_video.avi             # Annotated output
│
├── stubs/
│   ├── tracks_stub.pkl              # Cached detection tracks
│   └── camera_movement_stub.pkl     # Cached camera movement data
│
├── trackers/
│   ├── __init__.py
│   └── tracker.py                   # YOLO + ByteTrack tracker
│
├── camera_movement_estimator/
│   ├── __init__.py
│   └── camera_movement_estimator.py # Lucas-Kanade optical flow
│
├── team_assigner/
│   ├── __init__.py
│   └── team_assigner.py             # K-Means jersey color clustering
│
├── player_ball_assigner/
│   ├── __init__.py
│   └── player_ball_assigner.py      # Ball possession logic
│
└── utils/
    ├── __init__.py
    ├── video_utils.py               # Video I/O (read/save)
    └── bbox_utils.py                # Bounding box math utilities
```

## Person A Deliverables (CV & Camera Lead)

### Phase 1: YOLO Fine-Tuning & Entity Tracking
- **Model Training**: YOLOv8x fine-tuned on Roboflow football dataset (100 epochs)
  - Classes: ball (0), goalkeeper (1), player (2), referee (3)
  - Training notebook: `training/football_yolov8_training.ipynb`
- **Tracker** (`trackers/tracker.py`):
  - Batch YOLO inference (batch_size=20)
  - ByteTrack persistent ID assignment
  - Goalkeeper → Player class override
  - Pickle stub caching

### Phase 2: Camera Motion Estimation
- **Camera Movement Estimator** (`camera_movement_estimator/`):
  - Lucas-Kanade Optical Flow on stadium banner regions
  - 5-pixel noise threshold filtering
  - Position adjustment to remove camera panning from coordinates

## Person B Deliverables (ML Analytics Lead)

### Phase 1: Video Infrastructure & Utilities
- Video I/O utilities
- Bounding box math utilities
- Dummy stub generator for independent testing

### Phase 2: Team Clustering & Ball Possession
- K-Means jersey color clustering (2 teams)
- Ball position interpolation (pandas)
- Ball possession assignment (70px threshold)

## Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Train YOLO Model (Google Colab)
1. Open `training/football_yolov8_training.ipynb` in Google Colab
2. Set runtime to GPU
3. Run all cells
4. Download `best.pt` → `models/best.pt`

### 3. Generate Test Stubs (optional)
```bash
python generate_dummy_stub.py
```

### 4. Run Pipeline
```bash
# Place your input video at input_videos/input_video.mp4
python main.py
```

Output will be saved to `output_videos/output_video.avi`.

## Detection Classes

| ID | Class      | Tracking |
|----|------------|----------|
| 0  | Ball       | Fixed ID=1, interpolated |
| 1  | Goalkeeper | Merged into Player class |
| 2  | Player     | ByteTrack persistent IDs |
| 3  | Referee    | ByteTrack persistent IDs |

