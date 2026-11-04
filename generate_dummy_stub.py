"""
Dummy Stub & Synthetic Video Generator
=======================================

Generates:
  - input_videos/dummy_match.avi   — 1920x1080, 24 fps green pitch with
                                     realistic shirt/grass contrast.
  - stubs/dummy_tracks_stub.pkl    — tracks dict matching CONTRACT.md exactly.
  - stubs/dummy_ground_truth.pkl   — ground-truth metadata for downstream
                                     module testing.

Usage:
    python generate_dummy_stub.py [--frames N] [--seed S]
                                  [--tracks-out PATH] [--video-out PATH]
                                  [--gt-out PATH]

The ball is ABSENT in frames 0-1 (tests bfill) and frames 18-25 (tests
linear interpolation).  Possession changes are built into the ball path so
the PlayerBallAssigner should detect at least two handovers in a 50-frame run.

# TODO(A-PUSH): switch dummy_tracks_stub.pkl → stubs/tracks_stub.pkl once
#   Person A's real YOLO+ByteTrack stub is available and confirmed to match
#   CONTRACT.md (absent keys, ball key=1, plain list bboxes).
"""

import argparse
import math
import os
import pickle
import random
import sys

import cv2
import numpy as np


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

FRAME_W, FRAME_H = 1920, 1080
FPS = 24
PLAYER_W, PLAYER_H = 48, 108          # player bbox size (pixels)
BALL_RADIUS = 12                       # ball circle radius

# Team shirt colours in BGR — slight per-player noise added later
TEAM1_SHIRT_BGR  = (40,  40, 200)     # red-ish
TEAM1_SHORTS_BGR = (20,  20,  80)
TEAM2_SHIRT_BGR  = (200, 60,  40)     # blue-ish
TEAM2_SHORTS_BGR = (80,  20,  10)

GK1_SHIRT_BGR    = (20, 200,  20)     # green goalkeeper
GK2_SHIRT_BGR    = (20, 180, 200)     # yellow goalkeeper

REF_SHIRT_BGR    = (0,  220, 220)     # yellow-ish referee

# Which player IDs (1-indexed within team) are goalkeepers
GK_INDEX_TEAM1 = 1    # player_id == 1 for team 1
GK_INDEX_TEAM2 = 12   # player_id == 12 for team 2 (11 + 1)

# Ball absent frames
BALL_ABSENT_FRAMES = set(range(0, 2)) | set(range(18, 26))

# Possession-change design: ball path visits team-1 cluster then team-2
POSSESSION_HANDOVER_FRAME = 10   # ~frame 10 ball migrates toward team 2


# ---------------------------------------------------------------------------
# Pitch rendering helpers
# ---------------------------------------------------------------------------

def _make_pitch_frame(rng: random.Random) -> np.ndarray:
    """
    Create a 1920×1080 BGR frame that looks like a grass pitch:
    base green with sinusoidal stripe variation (lighter/darker bands)
    to give realistic shirt-vs-grass separation during K-Means.
    """
    base_g = 120
    frame = np.zeros((FRAME_H, FRAME_W, 3), dtype=np.uint8)

    for row in range(FRAME_H):
        # Sinusoidal stripe — period ≈ 90 px, amplitude ±18
        stripe = int(18 * math.sin(2 * math.pi * row / 90))
        g = max(60, min(180, base_g + stripe))
        frame[row, :] = (20, g, 20)   # BGR

    # Subtle per-pixel noise so the pitch isn't perfectly flat
    noise = np.random.RandomState(42).randint(-6, 7, (FRAME_H, FRAME_W, 3), dtype=np.int16)
    frame = np.clip(frame.astype(np.int16) + noise, 0, 255).astype(np.uint8)

    return frame


def _draw_player_rect(frame: np.ndarray, bbox: list, shirt_bgr: tuple,
                      shorts_bgr: tuple) -> None:
    """Draw a 2-colour rectangle: top half = shirt, bottom half = shorts."""
    x1, y1, x2, y2 = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])
    x1 = max(0, x1); y1 = max(0, y1)
    x2 = min(FRAME_W, x2); y2 = min(FRAME_H, y2)
    if x2 <= x1 or y2 <= y1:
        return
    mid_y = (y1 + y2) // 2
    cv2.rectangle(frame, (x1, y1), (x2, mid_y), shirt_bgr, -1)
    cv2.rectangle(frame, (x1, mid_y), (x2, y2), shorts_bgr, -1)


def _noisy_color(base_bgr: tuple, rng: random.Random, amplitude: int = 20) -> tuple:
    """Add reproducible per-player colour noise to a base BGR colour."""
    return tuple(
        max(0, min(255, c + rng.randint(-amplitude, amplitude)))
        for c in base_bgr
    )


# ---------------------------------------------------------------------------
# Player position helpers — smooth drift using sinusoids per track
# ---------------------------------------------------------------------------

def _make_player_positions(num_frames: int, player_id: int, is_gk: bool,
                            team: int, rng: random.Random):
    """
    Generate (cx, cy) centre positions for one player across all frames.
    Uses a fixed base position per player_id with low-frequency sinusoidal
    drift so track IDs remain stable and motion looks realistic.
    """
    rng_local = random.Random(player_id * 7919)   # independent per player

    if is_gk:
        # Goalkeepers sit near their goal ends
        if team == 1:
            base_x = rng_local.randint(80, 200)
        else:
            base_x = rng_local.randint(FRAME_W - 200, FRAME_W - 80)
        base_y = rng_local.randint(FRAME_H // 2 - 150, FRAME_H // 2 + 150)
        amp_x, amp_y = 20, 40
    else:
        half_x = FRAME_W // 2
        if team == 1:
            base_x = rng_local.randint(150, half_x - 50)
        else:
            base_x = rng_local.randint(half_x + 50, FRAME_W - 150)
        base_y = rng_local.randint(100, FRAME_H - 100)
        amp_x, amp_y = 60, 50

    freq_x = rng_local.uniform(0.04, 0.12)
    freq_y = rng_local.uniform(0.03, 0.10)
    phase_x = rng_local.uniform(0, 2 * math.pi)
    phase_y = rng_local.uniform(0, 2 * math.pi)

    positions = []
    for f in range(num_frames):
        cx = int(base_x + amp_x * math.sin(2 * math.pi * freq_x * f + phase_x))
        cy = int(base_y + amp_y * math.sin(2 * math.pi * freq_y * f + phase_y))
        cx = max(PLAYER_W // 2, min(FRAME_W - PLAYER_W // 2, cx))
        cy = max(PLAYER_H // 2, min(FRAME_H - PLAYER_H // 2, cy))
        positions.append((cx, cy))

    return positions


def _cx_cy_to_bbox(cx: int, cy: int) -> list:
    """Convert player centre to [x1,y1,x2,y2] plain float list."""
    hw = PLAYER_W / 2
    hh = PLAYER_H / 2
    return [float(cx - hw), float(cy - hh), float(cx + hw), float(cy + hh)]


# ---------------------------------------------------------------------------
# Ball path helpers
# ---------------------------------------------------------------------------

def _make_ball_path(num_frames: int, player_positions_by_id: dict,
                    team_per_id: dict, rng: random.Random):
    """
    Generate smooth ball (cx, cy) positions for every frame,
    including the ones where the ball is marked as absent (ground truth).

    The path visits a team-1 player cluster for frames 0-17, then
    migrates to a team-2 cluster for frames 26+ so possession changes.
    """
    # Identify target players for each half
    t1_ids = [pid for pid, t in team_per_id.items() if t == 1 and not (pid == GK_INDEX_TEAM1)]
    t2_ids = [pid for pid, t in team_per_id.items() if t == 2 and not (pid == GK_INDEX_TEAM2)]

    target_t1 = rng.choice(t1_ids)
    target_t2 = rng.choice(t2_ids)

    ball_positions = []
    for f in range(num_frames):
        if f < POSSESSION_HANDOVER_FRAME:
            tx, ty = player_positions_by_id[target_t1][f]
        else:
            tx, ty = player_positions_by_id[target_t2][f]

        # Smooth sinusoidal wobble around the target player
        wobble_x = int(15 * math.sin(2 * math.pi * 0.15 * f))
        wobble_y = int(10 * math.cos(2 * math.pi * 0.12 * f))
        bx = max(BALL_RADIUS, min(FRAME_W - BALL_RADIUS, tx + wobble_x))
        by = max(BALL_RADIUS, min(FRAME_H - BALL_RADIUS, ty + PLAYER_H // 2 - 10 + wobble_y))
        ball_positions.append((bx, by))

    return ball_positions


# ---------------------------------------------------------------------------
# Main generator
# ---------------------------------------------------------------------------

def generate(num_frames: int = 50, seed: int = 42,
             tracks_out: str = "stubs/dummy_tracks_stub.pkl",
             video_out: str = "input_videos/dummy_match.avi",
             gt_out: str = "stubs/dummy_ground_truth.pkl") -> None:
    """
    Generate synthetic video, tracks stub, and ground-truth file.

    Args:
        num_frames: Number of video frames to generate (default 50).
        seed: Random seed for reproducibility.
        tracks_out: Path to write the tracks stub pickle.
        video_out: Path to write the AVI video.
        gt_out: Path to write the ground-truth pickle.
    """
    rng = random.Random(seed)
    np.random.seed(seed)

    # ------------------------------------------------------------------
    # 1. Build player roster
    # ------------------------------------------------------------------
    # player_id → team (1 or 2)
    # IDs 1–11: team 1 (ID 1 is GK), IDs 12–22: team 2 (ID 12 is GK)
    team_per_id: dict[int, int] = {}
    gk_ids: list[int] = []
    for pid in range(1, 12):        # 11 team-1 players
        team_per_id[pid] = 1
    for pid in range(12, 23):       # 11 team-2 players
        team_per_id[pid] = 2
    gk_ids = [GK_INDEX_TEAM1, GK_INDEX_TEAM2]   # [1, 12]

    # referee IDs
    ref_ids = [101, 102]

    # Shirt colours per player (with noise)
    shirt_color: dict[int, tuple] = {}
    shorts_color: dict[int, tuple] = {}
    for pid, team in team_per_id.items():
        is_gk = pid in gk_ids
        if is_gk:
            shirt_base = GK1_SHIRT_BGR if team == 1 else GK2_SHIRT_BGR
        else:
            shirt_base = TEAM1_SHIRT_BGR if team == 1 else TEAM2_SHIRT_BGR
        shorts_base = TEAM1_SHORTS_BGR if team == 1 else TEAM2_SHORTS_BGR
        shirt_color[pid]  = _noisy_color(shirt_base, rng, amplitude=15)
        shorts_color[pid] = _noisy_color(shorts_base, rng, amplitude=10)

    for rid in ref_ids:
        shirt_color[rid]  = _noisy_color(REF_SHIRT_BGR, rng, amplitude=8)
        shorts_color[rid] = _noisy_color((0, 80, 80), rng, amplitude=8)

    # ------------------------------------------------------------------
    # 2. Generate smooth positions for all players and referees
    # ------------------------------------------------------------------
    player_positions: dict[int, list] = {}
    for pid, team in team_per_id.items():
        is_gk = pid in gk_ids
        player_positions[pid] = _make_player_positions(
            num_frames, pid, is_gk, team, rng)

    ref_base_positions = {
        101: (rng.randint(700, 900), rng.randint(300, 600)),
        102: (rng.randint(1000, 1200), rng.randint(400, 700)),
    }
    ref_positions: dict[int, list] = {}
    for rid, (bx, by) in ref_base_positions.items():
        rng_ref = random.Random(rid)
        positions = []
        for f in range(num_frames):
            cx = int(bx + 25 * math.sin(2 * math.pi * 0.07 * f))
            cy = int(by + 20 * math.cos(2 * math.pi * 0.09 * f))
            cx = max(PLAYER_W//2, min(FRAME_W - PLAYER_W//2, cx))
            cy = max(PLAYER_H//2, min(FRAME_H - PLAYER_H//2, cy))
            positions.append((cx, cy))
        ref_positions[rid] = positions

    # ------------------------------------------------------------------
    # 3. Generate ball path (all frames, including hidden ones)
    # ------------------------------------------------------------------
    ball_path = _make_ball_path(num_frames, player_positions, team_per_id, rng)

    # ------------------------------------------------------------------
    # 4. Build tracks dict (CONTRACT.md format)
    # ------------------------------------------------------------------
    tracks: dict = {"players": [], "referees": [], "ball": []}

    for f in range(num_frames):
        # --- players ---
        players_dict: dict = {}
        for pid in team_per_id:
            cx, cy = player_positions[pid][f]
            players_dict[pid] = {"bbox": _cx_cy_to_bbox(cx, cy)}
        tracks["players"].append(players_dict)

        # --- referees ---
        refs_dict: dict = {}
        for rid in ref_ids:
            cx, cy = ref_positions[rid][f]
            refs_dict[rid] = {"bbox": _cx_cy_to_bbox(cx, cy)}
        tracks["referees"].append(refs_dict)

        # --- ball (absent in specified frames) ---
        if f not in BALL_ABSENT_FRAMES:
            bx, by = ball_path[f]
            r = float(BALL_RADIUS)
            ball_bbox = [float(bx) - r, float(by) - r,
                         float(bx) + r, float(by) + r]
            tracks["ball"].append({1: {"bbox": ball_bbox}})
        else:
            tracks["ball"].append({})

    # ------------------------------------------------------------------
    # 5. Build ground-truth dict
    # ------------------------------------------------------------------
    gt = {
        "team_per_id": team_per_id,         # {player_id: 1 or 2}
        "gk_ids": gk_ids,                   # [1, 12]
        "ref_ids": ref_ids,                  # [101, 102]
        "ball_centre_all_frames": ball_path, # list of (cx,cy) for EVERY frame
        "ball_absent_frames": sorted(BALL_ABSENT_FRAMES),
        "num_frames": num_frames,
        "possession_handover_frame": POSSESSION_HANDOVER_FRAME,
        "frame_w": FRAME_W,
        "frame_h": FRAME_H,
    }

    # ------------------------------------------------------------------
    # 6. Render synthetic video
    # ------------------------------------------------------------------
    os.makedirs(os.path.dirname(os.path.abspath(video_out)), exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"XVID")
    writer = cv2.VideoWriter(str(video_out), fourcc, float(FPS), (FRAME_W, FRAME_H))

    base_pitch = _make_pitch_frame(rng)   # render once; stamp objects per frame

    for f in range(num_frames):
        frame = base_pitch.copy()

        # Draw referees (behind players visually — draw first)
        for rid in ref_ids:
            cx, cy = ref_positions[rid][f]
            bbox = _cx_cy_to_bbox(cx, cy)
            _draw_player_rect(frame, bbox, shirt_color[rid], shorts_color[rid])

        # Draw players
        for pid in team_per_id:
            cx, cy = player_positions[pid][f]
            bbox = _cx_cy_to_bbox(cx, cy)
            _draw_player_rect(frame, bbox, shirt_color[pid], shorts_color[pid])

        # Draw ball (even in absent frames so the video looks continuous)
        bx, by = ball_path[f]
        cv2.circle(frame, (bx, by), BALL_RADIUS, (255, 255, 255), -1)

        # Frame number overlay (for debugging)
        cv2.putText(frame, f"Frame {f}", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 0), 3)

        writer.write(frame)

    writer.release()
    print(f"Video written to: {video_out} ({num_frames} frames, {FPS} fps)")

    # ------------------------------------------------------------------
    # 7. Write stubs
    # ------------------------------------------------------------------
    for path, obj, label in [
        (tracks_out, tracks, "tracks stub"),
        (gt_out,     gt,     "ground-truth"),
    ]:
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        with open(path, "wb") as fh:
            pickle.dump(obj, fh, protocol=4)
        print(f"{label} written to: {path}")

    # ------------------------------------------------------------------
    # 8. Quick sanity report
    # ------------------------------------------------------------------
    total_frames = len(tracks["players"])
    assert total_frames == num_frames, "BUG: frame count mismatch in tracks"
    assert len(tracks["referees"]) == num_frames
    assert len(tracks["ball"]) == num_frames

    ball_present = sum(1 for b in tracks["ball"] if b)
    ball_absent  = sum(1 for b in tracks["ball"] if not b)
    print(f"Players per frame : {len(tracks['players'][0])} (IDs {sorted(tracks['players'][0])})")
    print(f"Referees per frame: {len(tracks['referees'][0])} (IDs {sorted(tracks['referees'][0])})")
    print(f"Ball present / absent: {ball_present} / {ball_absent} frames")
    print(f"Absent frame indices: {sorted(BALL_ABSENT_FRAMES)}")

    # Verify CONTRACT: bboxes are plain Python lists
    sample_pid = next(iter(tracks["players"][5]))
    sample_bbox = tracks["players"][5][sample_pid]["bbox"]
    assert isinstance(sample_bbox, list), f"bbox must be list, got {type(sample_bbox)}"
    assert all(isinstance(v, float) for v in sample_bbox), "bbox values must be float"

    print("Contract checks passed.")


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate synthetic football video and tracking stubs."
    )
    parser.add_argument("--frames", type=int, default=50,
                        help="Number of video frames to generate (default: 50).")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducibility (default: 42).")
    parser.add_argument("--tracks-out", default="stubs/dummy_tracks_stub.pkl",
                        help="Output path for the tracks stub pickle.")
    parser.add_argument("--video-out", default="input_videos/dummy_match.avi",
                        help="Output path for the synthetic AVI video.")
    parser.add_argument("--gt-out", default="stubs/dummy_ground_truth.pkl",
                        help="Output path for the ground-truth pickle.")
    args = parser.parse_args()

    generate(
        num_frames=args.frames,
        seed=args.seed,
        tracks_out=args.tracks_out,
        video_out=args.video_out,
        gt_out=args.gt_out,
    )
