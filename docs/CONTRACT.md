# Shared Data Contract — `tracks` Dictionary

> This document is the **single source of truth** for the data format shared
> between Person A (Detection / Camera / Homography / Speed lead) and
> Person B (Analytics / Team Clustering / Possession lead).
> Any change that breaks either party's assumptions must be discussed and
> reflected here before the code changes land.

---

## 1. Top-Level Structure

```python
tracks = {
    "players":  [ {track_id: {"bbox": [x1, y1, x2, y2], ...}}, ... ],
    "referees": [ {track_id: {"bbox": [x1, y1, x2, y2], ...}}, ... ],
    "ball":     [ {1:         {"bbox": [x1, y1, x2, y2], ...}}, ... ],
}
```

| Field | Type | Description |
|-------|------|-------------|
| `"players"` | `list[dict]` | One dict per video frame; maps `track_id → object_info`. |
| `"referees"` | `list[dict]` | Same structure as `"players"`. |
| `"ball"` | `list[dict]` | Same structure; ball is **always assigned key `1`**. |

---

## 2. Structural Invariants

### 2.1 Length
```
len(tracks["players"]) == len(tracks["referees"]) == len(tracks["ball"])
                       == total number of video frames
```

### 2.2 Missed Detections → Absent Key (Never `None`)
When an object is not detected in a frame the **key is simply absent** from that
frame's dict.  Code must never store `None`, `{}` for a present-but-empty bbox,
or any sentinel value.

```python
# CORRECT — ball not seen in frame 5
tracks["ball"][5] == {}

# WRONG
tracks["ball"][5] == {1: None}
tracks["ball"][5] == {1: {"bbox": None}}
```

### 2.3 Bounding Box Format
```python
bbox: list[float]  # [x1, y1, x2, y2]  — plain Python list, not numpy array
```
- Coordinates are in **pixel space** of the original video frame.
- `x1 < x2`, `y1 < y2` (top-left → bottom-right convention).
- Must be a **plain Python `list`** — not `np.ndarray`, not `tuple`.
  Downstream code uses `bbox[0]` indexing and `pickle` round-trips must stay clean.

### 2.4 Track IDs
- `track_id: int` — ByteTrack persistent integer IDs.
- Goalkeepers are detected as a separate class by YOLO but are **merged into
  `"players"`** by Person A's tracker before the dict is returned.  There is no
  `"goalkeepers"` key.
- Ball track ID is always `1` (fixed by convention, not from ByteTrack).
- Referee track IDs are independent integers from ByteTrack.

---

## 3. Per-Object-Info Keys

Each value in the per-frame dict is itself a `dict` with the following keys.
Keys are split by **ownership** — each person writes only their own keys and
must not modify the other's.

### 3.1 Shared / written by Person A (Detection & Camera lead)

| Key | Type | Written when | Description |
|-----|------|-------------|-------------|
| `"bbox"` | `list[float]` | Always (by tracker) | `[x1, y1, x2, y2]` pixel coords. |
| `"position"` | `tuple[int, int]` | After `Tracker.add_position_to_tracks()` | Foot position (bottom-centre) for players/refs; centre for ball. |
| `"position_adjusted"` | `tuple[float, float]` | After `CameraMovementEstimator.add_adjust_positions_to_tracks()` | `position` minus cumulative camera displacement. |
| `"position_transformed"` | `tuple[float, float]` | After homography (future) | Real-world pitch coordinates (metres). |
| `"speed"` | `float` | After speed estimator (future) | Speed in km/h. |
| `"distance"` | `float` | After speed estimator (future) | Cumulative distance in metres. |

### 3.2 Written by Person B (Analytics lead)

| Key | Type | Written when | Description |
|-----|------|-------------|-------------|
| `"team"` | `int` | After `TeamAssigner.get_player_team()` | `1` or `2`; only for `"players"`. |
| `"team_color"` | `tuple/np.ndarray` | After team assignment | BGR colour of the team (for rendering ellipses). |
| `"has_ball"` | `bool` | After `PlayerBallAssigner` | `True` if this player currently has possession. |

> [!IMPORTANT]
> Person B **must not** write `"position"`, `"position_adjusted"`, `"speed"`, or
> `"distance"`. Person A **must not** write `"team"`, `"team_color"`, or `"has_ball"`.

---

## 4. Ball Interpolation

Before team assignment and possession logic run, the ball list is processed by
`Tracker.interpolate_ball_positions()` (Person A's code).  After interpolation:
- Every frame has a ball entry (`{1: {"bbox": [...]}}`) — no absent frames remain.
- Bboxes that were absent are filled by **linear interpolation**; leading absent
  frames are **backfilled** (`bfill`).
- The interpolated bboxes are plain Python lists (the method calls `.tolist()`).

---

## 5. Example — Frame-Level Snapshot

```python
# Frame 42, single player, ball visible
tracks["players"][42] = {
    7:  {"bbox": [320.0, 410.0, 370.0, 530.0],
         "position": (345, 530),
         "position_adjusted": (341.2, 527.8),
         "team": 1,
         "team_color": (60, 40, 200),
         "has_ball": True},
    12: {"bbox": [800.0, 380.0, 845.0, 490.0],
         "position": (822, 490),
         "position_adjusted": (818.2, 487.8),
         "team": 2,
         "team_color": (200, 80, 50),
         "has_ball": False},
}
tracks["referees"][42] = {
    3: {"bbox": [600.0, 400.0, 635.0, 500.0],
        "position": (617, 500),
        "position_adjusted": (613.2, 497.8)},
}
tracks["ball"][42] = {
    1: {"bbox": [336.0, 518.0, 356.0, 538.0]},
}
```

---

## 6. Pickle Stub Convention

- Stubs are plain `pickle` files (`protocol=4`) saved with `open(path, "wb")`.
- Stub keys and value types must match this contract exactly.
- Person A's real stub path: `stubs/tracks_stub.pkl`
- Dummy test stub path:      `stubs/dummy_tracks_stub.pkl`

---

## 7. Contract Decisions Made by Person B (pending A's confirmation)

The following decisions were made unilaterally during scaffold creation and
should be confirmed with Person A:

1. **Ball key is always integer `1`** — not a string, not a ByteTrack ID.
2. **Missing detection = absent key** — `{}` for the frame dict, never `None`.
3. **`bbox` is a plain Python `list[float]`** — `.tolist()` must be called on
   any numpy output before storing.
4. **Goalkeeper merge happens inside `Tracker.get_object_tracks()`** — the
   returned dict has no `"goalkeepers"` key.
5. **`"team"` is 1-indexed** (`1` or `2`, matching `team_colors[1]` and
   `team_colors[2]` in `TeamAssigner`).

---

*Last updated: 2026-11-04 by Person B*

