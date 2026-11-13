# Integration TODO — Person B → Person A handshake items

> Every item in this file has a corresponding `# TODO(A-PUSH)` comment in the
> source code.  When Person A's code lands, work through this list top-to-bottom,
> check the box, and delete the matching comment from the code.

---

## A-PUSH Items (from source code)

- [x] **[generate_dummy_stub.py]** Switch `stubs/dummy_tracks_stub.pkl` to
  `stubs/tracks_stub.pkl` once Person A's real YOLO+ByteTrack stub is available.
  Confirm it matches CONTRACT.md (absent keys, ball key `1`, plain Python list bboxes,
  no `None` sentinels). *(Confirmed and verified: 1,125 frames, 0 contract violations)*

- [x] **[utils/video_utils.py — `get_video_fps`]** Person A's speed estimator
  must call `get_video_fps(video_path)` from `utils/video_utils.py` instead of
  hard-coding `fps = 24`.  Failing to do so will produce incorrect km/h values
  for footage at non-24-fps frame rates (e.g. 25 fps broadcast, 50 fps slow-mo).
  *(Integrated into main.py and run_person_a_on_real_video.py)*

- [x] **[dev_run.py]** The stub-loading, CONTRACT invariant checks, and annotation
  logic in `dev_run.py` should migrate into `main.py` at integration time.
  `dev_run.py` can then be removed or repurposed as a pure CI smoke test.
  *(CONTRACT invariant checks and Annotator drawing integrated into main.py)*

- [ ] **[annotations/annotator.py — speed/distance telemetry]** Confirm key names
  and units (`"speed"` in km/h, `"distance"` in metres) with Person A, and confirm
  Person A is not also drawing them in `trackers/tracker.py` so there are no duplicate
  overlays at integration time. *(annotations/annotator.py)*

---

## Known Integration Items (not yet in source code)

- [x] **`main.py` must call `Annotator.draw_annotations` for all drawing.**
  At integration, replace `Tracker.draw_annotations` with `Annotator().draw_annotations`
  in `main.py`. The `Annotator` reads `team_color`, `has_ball`, `speed`, and `distance`
  directly from the tracks dict. *(Integrated into main.py)*

- [ ] **Person A's drawing code duplicates `annotations/annotator.py`.**
  Person A's code in `trackers/tracker.py`, `demo_person_a.py`, and
  `run_person_a_on_real_video.py` contains redundant drawing logic (`draw_bbox`,
  `draw_ellipse`, `draw_triangle`, `draw_team_ball_control`, `draw_person_a_visuals`).
  At integration, his drawing code is removed or bypassed in favor of `annotations/annotator.py`.

- [ ] **Confirm speed/distance key names, units, and who draws them.**
  Confirm Person A's speed and distance module outputs `"speed"` (km/h) and
  `"distance"` (metres) in the tracks dict, and that Person A does not attempt to draw
  them in his tracker.

- [x] **`main.py` and `run_person_a_on_real_video.py` FPS handling.**
  `main.py` and `run_person_a_on_real_video.py` pass `fps=fps`. Confirm that this value
  comes from `get_video_fps(video_path)`, not a hardcoded number. *(Done)*

- [x] **`test_person_a.py` location.**
  `test_person_a.py` is currently in the repo root, so pytest (`testpaths: tests` in `pytest.ini`)
  never runs it. Person A should move it into `tests/test_person_a.py` and update imports so it
  participates in automated test runs. *(Moved canonical tests to tests/test_person_a.py, root wrapper retained)*

- [x] **Confirm CONTRACT format of A's real stub.**
  When `stubs/tracks_stub.pkl` lands, verify:
  - `len(tracks["players"]) == len(tracks["referees"]) == len(tracks["ball"]) == num_frames`
  - Absent detection frames are **empty dicts `{}`**, never `{key: None}`.
  - Ball is always `{1: {"bbox": [...]}}` (integer key `1`, not string `"1"`).
  - Every `bbox` is a **plain Python `list[float]`** (call `.tolist()` on numpy arrays).
  *(Verified: 1,125 frames match CONTRACT.md invariants)*

- [x] **Confirm goalkeeper merge.**
  `Tracker.get_object_tracks()` overrides goalkeeper class ID → player class ID before
  ByteTrack runs. Confirm with Person A that:
  - After the override, goalkeeper track IDs appear inside `tracks["players"]`, not in a
    separate key.
  - The ground-truth GK IDs from `stubs/dummy_ground_truth.pkl` (`gk_ids = [1, 12]`)
    should be replaced with the **real ByteTrack IDs** once A's stub is used.
  *(Verified in trackers/tracker.py lines 195-200)*

- [ ] **Homography / `position_transformed` key.**
  Once Person A's `ViewTransformer` / homography module lands, confirm it writes
  `"position_transformed"` into the tracks dict (not any other key name) and that
  Person B's downstream modules do not accidentally read stale `"position_adjusted"` values
  when `"position_transformed"` is available.

- [x] **`camera_movement_stub.pkl` format.**
  Confirm Person A's camera stub is `list[list[float]]` with `len == num_frames`,
  first element `[0.0, 0.0]`.  Person B's `dev_run.py` does not use it, but `main.py` does.
  *(Verified)*

- [x] **`measure_xy_distance` adoption by Person A.**
  `utils.measure_xy_distance(p1, p2)` has been added to `utils/bbox_utils.py` and
  exported from `utils/__init__.py`.  Person A's camera estimator currently computes
  dx/dy inline — consider switching to `measure_xy_distance` for consistency.
  *(Adopted in camera_movement_estimator/camera_movement_estimator.py)*

---

## Completed Items

- Backward-compatible `generate_dummy_tracks()` restored to `generate_dummy_stub.py`.
- `main.py` now verifies CONTRACT invariants, reads FPS via `get_video_fps()`, and renders with `Annotator` (ellipses, inverted triangles, player badges) + possession statistics HUD overlay.
- `camera_movement_estimator.py` uses `utils.measure_xy_distance`.
- Test suite in `tests/test_person_a.py` added to pytest runner (56 tests passing).
- `run_person_a_on_real_video.py` uses `get_video_fps` and `ellipse` draw mode.

---

*Last updated: 2026-11-05*
