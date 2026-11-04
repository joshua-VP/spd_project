"""
tests/test_annotator.py
=======================

Unit test suite for the Annotator class (annotations/annotator.py).
Tests all functional requirements using synthetic in-memory frames:
  - Output frame count matches input frame count
  - Input frames are never modified in-place
  - Local pixel modification (ellipse area modified, distant pixels untouched)
  - Resilient to empty frame dicts and missing ball keys
  - Resilient to off-screen, clipped, or degenerate bounding boxes
  - Telemetry speed text drawn only when 'speed' key is present
  - Possession indicator triangle appears only when 'has_ball' is True
"""

import sys
import os
import cv2
import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from annotations.annotator import Annotator, DEFAULT_PLAYER_COLOR, POSSESSION_TRIANGLE_COLOR


# ---------------------------------------------------------------------------
# Test Helpers & Fixtures
# ---------------------------------------------------------------------------

def _make_frame(height=300, width=400, color=(50, 120, 50)):
    """Create a solid color synthetic BGR frame."""
    frame = np.full((height, width, 3), color, dtype=np.uint8)
    return frame


def _make_single_frame_tracks(players=None, referees=None, ball=None):
    """Create a minimal 1-frame tracks dict."""
    return {
        "players": [players if players is not None else {}],
        "referees": [referees if referees is not None else {}],
        "ball": [ball if ball is not None else {}],
    }


# ---------------------------------------------------------------------------
# 1. Output Frame Count
# ---------------------------------------------------------------------------

class TestOutputFrameCount:
    def test_empty_input(self):
        annotator = Annotator()
        out = annotator.draw_annotations([], {"players": [], "referees": [], "ball": []})
        assert len(out) == 0

    def test_single_frame(self):
        annotator = Annotator()
        frames = [_make_frame()]
        tracks = _make_single_frame_tracks()
        out = annotator.draw_annotations(frames, tracks)
        assert len(out) == 1

    def test_multiple_frames(self):
        annotator = Annotator()
        n = 10
        frames = [_make_frame() for _ in range(n)]
        tracks = {
            "players": [{} for _ in range(n)],
            "referees": [{} for _ in range(n)],
            "ball": [{} for _ in range(n)],
        }
        out = annotator.draw_annotations(frames, tracks)
        assert len(out) == n


# ---------------------------------------------------------------------------
# 2. Input Frames Immutability
# ---------------------------------------------------------------------------

class TestInputImmutability:
    def test_frames_not_modified_in_place(self):
        annotator = Annotator()
        frame = _make_frame()
        original = frame.copy()

        tracks = _make_single_frame_tracks(
            players={1: {"bbox": [50.0, 50.0, 100.0, 150.0], "team_color": (0, 0, 255)}},
            ball={1: {"bbox": [150.0, 150.0, 160.0, 160.0]}}
        )

        out = annotator.draw_annotations([frame], tracks)
        # Original input frame must remain identical
        np.testing.assert_array_equal(frame, original, "Input frame was modified in-place!")
        # Output must be a distinct object
        assert out[0] is not frame


# ---------------------------------------------------------------------------
# 3. Local Pixel Changes
# ---------------------------------------------------------------------------

class TestPixelModifications:
    def test_ellipse_pixels_change_and_far_pixels_do_not(self):
        annotator = Annotator()
        frame = _make_frame(height=400, width=400, color=(0, 0, 0))

        # Player bbox at [100, 100, 160, 200]
        # Foot is at cx=130, cy=200
        bbox = [100.0, 100.0, 160.0, 200.0]
        tracks = _make_single_frame_tracks(
            players={1: {"bbox": bbox, "team_color": (255, 0, 0)}}
        )

        out = annotator.draw_annotations([frame], tracks)[0]

        # Area near ellipse arc (e.g. around foot cx=130, cy=200) should have changed
        foot_region = out[180:230, 90:170]
        assert np.any(foot_region != 0), "Ellipse and badge were not drawn near foot position!"

        # Far away pixel at corner (10, 10) must remain untouched
        assert np.all(out[10, 10] == 0), "Far corner pixel was unexpectedly modified!"
        assert np.all(out[380, 20] == 0), "Far bottom-left pixel was unexpectedly modified!"


# ---------------------------------------------------------------------------
# 4. Empty Dicts & Missing Ball Resilience
# ---------------------------------------------------------------------------

class TestEmptyDictsResilience:
    def test_empty_frame_dicts(self):
        annotator = Annotator()
        frames = [_make_frame()]
        tracks = {"players": [{}], "referees": [{}], "ball": [{}]}
        out = annotator.draw_annotations(frames, tracks)
        assert len(out) == 1

    def test_missing_ball_key(self):
        annotator = Annotator()
        frames = [_make_frame(), _make_frame()]
        tracks = {
            "players": [{}, {}],
            "referees": [{}, {}],
            "ball": [{1: {"bbox": [10.0, 10.0, 20.0, 20.0]}}, {}]  # frame 1 has no ball
        }
        out = annotator.draw_annotations(frames, tracks)
        assert len(out) == 2

    def test_missing_bbox_field(self):
        annotator = Annotator()
        frames = [_make_frame()]
        tracks = _make_single_frame_tracks(
            players={1: {"team_color": (0, 0, 255)}},  # missing 'bbox'
            referees={2: {}},
            ball={1: {}}
        )
        out = annotator.draw_annotations(frames, tracks)
        assert len(out) == 1


# ---------------------------------------------------------------------------
# 5. Off-Screen & Degenerate Bounding Boxes
# ---------------------------------------------------------------------------

class TestOffScreenBoxes:
    @pytest.mark.parametrize("bad_bbox", [
        [-50.0, -50.0, 20.0, 20.0],                # Top-left clipped
        [380.0, 280.0, 450.0, 350.0],              # Bottom-right clipped
        [-200.0, -200.0, -100.0, -100.0],          # Completely off-screen
        [100.0, 100.0, 100.0, 100.0],              # Zero width/height degenerate
        [0.0, 0.0, 1000.0, 1000.0],                # Larger than frame
    ])
    def test_clipped_boxes_do_not_crash(self, bad_bbox):
        annotator = Annotator()
        frames = [_make_frame(height=300, width=400)]
        tracks = _make_single_frame_tracks(
            players={1: {"bbox": bad_bbox, "team_color": (255, 0, 0)}},
            referees={2: {"bbox": bad_bbox}},
            ball={1: {"bbox": bad_bbox}}
        )
        out = annotator.draw_annotations(frames, tracks)
        assert len(out) == 1


# ---------------------------------------------------------------------------
# 6. Speed & Distance Telemetry
# ---------------------------------------------------------------------------

class TestSpeedAndDistance:
    def test_speed_drawn_only_when_present(self):
        annotator = Annotator()
        frame = _make_frame(color=(0, 0, 0))

        bbox = [100.0, 100.0, 150.0, 180.0]
        tracks_without_speed = _make_single_frame_tracks(
            players={1: {"bbox": bbox, "team_color": (255, 0, 0)}}
        )
        tracks_with_speed = _make_single_frame_tracks(
            players={1: {"bbox": bbox, "team_color": (255, 0, 0), "speed": 24.1}}
        )

        out_no_speed = annotator.draw_annotations([frame], tracks_without_speed)[0]
        out_speed = annotator.draw_annotations([frame], tracks_with_speed)[0]

        # Must differ because speed text was rendered
        assert not np.array_equal(out_no_speed, out_speed), "Speed text was not rendered when 'speed' key was provided!"

    def test_distance_drawn_only_when_present(self):
        annotator = Annotator()
        frame = _make_frame(color=(0, 0, 0))

        bbox = [100.0, 100.0, 150.0, 180.0]
        tracks_no_dist = _make_single_frame_tracks(
            players={1: {"bbox": bbox, "team_color": (255, 0, 0)}}
        )
        tracks_dist = _make_single_frame_tracks(
            players={1: {"bbox": bbox, "team_color": (255, 0, 0), "distance": 143.0}}
        )

        out_no_dist = annotator.draw_annotations([frame], tracks_no_dist)[0]
        out_dist = annotator.draw_annotations([frame], tracks_dist)[0]

        assert not np.array_equal(out_no_dist, out_dist), "Distance text was not rendered when 'distance' key was provided!"


# ---------------------------------------------------------------------------
# 7. Possession Indicator
# ---------------------------------------------------------------------------

class TestPossessionTriangle:
    def test_possession_triangle_when_has_ball_true(self):
        annotator = Annotator()
        frame = _make_frame(color=(0, 0, 0))

        bbox = [100.0, 100.0, 160.0, 200.0]
        tracks_no_ball = _make_single_frame_tracks(
            players={1: {"bbox": bbox, "team_color": (255, 0, 0), "has_ball": False}}
        )
        tracks_has_ball = _make_single_frame_tracks(
            players={1: {"bbox": bbox, "team_color": (255, 0, 0), "has_ball": True}}
        )

        out_no_ball = annotator.draw_annotations([frame], tracks_no_ball)[0]
        out_has_ball = annotator.draw_annotations([frame], tracks_has_ball)[0]

        assert not np.array_equal(out_no_ball, out_has_ball), "Possession triangle did not render when has_ball=True!"

        # Check for red channel pixels above player bbox (cx=130, y1=100)
        above_player_region = out_has_ball[70:100, 115:145]
        # Red in BGR is (0, 0, 255)
        has_red = np.any((above_player_region[:, :, 2] > 200) & (above_player_region[:, :, 0] < 50))
        assert has_red, "Red possession triangle was not found above the player bbox!"

    def test_possession_triangle_absent_when_false_or_missing(self):
        annotator = Annotator()
        frame = _make_frame(color=(0, 0, 0))

        bbox = [100.0, 100.0, 160.0, 200.0]
        tracks_false = _make_single_frame_tracks(
            players={1: {"bbox": bbox, "team_color": (255, 0, 0), "has_ball": False}}
        )
        tracks_missing = _make_single_frame_tracks(
            players={1: {"bbox": bbox, "team_color": (255, 0, 0)}}
        )

        out_false = annotator.draw_annotations([frame], tracks_false)[0]
        out_missing = annotator.draw_annotations([frame], tracks_missing)[0]

        np.testing.assert_array_equal(out_false, out_missing, "has_ball=False differed from missing has_ball key!")

