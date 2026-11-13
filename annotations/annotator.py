"""
annotations/annotator.py
========================

Standalone visual annotation layer for football tracking analysis.
Draws tracking results directly from the `tracks` dictionary onto video frames:
  - Players: semi-transparent foot anchor ellipses and track ID badges
  - Referees: yellow foot anchor ellipses
  - Ball: green inverted triangle above the ball
  - Possession: red inverted triangle above the possessing player
  - Speed & Distance: numeric telemetry text below the ID badge (when available)

All coordinates are cast to int and partial off-screen boxes are handled gracefully.
No HUD overlay is drawn here (reserved for Part 5).
"""

import cv2
import numpy as np


# ===========================================================================
# Tunable Constants (All colours in BGR format)
# ===========================================================================

# Colors
DEFAULT_PLAYER_COLOR = (128, 128, 128)      # Neutral grey (BGR)
REFEREE_COLOR = (0, 255, 255)              # Bright yellow (BGR)
BALL_TRIANGLE_COLOR = (0, 255, 0)          # Green (BGR)
POSSESSION_TRIANGLE_COLOR = (0, 0, 255)    # Red (BGR)
OUTLINE_COLOR = (0, 0, 0)                  # Black outline
BADGE_TEXT_COLOR = (0, 0, 0)               # Black text on ID badge
SPEED_TEXT_COLOR = (255, 255, 255)         # White telemetry text

# Transparency
ELLIPSE_ALPHA = 0.6                        # Overlay blend weight (0.0 to 1.0)

# Ellipse geometry
ELLIPSE_VERTICAL_RATIO = 0.20              # vertical axis = 0.35 * width
ELLIPSE_START_ANGLE = -45                  # degrees
ELLIPSE_END_ANGLE = 235                    # degrees (open ground arc)
ELLIPSE_THICKNESS = 2

# Player ID badge
BADGE_WIDTH = 34                           # px
BADGE_HEIGHT = 18                          # px
BADGE_OFFSET_Y = 2                        # px below bbox bottom (cy)
BADGE_FONT = cv2.FONT_HERSHEY_SIMPLEX
BADGE_FONT_SCALE = 0.45
BADGE_FONT_THICKNESS = 2

# Marker Triangles (Inverted: tip pointing down)
TRIANGLE_HALF_WIDTH = 10                   # px
TRIANGLE_HEIGHT = 20                       # px
TRIANGLE_OFFSET = 4                        # px above bbox top (y1)
TRIANGLE_OUTLINE_THICKNESS = 1

# Speed & Distance telemetry text
SPEED_FONT = cv2.FONT_HERSHEY_SIMPLEX
SPEED_FONT_SCALE = 0.4
SPEED_FONT_THICKNESS = 1
SPEED_OFFSET_Y = 16                        # px below badge bottom
DISTANCE_OFFSET_Y = 30                     # px below badge bottom


# ===========================================================================
# Internal Drawing Helpers
# ===========================================================================

def _clamp(val: int, low: int, high: int) -> int:
    return max(low, min(high, val))


def _draw_inverted_triangle(frame: np.ndarray, cx: int, tip_y: int, fill_color: tuple, outline_color: tuple):
    """
    Draw an inverted triangle with its tip pointing down towards (cx, tip_y).
    """
    base_y = tip_y - TRIANGLE_HEIGHT
    pts = np.array([
        [cx, tip_y],
        [cx - TRIANGLE_HALF_WIDTH, base_y],
        [cx + TRIANGLE_HALF_WIDTH, base_y]
    ], dtype=np.int32)

    cv2.drawContours(frame, [pts], 0, fill_color, cv2.FILLED)
    cv2.drawContours(frame, [pts], 0, outline_color, TRIANGLE_OUTLINE_THICKNESS)


# ===========================================================================
# Annotator Class
# ===========================================================================

class Annotator:
    """
    Renders object tracks onto video frames.
    Standalone implementation conforming to docs/CONTRACT.md.
    """

    def draw_annotations(self, video_frames: list, tracks: dict) -> list:
        """
        Draw all tracking annotations onto video frames.

        Args:
            video_frames (list[np.ndarray]): List of input BGR video frames.
            tracks (dict): Tracking data dictionary with keys 'players', 'referees', 'ball'.

        Returns:
            list[np.ndarray]: New list of annotated frames (input frames remain unmodified).
        """
        output_frames = []
        num_frames = len(video_frames)

        player_tracks = tracks.get("players", [])
        referee_tracks = tracks.get("referees", [])
        ball_tracks = tracks.get("ball", [])

        for frame_idx, raw_frame in enumerate(video_frames):
            frame = raw_frame.copy()
            frame_h, frame_w = frame.shape[:2]

            players_dict = player_tracks[frame_idx] if frame_idx < len(player_tracks) else {}
            referees_dict = referee_tracks[frame_idx] if frame_idx < len(referee_tracks) else {}
            ball_dict = ball_tracks[frame_idx] if frame_idx < len(ball_tracks) else {}

            # ------------------------------------------------------------------
            # a) & c) Ellipses under the feet (Semi-transparent overlay, blended ONCE)
            # ------------------------------------------------------------------
            overlay = frame.copy()

            # Depth sort entities back-to-front (increasing y2) so foreground players cleanly sit on top
            sorted_players = sorted(
                players_dict.items(),
                key=lambda item: item[1]["bbox"][3] if "bbox" in item[1] and len(item[1]["bbox"]) == 4 else 0,
            )
            sorted_referees = sorted(
                referees_dict.items(),
                key=lambda item: item[1]["bbox"][3] if "bbox" in item[1] and len(item[1]["bbox"]) == 4 else 0,
            )

            # Player ellipses
            for track_id, p_info in sorted_players:
                if "bbox" not in p_info:
                    continue
                bbox = p_info["bbox"]
                x1, y1, x2, y2 = [int(v) for v in bbox]
                cx = int((x1 + x2) / 2)
                cy = y2
                width = int(x2 - x1)
                if width < 2:
                    continue

                color = p_info.get("team_color", DEFAULT_PLAYER_COLOR)
                color = tuple(int(c) for c in color)

                axes = (max(8, int(width * 0.50)), max(3, int(width * ELLIPSE_VERTICAL_RATIO)))
                cv2.ellipse(
                    overlay,
                    center=(cx, cy),
                    axes=axes,
                    angle=0.0,
                    startAngle=ELLIPSE_START_ANGLE,
                    endAngle=ELLIPSE_END_ANGLE,
                    color=color,
                    thickness=ELLIPSE_THICKNESS,
                    lineType=cv2.LINE_AA,
                )

            # Referee ellipses
            for track_id, r_info in sorted_referees:
                if "bbox" not in r_info:
                    continue
                bbox = r_info["bbox"]
                x1, y1, x2, y2 = [int(v) for v in bbox]
                cx = int((x1 + x2) / 2)
                cy = y2
                width = int(x2 - x1)
                if width < 2:
                    continue

                axes = (max(8, int(width * 0.50)), max(3, int(width * ELLIPSE_VERTICAL_RATIO)))
                cv2.ellipse(
                    overlay,
                    center=(cx, cy),
                    axes=axes,
                    angle=0.0,
                    startAngle=ELLIPSE_START_ANGLE,
                    endAngle=ELLIPSE_END_ANGLE,
                    color=REFEREE_COLOR,
                    thickness=ELLIPSE_THICKNESS,
                    lineType=cv2.LINE_AA,
                )

            # Blend ellipses overlay once per frame
            cv2.addWeighted(overlay, ELLIPSE_ALPHA, frame, 1.0 - ELLIPSE_ALPHA, 0, frame)

            # ------------------------------------------------------------------
            # b) Player ID Badges, e) Possession, f) Speed & Distance
            # ------------------------------------------------------------------
            for track_id, p_info in sorted_players:
                if "bbox" not in p_info:
                    continue
                bbox = p_info["bbox"]
                x1, y1, x2, y2 = [int(v) for v in bbox]
                cx = int((x1 + x2) / 2)
                cy = y2

                color = p_info.get("team_color", DEFAULT_PLAYER_COLOR)
                color = tuple(int(c) for c in color)

                # b) ID Badge
                rect_x1 = cx - BADGE_WIDTH // 2
                rect_x2 = cx + BADGE_WIDTH // 2
                rect_y1 = cy + BADGE_OFFSET_Y
                rect_y2 = rect_y1 + BADGE_HEIGHT

                cv2.rectangle(frame, (rect_x1, rect_y1), (rect_x2, rect_y2), color, cv2.FILLED)
                cv2.rectangle(frame, (rect_x1, rect_y1), (rect_x2, rect_y2), (0, 0, 0), 1, cv2.LINE_AA)

                label = str(track_id)
                (tw, th), _ = cv2.getTextSize(label, BADGE_FONT, BADGE_FONT_SCALE, BADGE_FONT_THICKNESS)
                text_x = cx - tw // 2
                text_y = rect_y1 + (BADGE_HEIGHT + th) // 2
                cv2.putText(
                    frame,
                    label,
                    (text_x, text_y),
                    BADGE_FONT,
                    BADGE_FONT_SCALE,
                    BADGE_TEXT_COLOR,
                    BADGE_FONT_THICKNESS,
                    cv2.LINE_AA,
                )

                # e) Possession indicator: red inverted triangle above player bbox
                if p_info.get("has_ball", False):
                    tip_y = y1 - TRIANGLE_OFFSET
                    _draw_inverted_triangle(
                        frame,
                        cx=cx,
                        tip_y=tip_y,
                        fill_color=POSSESSION_TRIANGLE_COLOR,
                        outline_color=OUTLINE_COLOR,
                    )

                # f) Speed and distance telemetry text
                # TODO(A-PUSH): confirm key names and units (km/h, metres) with Person A, and confirm he is not also drawing them.
                has_speed = "speed" in p_info and p_info["speed"] is not None
                has_distance = "distance" in p_info and p_info["distance"] is not None

                if has_speed:
                    speed_val = p_info["speed"]
                    speed_text = f"{speed_val:.1f} km/h"
                    (sw, sh), _ = cv2.getTextSize(speed_text, SPEED_FONT, SPEED_FONT_SCALE, SPEED_FONT_THICKNESS)
                    sx = cx - sw // 2
                    sy = rect_y2 + SPEED_OFFSET_Y
                    cv2.putText(
                        frame,
                        speed_text,
                        (sx, sy),
                        SPEED_FONT,
                        SPEED_FONT_SCALE,
                        SPEED_TEXT_COLOR,
                        SPEED_FONT_THICKNESS,
                        cv2.LINE_AA,
                    )

                if has_distance:
                    dist_val = p_info["distance"]
                    dist_text = f"{dist_val:.0f} m"
                    (dw, dh), _ = cv2.getTextSize(dist_text, SPEED_FONT, SPEED_FONT_SCALE, SPEED_FONT_THICKNESS)
                    dx = cx - dw // 2
                    dy = rect_y2 + (DISTANCE_OFFSET_Y if has_speed else SPEED_OFFSET_Y)
                    cv2.putText(
                        frame,
                        dist_text,
                        (dx, dy),
                        SPEED_FONT,
                        SPEED_FONT_SCALE,
                        SPEED_TEXT_COLOR,
                        SPEED_FONT_THICKNESS,
                        cv2.LINE_AA,
                    )

            # ------------------------------------------------------------------
            # Referee Badge: Dedicated "REF" sign below feet in referee yellow
            # ------------------------------------------------------------------
            for track_id, r_info in sorted_referees:
                if "bbox" not in r_info:
                    continue
                bbox = r_info["bbox"]
                x1, y1, x2, y2 = [int(v) for v in bbox]
                cx = int((x1 + x2) / 2)
                cy = y2

                ref_badge_w = max(BADGE_WIDTH, 38)
                rect_x1 = cx - ref_badge_w // 2
                rect_x2 = cx + ref_badge_w // 2
                rect_y1 = cy + BADGE_OFFSET_Y
                rect_y2 = rect_y1 + BADGE_HEIGHT

                cv2.rectangle(frame, (rect_x1, rect_y1), (rect_x2, rect_y2), REFEREE_COLOR, cv2.FILLED)
                cv2.rectangle(frame, (rect_x1, rect_y1), (rect_x2, rect_y2), OUTLINE_COLOR, 1, cv2.LINE_AA)

                label = "REF"
                (tw, th), _ = cv2.getTextSize(label, BADGE_FONT, BADGE_FONT_SCALE, BADGE_FONT_THICKNESS)
                text_x = cx - tw // 2
                text_y = rect_y1 + (BADGE_HEIGHT + th) // 2
                cv2.putText(
                    frame,
                    label,
                    (text_x, text_y),
                    BADGE_FONT,
                    BADGE_FONT_SCALE,
                    BADGE_TEXT_COLOR,
                    BADGE_FONT_THICKNESS,
                    cv2.LINE_AA,
                )

            # ------------------------------------------------------------------
            # d) Ball: filled green inverted triangle with dark outline
            # ------------------------------------------------------------------
            for b_id, b_info in ball_dict.items():
                if "bbox" not in b_info:
                    continue
                b_box = b_info["bbox"]
                bx1, by1, bx2, by2 = [int(v) for v in b_box]
                bcx = int((bx1 + bx2) / 2)
                btip_y = by1 - TRIANGLE_OFFSET
                _draw_inverted_triangle(
                    frame,
                    cx=bcx,
                    tip_y=btip_y,
                    fill_color=BALL_TRIANGLE_COLOR,
                    outline_color=OUTLINE_COLOR,
                )

            output_frames.append(frame)

        return output_frames

