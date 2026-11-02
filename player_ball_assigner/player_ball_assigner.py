import sys

sys.path.append("../")
from utils import get_center_of_bbox, measure_distance


class PlayerBallAssigner:
    """
    Assigns ball possession to the closest player within a threshold distance.

    Uses Euclidean distance between ball center and each player's foot position
    to determine which player currently has the ball.
    """

    def __init__(self):
        self.max_player_ball_distance = 70  # Pixel threshold

    def assign_ball_to_player(self, players, ball_bbox):
        """
        Find the player closest to the ball within the distance threshold.

        Args:
            players (dict): {track_id: {"bbox": [...], "position_adjusted": (x, y)}}.
            ball_bbox (list): Ball bounding box [x1, y1, x2, y2].

        Returns:
            int: Track ID of the player assigned the ball, or -1 if no player
                is within the threshold.
        """
        ball_position = get_center_of_bbox(ball_bbox)

        minimum_distance = 99999
        assigned_player = -1

        for player_id, player in players.items():
            player_position = player.get("position_adjusted", player.get("position"))
            if player_position is None:
                continue

            distance = measure_distance(ball_position, player_position)

            if distance < self.max_player_ball_distance:
                if distance < minimum_distance:
                    minimum_distance = distance
                    assigned_player = player_id

        return assigned_player

