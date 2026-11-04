from sklearn.cluster import KMeans
import numpy as np


class TeamAssigner:
    """
    Assigns players to teams based on jersey color using K-Means clustering.

    Uses a two-stage approach:
    1. Extract dominant jersey color per player (2-cluster K-Means on top half of bbox).
    2. Cluster all jersey colors into 2 global teams.
    """

    def __init__(self):
        self.team_colors = {}
        self.player_team_dict = {}
        self.kmeans = None

    def get_clustering_model(self, image):
        """
        Fit a 2-cluster K-Means model on the pixel colors of an image.

        Args:
            image (np.ndarray): BGR image crop.

        Returns:
            KMeans: Fitted KMeans model.
        """
        # Reshape image to 2D array of pixels
        image_2d = image.reshape(-1, 3)

        kmeans = KMeans(n_clusters=2, init="k-means++", n_init=1, random_state=0)
        kmeans.fit(image_2d)

        return kmeans

    def get_player_color(self, frame, bbox):
        """
        Extract the dominant jersey color from the top half of a player bbox.

        Crops the top 50% of the bounding box (the jersey area), runs 2-cluster
        K-Means to separate shirt color from grass/background, and returns the
        non-corner cluster center (the jersey color).

        Args:
            frame (np.ndarray): Full video frame.
            bbox (list): Player bounding box [x1, y1, x2, y2].

        Returns:
            np.ndarray: RGB color array of the jersey.
        """
        image = frame[int(bbox[1]) : int(bbox[3]), int(bbox[0]) : int(bbox[2])]

        # Take top half of the player crop (jersey area)
        top_half_image = image[0 : int(image.shape[0] / 2), :]

        # Get K-Means clustering
        kmeans = self.get_clustering_model(top_half_image)

        # Get the cluster labels for each pixel
        labels = kmeans.labels_

        # Reshape labels to original image shape
        clustered_image = labels.reshape(
            top_half_image.shape[0], top_half_image.shape[1]
        )

        # Determine which cluster is the player (not the corner/background)
        corner_clusters = [
            clustered_image[0, 0],
            clustered_image[0, -1],
            clustered_image[-1, 0],
            clustered_image[-1, -1],
        ]
        non_player_cluster = max(set(corner_clusters), key=corner_clusters.count)
        player_cluster = 1 - non_player_cluster

        player_color = kmeans.cluster_centers_[player_cluster]

        return player_color

    def assign_team_color(self, frame, player_detections):
        """
        Collect jersey colors across detected players and establish two
        global team colors using 2-cluster K-Means.

        Should be called once with detections from the first frame (or a few
        initial frames) to establish team colors.

        Args:
            frame (np.ndarray): Video frame.
            player_detections (dict): {track_id: {"bbox": [x1,y1,x2,y2]}}.
        """
        player_colors = []
        for _, player_detection in player_detections.items():
            bbox = player_detection["bbox"]
            player_color = self.get_player_color(frame, bbox)
            player_colors.append(player_color)

        if len(player_colors) < 2:
            self.team_colors[1] = np.array([220, 50, 40])  # Blue default
            self.team_colors[2] = np.array([40, 40, 220])  # Red default
            return

        kmeans = KMeans(n_clusters=2, init="k-means++", n_init=10, random_state=0)
        kmeans.fit(player_colors)

        self.kmeans = kmeans

        self.team_colors[1] = kmeans.cluster_centers_[0]
        self.team_colors[2] = kmeans.cluster_centers_[1]

    def get_player_team(self, frame, player_bbox, player_id):
        """
        Predict which team a player belongs to based on jersey color.

        Caches results per player_id so the same player isn't re-clustered
        every frame.

        Args:
            frame (np.ndarray): Video frame.
            player_bbox (list): Player bounding box [x1,y1,x2,y2].
            player_id (int): Persistent track ID.

        Returns:
            int: Team ID (1 or 2).
        """
        if player_id in self.player_team_dict:
            return self.player_team_dict[player_id]

        if self.kmeans is None:
            team_id = (player_id % 2) + 1
            self.player_team_dict[player_id] = team_id
            return team_id

        player_color = self.get_player_color(frame, player_bbox)

        team_id = self.kmeans.predict(player_color.reshape(1, -1))[0]
        team_id += 1  # Make 1-indexed (team 1 or team 2)

        # Override known goalkeeper ID if needed
        # (Goalkeeper typically has a unique jersey, may be misassigned)
        self.player_team_dict[player_id] = team_id

        return team_id

