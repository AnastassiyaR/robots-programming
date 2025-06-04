"""EX06: Colors."""
from __future__ import annotations
from typing import Union, List, Tuple
import numpy as np
from scipy.ndimage import label


class Robot:
    """Turtlebot robot."""

    def __init__(self, robot: object) -> None:
        """Class initializer.

        Args:
          robot (object): An instance of a Turtlebot-like robot interface.
        """
        self.robot = robot
        self.image = None
        self.bounding_boxes = None
        self.object_locations = None

    def get_camera_rgb_image(self):
        """Retrieve the latest image from the robot's camera."""
        return self.robot.get_camera_rgb_image()

    def get_object_location_list(self) -> Union[List[List[float]], None]:
        """Calculate the coordinates for detected object center and corresponding angle."""
        return self.object_locations

    def get_object_bounding_box_list(self) -> Union[List[Tuple[int, int, int, int]], None]:
        """Calculate the bounding box for any detected blue object."""
        return self.bounding_boxes

    def sense(self) -> None:
        """Gather sensor data."""
        self.image = self.get_camera_rgb_image()
        if self.image is None:
            self.bounding_boxes = None
            self.object_locations = None
            return

        # Convert image to a NumPy array
        np_image = np.asarray(self.image, dtype=np.uint8)
        if np_image.ndim != 3 or np_image.shape[2] < 3:
            self.bounding_boxes = None
            self.object_locations = None
            return

        # Extract RGB channels
        blue = np_image[:, :, 2]
        red = np_image[:, :, 0]
        green = np_image[:, :, 1]

        # Define blue color threshold
        blue_mask = (blue > 120) & (blue > red + 40) & (blue > green + 40)

        # Label connected regions in the blue mask
        labeled_array, num_features = label(blue_mask)

        # Process each detected region
        self.bounding_boxes = []
        self.object_locations = []

        for i in range(1, num_features + 1):
            y_indices, x_indices = np.where(labeled_array == i)

            # Skip small regions (likely noise)
            if x_indices.size < 200 or y_indices.size < 200:
                continue

            # Compute bounding box
            x_min, x_max = int(x_indices.min()), int(x_indices.max())
            y_min, y_max = int(y_indices.min()), int(y_indices.max())
            self.bounding_boxes.append((x_min, x_max, y_min, y_max))

            # Compute centroid
            cx, cy = (x_min + x_max) / 2, (y_min + y_max) / 2

            # Compute angle relative to the robot
            img_width = np_image.shape[1]
            fov = 60  # Assuming a 60-degree camera field of view
            angle = ((cx / img_width) - 0.5) * fov

            self.object_locations.append([cx, cy, angle])

        # Handle case where no objects are detected
        if not self.bounding_boxes:
            self.bounding_boxes = None
            self.object_locations = None

    def plan(self) -> None:
        """Plan the robot's actions."""
        pass  # No movement required

    def act(self) -> None:
        """Execute planned actions."""
        pass  # No movement required

    def spin(self) -> None:
        """Spin the robot."""
        self.sense()
        self.plan()
        self.act()