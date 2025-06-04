"""EX07: Object Detection."""
from __future__ import annotations
import numpy as np
from typing import Optional, List, Tuple


class Robot:
    """Turtlebot robot."""

    def __init__(self, robot: object) -> None:
        """Class initializer.

        Args:
            robot (object): An instance of a Turtlebot-like robot interface.
        """
        self.robot = robot
        self.image: Optional[np.ndarray] = None
        self.bounding_boxes: List[Tuple[int, int, int, int]] = []

    def get_cube_objects(self) -> Optional[List[Tuple[int, int, int, int]]]:
        """Return the bounding boxes for detected objects.

        Returns:
            [(x_min, x_max, y_min, y_max), ...]: A list of bounding boxes for
            detected cubes. Returns None if no cubes are detected.
        """
        return self.bounding_boxes

    def sense(self) -> None:
        """Gather sensor data and detect blue cubes."""
        self.image = self.robot.get_camera_rgb_image()
        if self.image is None:
            self.bounding_boxes = None
            return

        np_image = np.asarray(self.image, dtype=np.uint8)
        if np_image.ndim != 3 or np_image.shape[2] < 3:
            self.bounding_boxes = None
            return

        # Extract color channels
        blue_channel = np_image[:, :, 0].astype(int)
        green_channel = np_image[:, :, 1].astype(int)
        red_channel = np_image[:, :, 2].astype(int)

        # Adjusted Blue Thresholding to reduce false positives
        blue_threshold = 50  # Increased threshold to avoid detecting non-blue objects
        mask = (blue_channel > green_channel + blue_threshold) & (blue_channel > red_channel + blue_threshold)

        # Find connected components
        labeled_mask, label_count = self._find_blobs(mask)
        self.bounding_boxes = []

        for i in range(1, label_count + 1):
            y_indices, x_indices = np.where(labeled_mask == i)

            if x_indices.size < 500 or y_indices.size < 500:
                continue  # Ignore small objects

            x_min, x_max = int(x_indices.min()), int(x_indices.max())
            y_min, y_max = int(y_indices.min()), int(y_indices.max())
            width = x_max - x_min + 1
            height = y_max - y_min + 1
            aspect_ratio = width / height

            if not (0.7 < aspect_ratio < 1.5):
                continue  # Ensure it's more cube-like

            blue_pixels_count = np.sum(mask[y_min:y_max + 1, x_min:x_max + 1])
            total_pixel_count = width * height
            color_ratio = blue_pixels_count / total_pixel_count

            if color_ratio < 0.8:  # Stricter color ratio check
                continue

            self.bounding_boxes.append((x_min, x_max, y_min, y_max))

        if not self.bounding_boxes:
            self.bounding_boxes = None

    def update_cube_objects(self) -> None:
        """Update detected blue cubes based on the camera image."""
        if self.image is None:
            self.bounding_boxes = None  # Set to None when no image is available
            return

        np_image = np.asarray(self.image, dtype=np.uint8)
        if np_image.ndim != 3 or np_image.shape[2] < 3:
            self.bounding_boxes = None  # Set to None for invalid images
            return

        # Extract color channels
        blue_channel = np_image[:, :, 0]
        green_channel = np_image[:, :, 1]
        red_channel = np_image[:, :, 2]

        # Create a mask for blue objects
        threshold = 30  # Adjust this threshold as needed
        mask = (blue_channel > green_channel + threshold) & (blue_channel > red_channel + threshold)

        # Debug: Print the mask to see what is being detected
        print("Mask shape:", mask.shape)
        print("Mask sum (blue pixels):", np.sum(mask))

        # Find connected components (blobs) in the mask
        labeled_mask, label_count = self._find_blobs(mask)

        self.bounding_boxes = []  # Initialize as an empty list

        for i in range(1, label_count + 1):
            y_indices, x_indices = np.where(labeled_mask == i)

            # Filter out small noise regions
            if x_indices.size < 500 or y_indices.size < 500:
                continue

            # Compute bounding box
            x_min, x_max = int(x_indices.min()), int(x_indices.max())
            y_min, y_max = int(y_indices.min()), int(y_indices.max())

            width = x_max - x_min + 1
            height = y_max - y_min + 1
            aspect_ratio = width / height

            # Check aspect ratio to filter out non-cube shapes
            if not (0.5 < aspect_ratio < 2.0):
                continue

            # Check the color ratio within the bounding box
            blue_pixels_count = np.sum(mask[y_min:y_max + 1, x_min:x_max + 1])
            total_pixel_count = width * height
            color_ratio = blue_pixels_count / total_pixel_count

            if color_ratio < 0.7:  # If less than 70% of pixels are blue, discard
                continue

            # Log the bounding box details for debugging
            print(f"Detected bounding box: {(x_min, x_max, y_min, y_max)}")
            print(f"Aspect ratio: {aspect_ratio}, Color ratio: {color_ratio}")

            # Append the bounding box to the list if all conditions are met
            self.bounding_boxes.append((x_min, x_max, y_min, y_max))

        # Set to None if no bounding boxes are detected
        if not self.bounding_boxes:
            self.bounding_boxes = None

    def _find_blobs(self, mask: np.ndarray) -> Tuple[np.ndarray, int]:
        """Find connected components (blobs) in a binary mask.

        Args:
            mask (np.ndarray): Binary mask where 1 represents the object.

        Returns:
            Tuple[np.ndarray, int]: Labeled mask and the number of labels.
        """
        height, width = mask.shape
        labeled_mask = np.zeros_like(mask, dtype=np.uint32)
        label_id = 1
        to_visit = []
        neighbors = ((-1, 0), (1, 0), (0, -1), (0, 1))

        for y, x in np.argwhere(mask):
            if labeled_mask[y, x] == 0:
                labeled_mask[y, x] = label_id
                to_visit.append((y, x))

                while to_visit:
                    current_y, current_x = to_visit.pop()
                    for dy, dx in neighbors:
                        ny, nx = current_y + dy, current_x + dx
                        if 0 <= ny < height and 0 <= nx < width:
                            if mask[ny, nx] and labeled_mask[ny, nx] == 0:
                                labeled_mask[ny, nx] = label_id
                                to_visit.append((ny, nx))
                label_id += 1

        return labeled_mask, label_id - 1

    def plan(self) -> None:
        """Plan the robot's actions.

        Process the data collected during sensing and decide the next course
        of action for the robot.
        """
        pass

    def act(self) -> None:
        """Execute planned actions.

        Perform the actions decided in the planning step, such as moving or
        interacting with the environment.
        """
        pass

    def spin(self) -> None:
        """Spin the robot.

        This is the main loop where the robot performs its sense-plan-act cycle.
        """
        self.sense()
        self.plan()
        self.act()
