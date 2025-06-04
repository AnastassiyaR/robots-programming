"""EX03: Sensors."""

import math
import numpy as np


class Robot:
    """Turtlebot robot."""

    def __init__(self, robot: object) -> None:
        """Class initializer.

        Args: robot (object): An instance of a Turtlebot-like robot interface.
        """
        self.robot = robot
        self.objects = None

    def sense(self) -> None:
        """Gather sensor data."""
        lidar_data = self.robot.get_lidar_range_list()

        if not lidar_data:
            self.objects = None
            return

        lidar_points = len(lidar_data)
        self.objects = []

        # Filter and process lidar data
        lidar_data = self.filter_lidar_data(lidar_data)

        # Detect objects
        self.objects = self.detect_objects(lidar_data, lidar_points)

        if not self.objects:
            self.objects = None

    def filter_lidar_data(self, lidar_data: list) -> list:
        """Filter lidar data by removing invalid points (inf)."""
        return [i if i != math.inf else None for i in lidar_data]

    def detect_objects(self, lidar_data: list, lidar_points: int) -> list:
        """Detect objects based on lidar data."""
        objects = []
        current_object = []
        checked = False

        for i in range(1, len(lidar_data) - 1):
            angle = (i * 360 / lidar_points)
            angle_rad = np.radians(angle)

            if self._is_invalid_point(i, lidar_data):
                if checked:
                    checked = False
                    current_object = []
                continue

            if not checked and abs(lidar_data[i] - lidar_data[i - 1]) > 0.3:
                if lidar_data[i] < lidar_data[i - 1]:
                    checked = True
                    current_object = [(lidar_data[i], angle_rad)]

            elif checked and abs(lidar_data[i] - lidar_data[i + 1]) > 0.3:
                if lidar_data[i] < lidar_data[i + 1]:
                    current_object.append((lidar_data[i], angle_rad))
                    min_distance, min_angle = min(current_object, key=lambda x: x[0])
                    objects.append((min_distance, min_angle))
                    checked = False
                    current_object = []

            elif checked:
                current_object.append((lidar_data[i], angle_rad))

        return objects

    def _is_invalid_point(self, index: int, lidar_data: list) -> bool:
        """Check if a lidar data point is invalid (inf or adjacent invalid points)."""
        return (lidar_data[index] is None or lidar_data[index - 1] is None or lidar_data[index + 1] is None)

    def get_objects_range_list(self) -> list:
        """Return the detected objects range list."""
        if self.objects is None:
            return None
        return self.objects

    def plan(self) -> None:
        """Plan the robot's actions."""
        pass

    def act(self) -> None:
        """Execute planned actions."""
        pass

    def spin(self) -> None:
        """Spin the robot."""
        self.sense()
        self.plan()
        self.act()
