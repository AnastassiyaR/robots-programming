"""EX05: Triangle Forming."""
from __future__ import annotations
import math
import numpy as np


class Robot:
    """Turtlebot robot."""

    def __init__(self, robot: object) -> None:
        """Class initializer.

        Args:
            robot (object): An instance of a Turtlebot-like robot interface.
        """
        self.robot = robot

        self.WHEEL_BASE = 0.233
        self.TRACK_WIDTH = self.WHEEL_BASE
        self.cylinders = None
        self.TICKS_PER_RADIANS = 508.8 / (2 * math.pi)
        self.WHEEL_RADIUS = 0.03575

        self.previous_x = 0.0
        self.previous_y = 0.0

        self.start_orientation = None
        self.theta = 0.0

        self.left_ticks = 0
        self.right_ticks = 0

        self.previous_left_ticks = 0
        self.previous_right_ticks = 0
        self.previous_time = 0
        self.current_time = 0

        self.data = None
        self.robot_x = 0.0
        self.robot_y = 0.0

    def get_triangle_vertex_coordinates(self) -> tuple | None:
        """Return the triangle corner coordinates.

        Based on lidar range list and current robot position, calculate the world
        position of the equilateral triangle corner, and return coordinates of
        x, y.

        Logic:
        - This method uses lidar data to find the two objects that form the base of
          triangle (vertex)
        - Based on the found objects transform them to world frame coordinates and
          calculate triangle corner coordinates (there are two solutions since the
        equilateral triangle can be formed on both sides of the line connecting
        the two objects).
        - The robot's orientation and position are used to compute the actual world
          coordinates of the corner.

        Returns:
            A tuple of two tuples representing the (x, y) world coordinates of the
        two possible equilateral triangle's corners.
        (i.e., ((x1, y1), (x2, y2)))
            Returns `None` if no valid triangle corner can be detected.
        """
        detected_objects = []
        start = None

        threshold = 0.1
        min_object_size = 1

        if self.data is not None:

            for i in range(1, len(self.data)):
                if self.data[i] == float('inf') or self.data[i - 1] == float('inf'):
                    start = None
                    continue

                delta = self.data[i] - self.data[i - 1]

                if start is None and abs(delta) > threshold and delta < 0:
                    start = i

                elif start is not None and abs(delta) > threshold and delta > 0:
                    end_index = i - 1

                    if abs(end_index - start) >= min_object_size:
                        object_values = self.data[start:end_index]
                        min_distance = np.min(object_values)
                        min_index = np.argmin(object_values)
                        center_index = start + min_index

                        angle = (center_index / len(self.data)) * (2 * np.pi)
                        detected_objects.append((min_distance, angle))

                    start = None

        objects_coordinates_robot = []
        objects_coordinates_world = []

        for object in detected_objects:
            x_coordinate_relative_to_the_robot_position = -(object[0] * math.sin(object[1]))
            y_coordinate_relative_to_the_robot_position = -(object[0] * math.cos(object[1]))
            objects_coordinates_robot.append(
                (x_coordinate_relative_to_the_robot_position, y_coordinate_relative_to_the_robot_position))

        for object in objects_coordinates_robot:
            x_coordinate_relative_to_the_world = (self.robot_x + object[0]
                                                  * math.cos(self.theta) - object[1] * math.sin(self.theta))
            y_coordinate_relative_to_the_world = (self.robot_y + object[0]
                                                  * math.sin(self.theta) + object[1] * math.cos(self.theta))
            objects_coordinates_world.append((x_coordinate_relative_to_the_world, y_coordinate_relative_to_the_world))

        if len(objects_coordinates_world) < 2:
            return None

        x1, y1 = objects_coordinates_world[0]
        x2, y2 = objects_coordinates_world[1]

        xm = (x1 + x2) / 2
        ym = (y1 + y2) / 2

        dx = (math.sqrt(3) / 2) * (y2 - y1)
        dy = (math.sqrt(3) / 2) * (x2 - x1)

        c_1 = (xm + dx, ym - dy)
        c_2 = (xm - dx, ym + dy)

        return (float(c_1[0]), float(c_1[1])), (float(c_2[0]), float(c_2[1]))

    def get_robot_pose(self) -> tuple:
        """Return the current robot pose.

        Return the robot's pose as a tuple, based on wheel encoders and IMU.

        Returns:
            A tuple representing the (x, y, theta) robot's pose. Theta is the
            angle between robot's starting direction and its current direction
            (in radians, with -pi < theta <= pi).
        """
        delta_time = self.current_time - self.previous_time
        if delta_time <= 0:
            return self.robot_x, self.robot_y, self.theta

        left_ticks = self.robot.get_left_motor_encoder_ticks()
        right_ticks = self.robot.get_right_motor_encoder_ticks()

        delta_left_ticks = left_ticks - self.previous_left_ticks
        delta_right_ticks = right_ticks - self.previous_right_ticks

        left_velocity = (delta_left_ticks / self.TICKS_PER_RADIANS) / delta_time
        right_velocity = (delta_right_ticks / self.TICKS_PER_RADIANS) / delta_time

        linear_velocity = (self.WHEEL_RADIUS / 2) * (left_velocity + right_velocity)
        angular_velocity = (self.WHEEL_RADIUS / self.WHEEL_BASE) * (right_velocity - left_velocity)

        self.theta += angular_velocity * delta_time
        self.theta = (self.theta + np.pi) % (2 * np.pi) - np.pi  # Normalize angle

        self.robot_x += linear_velocity * math.cos(self.theta) * delta_time
        self.robot_y += linear_velocity * math.sin(self.theta) * delta_time

        self.previous_time = self.current_time
        self.previous_left_ticks = left_ticks
        self.previous_right_ticks = right_ticks

        return self.robot_x, self.robot_y, self.theta

    def sense(self):
        """Gather sensor data.

        Use the robot's sensors to collect data about its environment.
        This method updates internal state variables based on sensor readings.
        """
        self.data = self.robot.get_lidar_range_list()
        self.current_time = self.robot.get_time()
        self.left_ticks = self.robot.get_left_motor_encoder_ticks()
        self.right_ticks = self.robot.get_right_motor_encoder_ticks()

        if self.start_orientation is None:
            self.start_orientation = self.robot.get_orientation()
        self.theta = self.robot.get_orientation() - self.start_orientation

    def plan(self):
        """Plan the robot's actions.

        Process the data collected during sensing and decide the next course
        of action for the robot.
        """
        pass

    def act(self):
        """Execute planned actions.

        Perform the actions decided in the planning step, such as moving or
        interacting with the environment.
        """
        pass

    def spin(self):
        """Spin the robot.

        This is the main loop where the robot performs its sense-plan-act cycle.
        """
        self.sense()
        self.plan()
        self.act()
