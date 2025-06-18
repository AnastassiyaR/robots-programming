"""S3."""

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

        self.target_point = None

    def get_triangle_vertex_coordinates(self) -> tuple | None:
        """Return the triangle corner coordinates."""
        # Logic to calculate the triangle's vertices based on lidar data
        detected_objects = []
        start_index = None

        threshold = 0.1
        min_object_size = 1

        if self.data is not None:
            for i in range(1, len(self.data)):
                if self.data[i] == float('inf') or self.data[i - 1] == float('inf'):
                    start_index = None
                    continue

                delta = self.data[i] - self.data[i - 1]

                if start_index is None and abs(delta) > threshold and delta < 0:
                    start_index = i

                elif start_index is not None and abs(delta) > threshold and delta > 0:
                    end_index = i - 1

                    if abs(end_index - start_index) >= min_object_size:
                        object_values = self.data[start_index:end_index]
                        # minimum distance in the object segment
                        min_distance = np.min(object_values)
                        # index of the minimum value within the object
                        min_index = np.argmin(object_values)
                        center_index = start_index + min_index

                        angle = (center_index / len(self.data)) * (2 * np.pi) + (np.pi * 2)
                        detected_objects.append((min_distance, angle))

                    start_index = None

        if len(detected_objects) != 2:
            return None

        # Compute triangle vertex coordinates
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

        x1, y1 = objects_coordinates_world[0]
        x2, y2 = objects_coordinates_world[1]

        xm = (x1 + x2) / 2
        ym = (y1 + y2) / 2

        dx = (math.sqrt(3) / 2) * (y2 - y1)
        dy = (math.sqrt(3) / 2) * (x2 - x1)

        c_1 = (xm + dx, ym - dy)
        c_2 = (xm - dx, ym + dy)
        return c_1, c_2

    def get_robot_pose(self) -> tuple:
        """Return the current robot pose."""
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
        self.theta = (self.theta + np.pi) % (2 * np.pi) - np.pi

        self.robot_x += linear_velocity * math.cos(self.theta) * delta_time
        self.robot_y += linear_velocity * math.sin(self.theta) * delta_time

        self.previous_time = self.current_time
        self.previous_left_ticks = left_ticks
        self.previous_right_ticks = right_ticks

        return self.robot_x, self.robot_y, self.theta

    def sense(self) -> None:
        """Gather sensor data."""
        self.data = self.robot.get_lidar_range_list()
        self.current_time = self.robot.get_time()

        if self.start_orientation is None:
            self.start_orientation = self.robot.get_orientation()
        self.theta = self.robot.get_orientation() - self.start_orientation

    def plan(self) -> None:
        """Plan the robot's actions."""
        if self.target_point is not None:
            return

        triangle_points = self.get_triangle_vertex_coordinates()

        if triangle_points is None:
            print("No valid triangle points detected.")
            return

        c_1, c_2 = triangle_points

        if self.is_point_within_simulation_bounds(c_1):
            self.target_point = c_1
        elif self.is_point_within_simulation_bounds(c_2):
            self.target_point = c_2
        else:
            print("No valid target point within simulation bounds.")
            return

        print(f"Target point: {self.target_point}")

    def is_point_within_simulation_bounds(self, point: tuple) -> bool:
        """Check if the point is within the simulation bounds."""
        x, y = point
        SIMULATION_X_MIN, SIMULATION_X_MAX = -5.0, 5.0
        SIMULATION_Y_MIN, SIMULATION_Y_MAX = -5.0, 5.0

        return (SIMULATION_X_MIN <= x <= SIMULATION_X_MAX and SIMULATION_Y_MIN <= y <= SIMULATION_Y_MAX)

    def act(self) -> None:
        """Execute planned actions."""
        if self.target_point is None:
            print("No target point set.")
            return

        current_x, current_y, current_theta = self.get_robot_pose()
        target_x, target_y = self.target_point

        dx = target_x - current_x
        dy = target_y - current_y

        target_angle = math.atan2(dy, dx)

        Kp_angle = 1.0
        angular_velocity = Kp_angle * (target_angle - current_theta)

        angular_velocity = max(min(angular_velocity, 1.0), -1.0)

        distance = math.hypot(dx, dy)
        Kp_distance = 0.5
        linear_velocity = Kp_distance * distance

        linear_velocity = max(min(linear_velocity, 1.0), -1.0)

        left_speed = linear_velocity - angular_velocity * self.WHEEL_BASE / 2
        right_speed = linear_velocity + angular_velocity * self.WHEEL_BASE / 2

        if distance < 0.05:
            self.robot.set_left_motor_velocity(0)
            self.robot.set_right_motor_velocity(0)
            print(f"Robot reached target point: {self.target_point}")
            return

        self.robot.set_left_motor_velocity(left_speed)
        self.robot.set_right_motor_velocity(right_speed)

    def spin(self) -> None:
        """Spin the robot."""
        self.sense()
        self.plan()
        self.act()
        print(f"Orientation: {self.get_robot_pose()}")
        print(f"Target point: {self.target_point if self.target_point else 'Not set'}")
