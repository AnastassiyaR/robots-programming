"""C2."""

from __future__ import annotations

import math

import numpy as np
import scipy.ndimage
from PIL import Image


class Robot:
    """Turtlebot robot."""

    def __init__(self, robot: object) -> None:
        """Class initializer.

        Args:
            robot (object): An instance of a Turtlebot-like robot interface.
        """
        self.robot = robot
        self.data = None
        self.image = None
        self.orientation = 0.0
        self.previous_orientation = 0.0
        self.total_rotation = 0.0
        self.image_width = 1280
        self.image_height = 800
        self.field_of_view = self.robot.get_camera_field_of_view()
        self.bounding_boxes = None
        self.object_locations = None
        self.detected_objects = []

        self.state = None
        self.target = None
        self.available_colors = ["blue", "red", "yellow"]
        self.target_color = "blue"
        self.distance_to_target = 0
        self.left_speed = 0.0
        self.right_speed = 0.0
        self.torque_left = 0.0
        self.torque_right = 0.0

        self.STOP_DISTANCE = 0.3  # in meters
        self.OBJECT_THRESHOLD = 0.3  # in meters
        self.ANGULAR_DIFFERENCE = 3  # angular distance between objects in degrees
        self.MAX_TORQUE = 0.1  # maximum torque to apply to the wheel
        self.TICKS_PER_ROTATION = 508.8  # Constants for encoder ticks to velocity conversion
        self.MIN_OBJECT_DISTANCE = 0.4
        self.MAX_OBJECT_DISTANCE = 3.5
        self.COLOR_THRESHOLD = 40
        self.MIN_AREA_SIZE = 100
        self.IMAGE_BASED_DISTANCE_THRESHOLD = 200  # at the stop distance cylynder will be this wide (in pixels)
        self.IMAGE_CUT_SIZE = 400

        self.kp = 0.08  # Proportional gain
        self.ki = 0.33  # Integral gain
        self.kd = 0.0  # Derivative gain

        self.image_kp = 1

        self.previous_error_left = 0
        self.integral_left = 0
        self.previous_left_ticks = 0

        self.previous_error_right = 0
        self.integral_right = 0
        self.previous_right_ticks = 0

        self.current_time = self.robot.get_time() or 0.0
        self.previous_time = self.current_time

    def set_target_speeds(self, left_target: float, right_target: float) -> None:
        """Set the target speeds for the robot's wheels.

        Args:
            left_target (float): Target speed for the left wheel.
            right_target (float): Target speed for the right wheel.
        """
        self.left_speed = left_target
        self.right_speed = right_target

    def ticks_to_velocity(self, ticks, delta_time):
        """Convert encoder ticks to velocity in radians per second."""
        if delta_time <= 0:
            return 0

        ticks_per_second = ticks / delta_time
        rotations_per_second = ticks_per_second / self.TICKS_PER_ROTATION
        radians_per_second = rotations_per_second * 2 * math.pi

        return radians_per_second

    def ticks_to_distance(self, ticks):
        """Convert encoder ticks to travelled distance."""
        wheel_circumference = math.pi * self.robot.WHEEL_DIAMETER
        return ticks / self.TICKS_PER_ROTATION * wheel_circumference

    def update_both_wheel_speeds(self) -> None:
        """Adjust speeds of both wheels."""
        delta_time = self.current_time - self.previous_time

        current_left_ticks = self.robot.get_left_motor_encoder_ticks()
        current_right_ticks = self.robot.get_right_motor_encoder_ticks()

        delta_left = current_left_ticks - self.previous_left_ticks
        delta_right = current_right_ticks - self.previous_right_ticks
        self.distance_to_target -= self.ticks_to_distance((delta_left + delta_right) / 2)

        left_angular_velocity = self.ticks_to_velocity(delta_left, delta_time)
        right_angular_velocity = self.ticks_to_velocity(delta_right, delta_time)

        self.previous_left_ticks = current_left_ticks
        self.previous_right_ticks = current_right_ticks

        error_left = self.left_speed - left_angular_velocity
        derivative_left = (error_left - self.previous_error_left) / delta_time if delta_time > 0 else 0
        self.integral_left += error_left * delta_time
        self.integral_left = max(min(self.integral_left, 0.1), -0.1)
        self.torque_left = (self.kp * error_left + self.ki * self.integral_left + self.kd * derivative_left)
        self.previous_error_left = error_left

        error_right = self.right_speed - right_angular_velocity
        derivative_right = (error_right - self.previous_error_right) / delta_time if delta_time > 0 else 0
        self.integral_right += error_right * delta_time
        self.integral_right = max(min(self.integral_right, 0.1), -0.1)
        self.torque_right = (self.kp * error_right + self.ki * self.integral_right + self.kd * derivative_right)
        self.previous_error_right = error_right

        self.torque_left = max(min(self.torque_left, self.MAX_TORQUE), -self.MAX_TORQUE)
        self.torque_right = max(min(self.torque_right, self.MAX_TORQUE), -self.MAX_TORQUE)

    def get_pid_corrected_left_wheel_speed(self) -> float:
        """Return the corrected left wheel speed."""
        return self.torque_left

    def get_pid_corrected_right_wheel_speed(self) -> float:
        """Return the corrected right wheel speed."""
        return self.torque_right

    def get_object_location_list(self) -> list | None:
        """Calculate the coordinates for detected object center and corresponding angle."""
        return self.object_locations

    def get_object_bounding_box_list(self) -> list | None:
        """Calculate the bounding box for any detected color object."""
        return self.bounding_boxes

    def find_object_locations(self):
        """Find objects location with centroid and angle."""
        self.object_locations = None

        if not self.bounding_boxes:
            return

        new_object_locations = []

        for one_box in self.bounding_boxes:
            x_min, x_max, y_min, y_max = one_box

            cx, cy = (x_min + x_max) / 2, (y_min + y_max) / 2

            angle = ((cx / self.image_width) - 0.5) * self.field_of_view
            new_object_locations.append([cx, cy, angle])

        self.object_locations = new_object_locations if len(new_object_locations) > 0 else None

    def find_object_bounding_boxes(self, target_color: str):
        """Find bounding borders of all objects of given color - blue, red or yellow."""
        self.bounding_boxes = None

        if self.image is None:
            return

        np_image = np.asarray(self.image, dtype=np.uint8)
        if np_image.ndim != 3 or np_image.shape[2] < 3:
            return

        if self.state == "approach":
            np_image = np_image[self.IMAGE_CUT_SIZE:, self.IMAGE_CUT_SIZE:self.image_width - self.IMAGE_CUT_SIZE]
        else:
            np_image = np_image[self.IMAGE_CUT_SIZE:, :]

        blue_channel, green_channel, red_channel = np_image[:, :, 0], np_image[:, :, 1], np_image[:, :, 2]

        blue_mask = ((blue_channel > green_channel + self.COLOR_THRESHOLD)
                     & (blue_channel > red_channel + self.COLOR_THRESHOLD))
        red_mask = ((red_channel > green_channel + self.COLOR_THRESHOLD)
                    & (red_channel > blue_channel + self.COLOR_THRESHOLD))
        yellow_mask = (red_channel > 100) & \
                      (red_channel < 160) & \
                      (green_channel > 100) & \
                      (green_channel < 160) & \
                      (blue_channel < 65)

        mask = blue_mask if target_color == "blue" else red_mask if target_color == "red" else yellow_mask
        labeled_mask, label_count = scipy.ndimage.label(mask)

        new_bounding_boxes = []

        for i in range(1, label_count + 1):
            y_indices, x_indices = np.where(labeled_mask == i)

            if x_indices.size < self.MIN_AREA_SIZE or y_indices.size < self.MIN_AREA_SIZE:
                continue

            x_min, x_max = int(x_indices.min()), int(x_indices.max())
            y_min, y_max = int(y_indices.min()), int(y_indices.max())
            if self.state == "approach":
                new_bounding_boxes.append((x_min + self.IMAGE_CUT_SIZE, x_max + self.IMAGE_CUT_SIZE,
                                           y_min + self.IMAGE_CUT_SIZE, y_max + self.IMAGE_CUT_SIZE))
            else:
                new_bounding_boxes.append((x_min, x_max, y_min + self.IMAGE_CUT_SIZE, y_max + self.IMAGE_CUT_SIZE))

        self.bounding_boxes = new_bounding_boxes if len(new_bounding_boxes) > 0 else None

    def trace_boxes(self, produce_images):
        """Test with image."""
        if self.bounding_boxes:
            print(self.bounding_boxes)
            print(self.object_locations)
            if produce_images:
                # Save received frame as a PNG image on disk.
                img = Image.frombytes("RGBA", (self.image_width, self.image_height), self.image, "raw", "BGRA")
                img.save(f"C2_frame_{self.current_time}.png")

    def is_within_range(self):
        """Check the object is within a stop range."""
        # use only image-based distance control
        if self.bounding_boxes:
            x_min, x_max, y_min, y_max = self.bounding_boxes[0]
            if (x_max - x_min) > self.IMAGE_BASED_DISTANCE_THRESHOLD:
                return True

        for i in range(400, 560):
            if self.data[i] < self.STOP_DISTANCE:
                return True
        return False

    def align_to_image_center(self) -> None:
        """Adjust speeds of both wheels to center the object."""
        if self.target:
            error = self.target
            self.left_speed += error * self.image_kp
            self.right_speed -= error * self.image_kp

            self.left_speed = max(min(self.left_speed, 1), -1)
            self.right_speed = max(min(self.right_speed, 1), -1)

    def calculate_delta_rotation(self, orientation, previous_orientation) -> float:
        """Calculate the total rotation of the robot."""
        delta = orientation - previous_orientation
        if delta > math.pi:
            delta -= 2 * math.pi
        elif delta < -math.pi:
            delta += 2 * math.pi
        return delta

    def find_closest_object_angle(self, data_snapshot):
        """Find the largest by square object from saved data and return its orientation angle."""
        all_objects = []

        for instance in data_snapshot:
            objects_boxes, locations, robot_orientation = instance

            for index, each_box in enumerate(objects_boxes):
                x_min, x_max, y_min, y_max = each_box
                box_square = (x_max - x_min + 1) * (y_max - y_min + 1)

                all_objects.append((box_square, locations[index][2], robot_orientation))

        closest_object = max(all_objects, key=lambda x: x[0])
        return closest_object[2] - closest_object[1]

    def _handle_init_state(self) -> None:
        """Handle initialization state."""
        self.state = "searching"
        self.detected_objects = []
        self.total_rotation = 0.0
        self.target = None
        print(f"🤖 Searching the {self.target_color} color objects...")

    def _handle_searching_state(self) -> None:
        """Handle searching for objects state."""
        self.set_target_speeds(-0.5, 0.5)

        if math.floor(math.degrees(self.total_rotation)) % 30 == 0:
            if self.bounding_boxes:
                self.detected_objects.append((self.bounding_boxes, self.object_locations, self.orientation))

        if abs(self.total_rotation) >= 2 * math.pi:
            print(f"Search is over, selecting the closest {self.target_color} target")

            if self.detected_objects:
                self.target = self.find_closest_object_angle(self.detected_objects)
                self.state = "selecting"
            else:
                self._rotate_target_color()
                self.state = "init"

    def _handle_selecting_state(self) -> None:
        """Handle selecting and aligning with target object."""
        # Rotate to align with target
        self.set_target_speeds(-0.5, 0.5)

        # Check if aligned
        if abs(self.target - self.orientation) < 0.01:
            self.set_target_speeds(0, 0)
            self.distance_to_target = min(self.data[470:490])
            self.state = "approach"

    def _handle_approach_state(self) -> None:
        """Handle approaching target object."""
        # Move forward
        self.set_target_speeds(0.5, 0.5)

        # Update target information
        self._update_target_tracking()

        if self.is_within_range():
            self.trace_boxes(False)
            print(f"⭐ Found THAT {self.target_color} target object!")
            self.set_target_speeds(0, 0)
            self._rotate_target_color()
            self.state = "init"

    def _update_target_tracking(self) -> None:
        """Update tracking information for the current target."""
        self.detected_objects = []
        self.detected_objects.append((self.bounding_boxes, self.object_locations, self.orientation))

        self.target = min(self.object_locations, key=lambda x: abs(x[2]))[2]
        self.align_to_image_center()

    def _rotate_target_color(self) -> None:
        """Rotate to the next target color in the available colors list."""
        current_color_index = self.available_colors.index(self.target_color)
        self.target_color = self.available_colors[(current_color_index + 1) % len(self.available_colors)]
        print(f"🌈 The color was switched from to {self.target_color}")
    def sense(self) -> None:
        """Gather sensor data."""
        self.current_time = self.robot.get_time()
        self.orientation = self.robot.get_orientation()
        self.data = self.robot.get_lidar_range_list()
        self.total_rotation += self.calculate_delta_rotation(self.orientation, self.previous_orientation)
        self.previous_orientation = self.orientation
        self.image = self.robot.get_camera_rgb_image()

        do_scan = False
        if (self.state == "selecting" or self.state == "approach") and math.floor(
                math.degrees(self.total_rotation)) % 2 == 0:
            do_scan = True
        elif self.state == "searching" and math.floor(math.degrees(self.total_rotation)) % 5 == 0:
            do_scan = True

        if do_scan:
            self.find_object_bounding_boxes(self.target_color)
            self.find_object_locations()

        if not self.state:
            self.state = "init"

    def plan(self) -> None:
        """Plan the robot's actions based on current state."""
        if not self.state:
            self.state = "init"

        if self.state == "init":
            self._handle_init_state()
        elif self.state == "searching":
            self._handle_searching_state()
        elif self.state == "selecting":
            self._handle_selecting_state()
        elif self.state == "approach":
            self._handle_approach_state()
        else:
            # Default fallback
            self.set_target_speeds(0, 0)
            self.state = "stopped"

        # Common operations regardless of state
        self.update_both_wheel_speeds()
        self.previous_time = self.robot.get_time()

    def act(self) -> None:
        """Execute planned actions."""
        if self.robot.get_realistic():
            self.robot.set_left_motor_torque(self.get_pid_corrected_left_wheel_speed())
            self.robot.set_right_motor_torque(self.get_pid_corrected_right_wheel_speed())
        else:
            self.robot.set_left_motor_velocity(self.left_speed)
            self.robot.set_right_motor_velocity(self.right_speed)

    def spin(self) -> None:
        """Spin the robot."""
        self.sense()
        self.plan()
        self.act()
