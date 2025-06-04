"""C2."""
import math

import numpy as np
import scipy.ndimage


class Robot:
    """Turtlebot robot."""

    def __init__(self, robot: object) -> None:
        """Class initializer.

        Args:
            robot (object): An instance of a Turtlebot-like robot interface.
        """
        self.robot = robot
        self.speed = 20
        self.proximity_threshold = 0.25
        self.left_motor_speed = 0
        self.right_motor_speed = 0
        self.lidar_data = None
        self.image = None
        self.blue_object_center = None
        self.blue_object_size = None
        self.blue_object_angle = None
        self.lastTick = [0, 0]
        self.ticksPerSecond = [0, 0]
        self.prev_blue_object_size = None
        self.spin_counter = 0

        self.after_spin_failed = False
        self.going_straight = False
        self.straight_distance = 0.0

        self.is_spinning_180 = False
        self.backward_mode = False
        self.found_after_spin = False
        self.aligning_to_target_after_spin = False
        self.going_straight_after_spin = False

        self.final_mode = False
        self.final_phase = None
        self.final_ticks = 0
        self.final_distance_moved = 0.0

        self.done = False
        self.turn_speed = 0.4

        # Новые параметры для realistic-режима
        self.realistic_torque = 0.1
        self.realistic_speed_divider = 15
        self.realistic_min_speed = 0.05
        self.min_object_area = 100

        self.object_image_height_z = None
        self.colors = ["blue", "red", "yellow"]
        self.target_color = "blue"

        self.width = 0


    def sense(self) -> None:
        self.lidar_data = self.robot.get_lidar_range_list()
        self.image = self.robot.get_camera_rgb_image()
        self.detect_red_object()
        self.get_ticks_per_second()

    def _handle_init_state(self) -> None:
        """Handle initialization state."""
        self.state = "searching"
        self.detected_objects = []
        self.total_rotation = 0.0
        self.target = None
        print(f"Running in {'REALISTIC' if self.robot.get_realistic() else 'SIMULATION'} mode.")
        print(f"Search the {self.target_color} color objects...")

    def _handle_searching_state(self) -> None:
        """Handle searching for objects state."""
        self.set_target_speeds(-0.5, 0.5)

        if math.floor(math.degrees(self.total_rotation)) % 30 == 0:
            if self.bounding_boxes:
                self.detected_objects.append((self.bounding_boxes, self.object_locations, self.orientation))

        if abs(self.total_rotation) >= 2 * math.pi:
            if self.detected_objects:
                self.target = self.find_closest_object_angle(self.detected_objects)
                self.state = "selecting"
            else:
                print(f"No {self.target_color} object found. Switching to next color.")
                self._rotate_target_color()
                self.state = "init"

    def _handle_approach_state(self) -> None:
        """Handle approaching target object."""
        self.set_target_speeds(0.5, 0.5)
        self._update_target_tracking()

        if self.is_within_range():
            self.trace_boxes(False)
            print(f"Reached the {self.target_color} target object.")
            self.set_target_speeds(0, 0)

            # Переход к следующему цвету
            self._rotate_target_color()
            self.state = "init"

    def _rotate_target_color(self) -> None:
        """Rotate to the next target color in the available colors list."""
        current_color_index = self.colors.index(self.target_color)
        next_index = (current_color_index + 1) % len(self.colors)
        self.target_color = self.colors[next_index]
        print(f"Next target color: {self.target_color}")

    def detect_red_object(self) -> None:
        """Detect red cylindrical objects."""
        self.target_visible = False
        if self.image is None:
            self._clear_object_data()
            return

        np_image = np.asarray(self.image, dtype=np.uint8)
        if np_image.ndim != 3 or np_image.shape[2] < 3:
            self._clear_object_data()
            return

        if self.target_color == "blue":
            blue_channel = np_image[:, :, 0]
            green_channel = np_image[:, :, 1]
            red_channel = np_image[:, :, 2]

            threshold = 20
            mask = (blue_channel > green_channel + threshold) & (blue_channel > red_channel + threshold)

        elif self.target_color == "red":
            red_channel = np_image[:, :, 2]
            green_channel = np_image[:, :, 1]
            blue_channel = np_image[:, :, 0]

            threshold = 40
            mask = (red_channel > green_channel + threshold) & (red_channel > blue_channel + threshold)

        elif self.target_color == "yellow":
            red_channel = np_image[:, :, 2]
            green_channel = np_image[:, :, 1]
            blue_channel = np_image[:, :, 0]

            threshold = 40
            mask = (
                    (red_channel > threshold) &
                    (green_channel > threshold) &
                    (blue_channel < threshold) &
                    (np.abs(red_channel - green_channel) < 50)  # жёлтый должен быть сбалансирован между red и green
            )

        labeled_mask, num_features = scipy.ndimage.label(mask)

        best_candidate = None
        best_area = 0

        for i in range(1, num_features + 1):
            y_indices, x_indices = np.where(labeled_mask == i)
            if x_indices.size < 50 or y_indices.size < 50:
                continue

            x_min, x_max = int(x_indices.min()), int(x_indices.max())
            y_min, y_max = int(y_indices.min()), int(y_indices.max())
            width = x_max - x_min
            height = y_max - y_min

            self.width = width

            aspect_ratio = width / height if height > 0 else 0
            if aspect_ratio > 0.6:  # ограничение для цилиндра
                continue

            area = width * height
            if area > best_area:
                best_area = area
                best_candidate = (x_min, x_max, y_min, y_max)

        if best_candidate is None:
            self._clear_object_data()
            return

        x_min, x_max, y_min, y_max = best_candidate
        cx = (x_min + x_max) // 2
        cy = (y_min + y_max) // 2
        area = (x_max - x_min) * (y_max - y_min)

        self.blue_object_center = (cx, cy)
        self.target_visible = True
        self.object_image_height_z = y_max - y_min

        img_width = np_image.shape[1]
        img_height = np_image.shape[0]

        self.img_height = img_height
        self.max = y_max
        self.min = y_min
        raw_size = area
        smoothed_size = (
            raw_size if self.prev_blue_object_size is None
            else 0.7 * self.prev_blue_object_size + 0.3 * raw_size
        )
        self.prev_blue_object_size = smoothed_size
        self.blue_object_size = smoothed_size

        fov = self.robot.get_camera_field_of_view()
        self.blue_object_angle = ((cx / img_width) - 0.5) * fov
        self.object_is_low = cy > img_height * 0.5

    def _clear_object_data(self):
        """Clear object data."""
        self.blue_object_center = None
        self.blue_object_size = None
        self.blue_object_angle = None
        self.object_is_low = False

    def get_ticks_per_second(self):
        """Get ticks per second."""
        self.ticksPerSecond[0] = self.robot.get_left_motor_encoder_ticks() - self.lastTick[0]
        self.ticksPerSecond[1] = self.robot.get_right_motor_encoder_ticks() - self.lastTick[1]
        self.lastTick[0] = self.robot.get_left_motor_encoder_ticks()
        self.lastTick[1] = self.robot.get_right_motor_encoder_ticks()

    def drive_straight_after_align(self):
        """Drive straight after aligning."""
        avg_ticks = sum(self.ticksPerSecond) / 2
        dist_moved = avg_ticks * 0.1
        self.straight_distance -= dist_moved

        if self.straight_distance <= 0:
            self.left_motor_speed = 0
            self.right_motor_speed = 0
            self.going_straight_after_spin = False
        else:
            self.left_motor_speed = 0.4
            self.right_motor_speed = 0.4

    def align_to_target_after_spin(self):
        """Align to target after spinning."""
        if self.final_phase == "rotate":
            if self.blue_object_angle is not None:
                self.left_motor_speed += self.turn_speed * self.blue_object_angle / 45 + 0.1
                self.right_motor_speed += self.turn_speed * -self.blue_object_angle / 45 + 0.1

                if abs(self.blue_object_angle) < 0.05:
                    self.final_phase = "drive"
                    self.final_ticks = 0
            else:
                self.final_phase = "drive"
                self.final_ticks = 0

    def drive_straight(self):
        """Drive straight."""
        avg_ticks = sum(self.ticksPerSecond) / 2
        dist_moved = avg_ticks * 0.01
        self.straight_distance -= dist_moved

        if self.straight_distance <= 0:
            self.left_motor_speed = 0
            self.right_motor_speed = 0
            self.going_straight = False
        else:
            self.left_motor_speed = 0.4
            self.right_motor_speed = 0.4

    def calculate_distance_to_object(self) -> float:
        """Estimate distance based on object's pixel height."""
        if self.object_image_height_z is None or self.object_image_height_z == 0:
            return float('inf')  # если объект не найден

        k = 20.0  # эмпирическая константа — подбирается под эксперимент
        distance = k / self.object_image_height_z
        print(
            f"self.object_image_height_z {self.object_image_height_z}, self.max {self.max}, self.min {self.min}, img_height {self.img_height}")
        return distance

    def navigate_to_visible_object(self):
        """Navigate to visible object."""
        self.spin_counter = 0
        self.is_spinning_180 = False
        self.backward_mode = False
        self.found_after_spin = False
        self.aligning_to_target_after_spin = False

        if self.object_is_low:
            distance = self.calculate_distance_to_object()  # Используем новый метод для получения расстояния

            if self.min > 470:
                self.final_mode = True
                self.final_phase = "rotate"
                self.final_ticks = 0
                self.final_distance_moved = 0.0
                print(f"Final phase triggered! Distance: {distance:.2f} m")
                return

            # Продолжаем обычный процесс поиска объекта
            self.straight_distance = distance
            self.going_straight = True

            angle_tolerance = -0.3
            if abs(self.blue_object_angle) < angle_tolerance:
                k = 3.5
                turn = k * self.blue_object_angle

                base_speed = 0.8
                self.left_motor_speed = base_speed + turn
                self.right_motor_speed = base_speed - turn

    def spin_180(self):
        """Spin 180."""
        if self.target_visible:
            self.is_spinning_180 = False
            self.left_motor_speed = 0
            self.right_motor_speed = 0
            self.spin_counter = 0
            self.going_straight = True
            distance = 1.0 / (self.blue_object_size / 5000)
            distance = min(distance, 1.5)
            self.straight_distance = distance
            print(f"Target found NOT FINAL. Distance: {distance:.2f} m")
            return

        if self.spin_counter < 1060:
            self.left_motor_speed = -0.5
            self.right_motor_speed = 0.5
            self.spin_counter += 1
            print(f"Rotating... {self.spin_counter}")
        else:
            self._rotate_target_color()
            print(f"Rotating... {self.spin_counter}")
            self.what()

    def what(self):
        self.detect_red_object()

    def counter_1060_logic(self):
        """Counter 1060 logic."""
        if self.target_visible:
            self.after_spin_failed = False
            # self.backward_mode = True

            self.final_mode = True
            self.final_phase = "rotate"
            self.final_ticks = 0
            self.final_distance_moved = 0.0
        else:
            self.left_motor_speed = -0.3
            self.right_motor_speed = -0.3
            print("Reversing after failed turn...")

    def final_steps(self):
        """Move robot to target object (final approach phase)."""
        if self.final_phase == "rotate":
            if self.blue_object_angle is not None:
                self.left_motor_speed = self.turn_speed * self.blue_object_angle
                self.right_motor_speed = self.turn_speed * -self.blue_object_angle

                print(f"Final turn toward target... angle: {round(self.blue_object_angle, 2)}°")

                if abs(self.blue_object_angle) < 0.05:
                    self.final_phase = "drive"
                    self.final_ticks = 0
                    print(" Final turn completed. Target centered. Proceeding to movement.")
                return

        if self.final_phase == "drive":
            avg_ticks = sum(self.ticksPerSecond) / 2
            delta = avg_ticks * 0.001
            self.final_distance_moved += delta
            print(f"🫐🫐🫐 WIDTH {self.width}")
            if self.width >= 190:
                self.left_motor_speed = 0
                self.right_motor_speed = 0
                self.final_mode = False
                self.final_phase = None

                self._rotate_target_color()  # переключаем на следующий цвет
                self.state = "init"  # сбрасываем состояние для новой фазы поиска

                print(f"Target acquired! Switching to next color: {self.target_color}")

            else:
                self.left_motor_speed = 0.4
                self.right_motor_speed = 0.4
                print(f"Moving forward. Left: {1 - self.final_distance_moved:.2f} м")
            return

    def realistic_motor_control(self):
        """Realistic motor control."""
        left_error = self.left_motor_speed - (self.ticksPerSecond[0] / self.realistic_speed_divider)
        right_error = self.right_motor_speed - (self.ticksPerSecond[1] / self.realistic_speed_divider)

        left_torque = np.clip(left_error * self.realistic_torque,
                              -self.realistic_torque, self.realistic_torque)
        right_torque = np.clip(right_error * self.realistic_torque,
                               -self.realistic_torque, self.realistic_torque)

        self.robot.set_left_motor_torque(left_torque)
        self.robot.set_right_motor_torque(right_torque)

    def adjust_for_realistic(self):
        """Adjust motor position."""
        if not self.robot.get_realistic():
            return

        self.turn_speed = 0.2
        self.min_object_area = 150

    def act(self) -> None:
        """Execute planned actions.

        Perform the actions decided in the planning step, such as moving or
        interacting with the environment.
        """
        if not self.robot.get_realistic():
            self.robot.set_left_motor_velocity(self.left_motor_speed)
            self.robot.set_right_motor_velocity(self.right_motor_speed)
            return

        self.adjust_for_realistic()
        self.realistic_motor_control()

    def plan(self):
        """Plan the robot's actions by processing sensor data and deciding next steps.

        The planner handles different operational modes including:
        - Final approach maneuvers
        - Target search patterns
        - Recovery behaviors
        - Navigation to visible targets
        """
        if self._check_completion():
            return

        if self._handle_special_modes():
            return

        if self._handle_visible_target():
            return

        self._handle_target_search()

    def _check_completion(self):
        """Check if planning should terminate."""
        return self.done

    def _handle_special_modes(self):
        """Handle all special operational modes."""
        if self.final_mode:
            self.final_steps()
            return True

        if self.after_spin_failed:
            self.counter_1060_logic()
            return True

        if self.going_straight_after_spin:
            self.drive_straight_after_align()
            return True

        if self.found_after_spin and not self.going_straight_after_spin:
            self.align_to_target_after_spin()
            return True

        if self.going_straight:
            self.drive_straight()
            return True

        return False

    def _handle_visible_target(self):
        """Handle cases where target is currently visible."""
        if self.target_visible:
            self.navigate_to_visible_object()
            return True
        return False

    def _handle_target_search(self):
        """Initiate target search patterns when no target is visible."""
        if not self.is_spinning_180:
            self._initiate_360_scan()

        if self.is_spinning_180:
            self.spin_180()

    def _initiate_360_scan(self):
        """Initialize parameters for 360° search pattern."""
        self.is_spinning_180 = True
        print(" Target lost — initiating 360° scan turn")

    def spin(self) -> None:
        """Spin the robot.

        This is the main loop where the robot performs its sense-plan-act cycle.
        """
        self.sense()
        self.plan()
        self.act()