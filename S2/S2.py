"""S2."""

import math


class Robot:
    """Turtlebot robot."""

    def __init__(self, robot: object) -> None:
        """Class initializer."""
        self.robot = robot
        self.data = None
        self.detected_objects = []
        self.orientation = None
        self.previous_orientation = 0.0

        # State management
        self.state = None
        self.target = None
        self.distance_to_target = 0
        self.select_closest = True
        self.total_rotation = 0.0
        self.left_speed = 0.0
        self.right_speed = 0.0
        self.torque_left = 0.0
        self.torque_right = 0.0

        # Constants
        self.STOP_DISTANCE = 0.3  # in meters
        self.OBJECT_THRESHOLD = 0.3  # in meters
        self.ANGULAR_DIFFERENCE = 3  # angular distance between objects in degrees
        self.MAX_TORQUE = 0.1  # maximum torque to apply to the wheel
        self.TICKS_PER_ROTATION = 508.8  # Constants for encoder ticks to velocity conversion
        self.MIN_OBJECT_DISTANCE = 0.4
        self.MAX_OBJECT_DISTANCE = 3.7

        # PID Control
        self.kp = 0.08  # Proportional gain
        self.ki = 0.33  # Integral gain
        self.kd = 0.0  # Derivative gain

        # PID state variables for left motor
        self.previous_error_left = 0
        self.integral_left = 0
        self.previous_left_ticks = 0

        # PID state variables for right motor
        self.previous_error_right = 0
        self.integral_right = 0
        self.previous_right_ticks = 0

        # Time tracking
        self.current_time = self.robot.get_time() or 0.0
        self.previous_time = self.current_time

    def set_target_speeds(self, left_target: float, right_target: float) -> None:
        """Set the target speeds for the robot's wheels."""
        self.left_speed = left_target
        self.right_speed = right_target

    def convert_to_angle(self, index):
        """Convert lidar values to angle in degrees."""
        return index * 360 / 640 + 270

    def get_turn_angle(self, angle):
        """Set range of -pi...pi."""
        angle = angle % 360
        return (angle + 180) % 360 - 180

    def calculate_total_rotation(self, orientation, previous_orientation) -> float:
        """Calculate the total rotation of the robot."""
        delta = orientation - previous_orientation
        if delta > math.pi:
            delta -= 2 * math.pi
        elif delta < -math.pi:
            delta += 2 * math.pi
        return self.total_rotation + delta

    def get_detected_objects(self, lidar_data) -> list:
        """Return the detected objects range list."""
        new_objects = []

        if lidar_data:
            lidar_points = len(lidar_data)
            current_object = []
            in_object = False

            orientation_angle_in_degrees = self.get_turn_angle(math.degrees(self.orientation))

            def is_invalid(index):
                return any(lidar_data[j] == math.inf for j in (index, index - 1, index + 1))

            for i in range(1, lidar_points - 1):
                if is_invalid(i):
                    in_object, current_object = False, []
                    continue

                angle = self.convert_to_angle(i)
                absolute_angle = self.get_turn_angle(angle) - orientation_angle_in_degrees
                norm_angle = self.get_turn_angle(absolute_angle)

                # Detect object start
                if (not in_object and abs(lidar_data[i] - lidar_data[i - 1]) > self.OBJECT_THRESHOLD and lidar_data[i]
                        < lidar_data[i - 1]):
                    in_object = True
                    current_object = [(lidar_data[i],
                                       self.get_turn_angle(angle), i, orientation_angle_in_degrees, norm_angle)]

                # Continue tracking object
                elif in_object:
                    current_object.append((lidar_data[i],
                                           self.get_turn_angle(angle), i, orientation_angle_in_degrees, norm_angle))

                    if (abs(lidar_data[i] - lidar_data[i + 1])
                            > self.OBJECT_THRESHOLD and lidar_data[i] < lidar_data[i + 1]):
                        closest_obj = min(current_object, key=lambda x: x[0])

                        # only add far enough objects this time
                        if closest_obj[0] > self.MIN_OBJECT_DISTANCE and closest_obj[0] < self.MAX_OBJECT_DISTANCE:
                            new_objects.append(closest_obj)
                        elif closest_obj[0] > self.MAX_OBJECT_DISTANCE:
                            in_object, current_object = False, []
        return new_objects

    def deduplicate_objects(self, new_detected_objects: list, previous_detected_objects: list) -> list:
        """Reduce amount of duplicate targets to consider."""
        joined_list = previous_detected_objects.copy()

        for new_obj in new_detected_objects:
            # Check if this object is sufficiently different from ALL existing objects
            is_new = all(abs(new_obj[4] - existing_obj[4])
                         > self.ANGULAR_DIFFERENCE for existing_obj in previous_detected_objects)
            if is_new:
                joined_list.append(new_obj)
        return joined_list

    def select_object(self, existing_objects, select_closest):
        """Select appropriate target to go to."""
        valid_objects = existing_objects.copy()
        if select_closest:
            return min(valid_objects, key=lambda x: x[0])
        return max(valid_objects, key=lambda x: x[0])

    def is_within_range(self, data, distance_to_target, stop_distance):
        """Check the object is within a stop range."""
        for i in range(460, 500):
            if data[i] < stop_distance:
                return True

        # check alternative source of information, useful for realistic mode
        if self.robot.get_realistic() and distance_to_target < stop_distance:
            return True

        return False

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

        # for realistic mode adjust wheels speeds
        current_left_ticks = self.robot.get_left_motor_encoder_ticks()
        current_right_ticks = self.robot.get_right_motor_encoder_ticks()

        # calculate travelled distance left to go
        delta_left = current_left_ticks - self.previous_left_ticks
        delta_right = current_right_ticks - self.previous_right_ticks
        self.distance_to_target -= self.ticks_to_distance((delta_left + delta_right) / 2)

        # Compute current angular velocity (rad/s)
        left_angular_velocity = self.ticks_to_velocity(delta_left, delta_time)
        right_angular_velocity = self.ticks_to_velocity(delta_right, delta_time)

        self.previous_left_ticks = current_left_ticks
        self.previous_right_ticks = current_right_ticks

        # PID for left motor
        error_left = self.left_speed - left_angular_velocity
        derivative_left = (error_left - self.previous_error_left) / delta_time if delta_time > 0 else 0
        self.integral_left += error_left * delta_time
        self.integral_left = max(min(self.integral_left, 0.1), -0.1)
        self.torque_left = (self.kp * error_left + self.ki * self.integral_left + self.kd * derivative_left)
        self.previous_error_left = error_left

        # PID for right motor
        error_right = self.right_speed - right_angular_velocity
        derivative_right = (error_right - self.previous_error_right) / delta_time if delta_time > 0 else 0
        self.integral_right += error_right * delta_time
        self.integral_right = max(min(self.integral_right, 0.1), -0.1)
        self.torque_right = (self.kp * error_right + self.ki * self.integral_right + self.kd * derivative_right)
        self.previous_error_right = error_right

        # Apply correction within a bounded range of torques
        self.torque_left = max(min(self.torque_left, self.MAX_TORQUE), -self.MAX_TORQUE)
        self.torque_right = max(min(self.torque_right, self.MAX_TORQUE), -self.MAX_TORQUE)

    def get_pid_corrected_left_wheel_speed(self) -> float:
        """Return the corrected left wheel speed."""
        return self.torque_left

    def get_pid_corrected_right_wheel_speed(self) -> float:
        """Return the corrected right wheel speed."""
        return self.torque_right

    def align_to_cylinder_center(self) -> None:
        """Adjust speeds of both wheels to center the object."""
        if self.target:
            # find the angle deviation (in radians) from 480 lidar point which is a center one
            min_index = min(enumerate(self.data[470:490]), key=lambda x: x[1])[0]
            error = 2 * math.pi / 640 * (min_index - 10)
            self.left_speed += error
            self.right_speed -= error

            self.left_speed = max(min(self.left_speed, 1), -1)
            self.right_speed = max(min(self.right_speed, 1), -1)

    def sense(self) -> None:
        """Gather sensor data."""
        self.current_time = self.robot.get_time()
        self.orientation = self.robot.get_orientation()
        self.data = self.robot.get_lidar_range_list()
        self.total_rotation = self.calculate_total_rotation(self.orientation, self.previous_orientation)
        self.previous_orientation = self.orientation

        if self.state:
            if self.state == "searching":
                # do not try to search other object when it is not required
                self.detected_objects = self.deduplicate_objects(self.get_detected_objects(self.data), self.detected_objects)
        else:
            self.state = "init"

    def plan(self) -> None:
        """Plan the robot's actions."""
        if self.state == "init":
            self.detected_objects = []
            self.total_rotation = 0
            self.state = "searching"

        elif self.state == "searching":
            self.set_target_speeds(-0.5, 0.5)
            if abs(self.total_rotation) >= 2 * math.pi:
                self.state = "selecting"

        elif self.state == "selecting":
            self.target = self.select_object(self.detected_objects, self.select_closest)
            self.select_closest = not self.select_closest
            self.state = "turn_to_object"

        elif self.state == "turn_to_object":
            self.set_target_speeds(-0.5, 0.5)

            # measuring difference in degrees
            obj_distance, obj_angle_in_degrees, _, orientation_at_detection, norm_angle = self.target
            orientation_angle_in_degrees = self.get_turn_angle(math.degrees(self.orientation))
            desired_orientation = obj_angle_in_degrees - orientation_at_detection

            # allow 0.3 (1/2 of lidar accuracy) degree difference
            delta_orientation = abs(orientation_angle_in_degrees - self.get_turn_angle(180 - desired_orientation))
            # print(f"Delta orientation = {delta_orientation}")
            if delta_orientation < 0.3:
                # additional way to track the distance if robot misses a point a bit
                self.distance_to_target = obj_distance + 0.1
                self.state = "approach"

        elif self.state == "approach":
            self.set_target_speeds(0.5, 0.5)
            self.align_to_cylinder_center()

            if self.is_within_range(self.data, self.distance_to_target, self.STOP_DISTANCE):
                self.set_target_speeds(0, 0)
                self.state = "init"

        else:
            self.set_target_speeds(0, 0)

        self.update_both_wheel_speeds()

        # record current time to measure passed delta on next cycle
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
