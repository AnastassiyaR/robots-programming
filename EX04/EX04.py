"""EX04: PID Control."""


class Robot:
    """Turtlebot robot."""

    def __init__(self, robot: object) -> None:
        """Class initializer.

        Args:
            robot (object): An instance of a Turtlebot-like robot interface.
        """
        self.robot = robot
        self.kp = 1.0
        self.ki = 0.1
        self.kd = 0.05
        self.left_target = 0.0
        self.right_target = 0.0
        self.integral_left = 0.0
        self.integral_right = 0.0
        self.previous_error_left = 0.0
        self.previous_error_right = 0.0
        self.previous_time = self.robot.get_time() or 0.0
        self.current_left_speed = 0.0
        self.current_right_speed = 0.0
        self.previous_left_ticks = 0
        self.previous_right_ticks = 0

    def set_pid(self, kp: float = 1.0, ki: float = 0.1, kd: float = 0.05) -> None:
        """Set the PID controller gains for the robot's wheel speed control.

        Args:
            kp (float): Proportional gain.
            ki (float): Integral gain.
            kd (float): Derivative gain.
        """
        self.kp = kp
        self.ki = ki
        self.kd = kd

    def set_target_speeds(self, left_target: float, right_target: float) -> None:
        """Set the target speeds for the robot's wheels.

        Args:
            left_target (float): Target speed for the left wheel.
            right_target (float): Target speed for the right wheel.
        """
        self.left_target = left_target
        self.right_target = right_target

    def update_left_wheel_speed(self) -> None:
        """Update left wheel speed using PID control."""
        current_time = self.robot.get_time()
        delta_time = current_time - self.previous_time

        current_ticks = self.robot.get_left_motor_encoder_ticks()
        current_speed = (current_ticks - self.previous_left_ticks) / delta_time if delta_time > 0 else 0
        self.previous_left_ticks = current_ticks

        error = self.left_target - current_speed
        derivative_left = 0.0 if delta_time < 1e-3 else (error - self.previous_error_left) / delta_time
        self.integral_left += error * delta_time

        correction_left = self.kp * error + self.ki * self.integral_left + self.kd * derivative_left

        self.previous_error_left = error
        self.current_left_speed = correction_left

    def update_right_wheel_speed(self) -> None:
        """Update right wheel speed using PID control."""
        current_time = self.robot.get_time()
        delta_time = current_time - self.previous_time

        current_ticks = self.robot.get_right_motor_encoder_ticks()
        current_speed = (current_ticks - self.previous_right_ticks) / delta_time if delta_time > 0 else 0
        self.previous_right_ticks = current_ticks

        error = self.right_target - current_speed
        derivative_right = 0.0 if delta_time < 1e-3 else (error - self.previous_error_right) / delta_time
        self.integral_right += error * delta_time

        correction_right = self.kp * error + self.ki * self.integral_right + self.kd * derivative_right

        self.previous_error_right = error
        self.current_right_speed = correction_right

    def get_pid_corrected_left_wheel_speed(self) -> float:
        """Return the corrected left wheel speed."""
        return self.current_left_speed

    def get_pid_corrected_right_wheel_speed(self) -> float:
        """Return the corrected right wheel speed."""
        return self.current_right_speed

    def sense(self) -> None:
        """Gather sensor data."""
        pass

    def plan(self) -> None:
        """Plan robot actions."""
        self.update_left_wheel_speed()
        self.update_right_wheel_speed()
        self.previous_time = self.robot.get_time()

    def act(self) -> None:
        """Execute planned actions."""
        pass

    def spin(self) -> None:
        """Spin the robot."""
        self.sense()
        self.plan()
        self.act()
