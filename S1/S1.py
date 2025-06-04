import math


class Robot:
    """Turtlebot robot that navigates using LIDAR data.

    This class implements a sense-plan-act cycle where the robot:
    - Collects sensor data (LIDAR readings and encoder ticks)
    - Detects objects in its environment
    - Plans movement based on detected obstacles
    - Executes movement commands accordingly

    The robot adjusts its wheel speeds and torque dynamically based on object proximity and angle.
    """

    def __init__(self, robot: object) -> None:
        """Initialize the robot with necessary attributes.

        Args:
            robot (object): An instance of a Turtlebot-like robot interface.
        """
        self.robot = robot
        self.lidar_data = []  # Stores LIDAR sensor data
        self.detected_objects = []  # Stores detected objects
        self.lastTick = [0, 0]  # Stores the last encoder tick values for both wheels
        self.ticksPerSecond = [0, 0]  # Stores the encoder ticks per second for both wheels
        self.left_speed = -1  # Initial left wheel speed
        self.right_speed = 1  # Initial right wheel speed

    def sense(self) -> None:
        """Collect sensor data using the robot's LIDAR and encoders."""
        lidar_list = self.robot.get_lidar_range_list()
        if lidar_list:
            self.lidar_data = lidar_list  # Update LIDAR data
            self.detect_objects()  # Detect objects based on new LIDAR data
        else:
            self.lidar_data = []  # Clear LIDAR data if no readings are available
            self.detected_objects = []  # Clear detected objects list
        self.get_ticksPerSecond()  # Update encoder ticks per second

    def detect_objects(self) -> None:
        """Find objects using LIDAR data.

        The function analyzes LIDAR readings to detect objects based on sudden changes in distance values.
        It identifies the center position and angle of each detected object.
        """
        elements = len(self.lidar_data)  # Number of LIDAR data points
        self.angle = (2 * math.pi) / elements  # Angle between each LIDAR data point
        self.detected_objects = []  # Reset detected objects list

        index = 1
        while index < elements - 1:
            previous = self.lidar_data[index - 1]  # Previous LIDAR reading
            current = self.lidar_data[index]  # Current LIDAR reading
            next = self.lidar_data[index + 1]  # Next LIDAR reading

            # Skip if any of the readings are invalid (infinity)
            if math.isinf(previous) or math.isinf(current) or math.isinf(next):
                index += 1
                continue

            # Detect a sudden drop in distance, indicating an object
            if current < previous * 0.9:
                objStart = index  # Start index of the object
                valid = True  # Flag to check if the object is valid
                object_points = [current]  # List to store points belonging to the object

                while index < elements - 1:
                    current = self.lidar_data[index]
                    next = self.lidar_data[index + 1]

                    # Skip if readings are invalid
                    if math.isinf(current) or math.isinf(next):
                        valid = False
                        break

                    object_points.append(current)  # Add current point to the object

                    # Detect the end of the object (sudden increase in distance)
                    if next > current * 1.1:
                        objFinish = index  # End index of the object

                        # If the object is valid and has enough points, calculate its center
                        if valid and len(object_points) > 2:
                            center_index = (objStart + objFinish) // 2  # Center index of the object
                            centerDistance = self.lidar_data[center_index]  # Distance to the center
                            centerAngle = center_index * self.angle  # Angle to the center
                            self.detected_objects.append((centerDistance, centerAngle))  # Add object to the list
                        break

                    index += 1
            index += 1

    def get_ticksPerSecond(self):
        """Retrieve the encoder ticks per second for each wheel."""
        self.ticksPerSecond[0] = self.robot.get_left_motor_encoder_ticks() - self.lastTick[0]  # Left wheel ticks per second
        self.ticksPerSecond[1] = self.robot.get_right_motor_encoder_ticks() - self.lastTick[1]  # Right wheel ticks per second
        self.lastTick[0] = self.robot.get_left_motor_encoder_ticks()  # Update last left wheel tick count
        self.lastTick[1] = self.robot.get_right_motor_encoder_ticks()  # Update last right wheel tick count

    def get_objects(self) -> list:
        """Return the detected objects as a list."""
        return self.detected_objects if self.detected_objects else None  # Return detected objects or None if none are detected

    def plan(self) -> None:
        """Plan the robot's actions based on detected objects.

        The robot determines its movement strategy depending on the location of the closest object.
        If an obstacle is near, it adjusts its trajectory accordingly.
        """
        if self.detected_objects:
            target = math.pi / 60  # Target angle threshold for object avoidance
            closest_object = min(self.detected_objects, key=lambda x: x[0])  # Find the closest object
            distance, angle = closest_object  # Distance and angle to the closest object
            angle = angle % (2 * math.pi)  # Normalize the angle to [0, 2π)

            # Adjust wheel speeds based on the angle of the closest object
            if angle > 3 * math.pi / 2 - target:
                self.left_speed = 1  # Turn left
                self.right_speed = -1
            else:
                self.left_speed = -1  # Turn right
                self.right_speed = 1

            # If the object is directly in front, move forward
            if abs(angle - (3 * math.pi / 2)) < target:
                self.left_speed = 1
                self.right_speed = 1

            # Stop if the object is too close
            if distance < 0.3:
                self.right_speed = 0
                self.left_speed = 0

    def act(self) -> None:
        """Execute planned actions.

        Perform the actions decided in the planning step, such as moving or
        interacting with the environment.
        """
        if not self.robot.get_realistic():
            # Set wheel speeds directly if the robot is not in realistic mode
            self.robot.set_left_motor_velocity(self.left_speed)
            self.robot.set_right_motor_velocity(self.right_speed)
            return

        # Adjust torque for realistic movement
        div = 15  # Divisor for converting ticks per second to speed
        torque = 0.05  # Torque value for motor adjustment

        # Adjust left motor torque based on speed difference
        if self.ticksPerSecond[0] / div > self.left_speed:
            self.robot.set_left_motor_torque(-torque)  # Reduce speed
        elif self.ticksPerSecond[0] / div < self.left_speed:
            self.robot.set_left_motor_torque(torque)  # Increase speed
        else:
            self.robot.set_left_motor_torque(0)  # Maintain speed

        # Adjust right motor torque based on speed difference
        if self.ticksPerSecond[1] / div > self.right_speed:
            self.robot.set_right_motor_torque(-torque)  # Reduce speed
        elif self.ticksPerSecond[1] / div < self.right_speed:
            self.robot.set_right_motor_torque(torque)  # Increase speed
        else:
            self.robot.set_right_motor_torque(0)  # Maintain speed

    def spin(self) -> None:
        """Execute the sense-plan-act cycle continuously."""
        self.sense()  # Collect sensor data
        self.plan()  # Plan actions based on sensor data
        self.act()  # Execute planned actions
