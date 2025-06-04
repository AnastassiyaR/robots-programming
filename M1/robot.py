"""M1."""


class Robot:
    """Turtlebot robot."""

    def __init__(self, robot: object) -> None:
        """Class initializer."""
        self.robot = robot
        self.left_speed = 1
        self.right_speed = 1
        self.in_maze = True
        self.iteration = 0

        self.turns = 1

        self.turning_left = False
        self.turning_right = False
        self.turn_start_iter = 0

        self.last_step = False
        self.last_step_start_iter = 0
        self.seconds_to_iters = 100

        self.base_speed = 0.5
        self.turn_speed = 0.5
        self.wall_threshold = 150
        self.exit_counter = 0

    def sense(self) -> None:
        """Сбор данных с датчиков."""
        self.ir_left = self.robot.get_ir_intensity_left()
        self.ir_side_left = self.robot.get_ir_intensity_side_left()
        self.ir_center_left = self.robot.get_ir_intensity_center_left()
        self.ir_center = self.robot.get_ir_intensity_center()
        self.ir_center_right = self.robot.get_ir_intensity_center_right()
        self.ir_side_right = self.robot.get_ir_intensity_side_right()
        self.ir_right = self.robot.get_ir_intensity_right()
        self.ir_list = self.robot.get_ir_intensities_list()

        print(f"get_ir_intensities_list():        {self.ir_list}")
        self.iteration += 1

    def plan(self) -> None:
        """Plan."""
        if (self.turns == 1):
            self.left_speed = self.base_speed + 5
            self.right_speed = self.base_speed + 5

        if all(x == 12.0 for x in self.ir_list[0:6]) and self.turns > 8 and not self.last_step:
            self.last_step = True
            return

        if self.last_step:
            if self.last_step_start_iter >= 550:
                self.left_speed = 0
                self.right_speed = 0
                self.in_maze = False
            else:
                self.last_step_start_iter += 1
            return
        else:
            self.turning()

    def turning(self) -> None:
        """Turn."""
        if not self.turning_left and not self.turning_right and self.ir_center > 12:
            print(self.turns)
            if self.turns in [1, 2, 5, 6]:
                self.turning_left = True
                self.turn_start_iter = self.iteration
                self.left_speed = -self.turn_speed
                self.right_speed = self.turn_speed
            else:
                self.turning_right = True
                self.turn_start_iter = self.iteration
                self.left_speed = self.turn_speed
                self.right_speed = -self.turn_speed
            self.turns += 1
            return

        elif self.turning_left:
            if self.iteration - self.turn_start_iter >= 274:
                self.turning_left = False
                if (self.turns in [3, 5, 7, 9]):
                    self.left_speed = self.base_speed + 5
                    self.right_speed = self.base_speed + 5
                else:
                    self.left_speed = self.base_speed
                    self.right_speed = self.base_speed
            return

        elif self.turning_right:
            if self.iteration - self.turn_start_iter >= 274:
                self.turning_right = False
                if (self.turns in [3, 5, 7, 9]):
                    self.left_speed = self.base_speed + 5
                    self.right_speed = self.base_speed + 5
                else:
                    self.left_speed = self.base_speed
                    self.right_speed = self.base_speed
            return

    def act(self) -> None:
        """Act."""
        self.robot.set_left_motor_velocity(self.left_speed)
        self.robot.set_right_motor_velocity(self.right_speed)

    def spin(self) -> None:
        """Spin."""
        self.sense()
        self.plan()
        self.act()
