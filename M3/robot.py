"""M3."""

from __future__ import annotations

import math


class Robot:
    """Turtlebot robot."""

    def __init__(self, robot: object) -> None:
        """Class initializer.

        Args:
            robot (object): An instance of a Turtlebot-like robot interface.
        """
        self.robot = robot
        self.lidar_data = None
        self.position = (0, 0)
        self.target_position = None
        self.orientation = 0.0
        self.previous_orientation = 0.0
        self.total_rotation = 0.0
        self.turn_angle = 0
        self.traversable_cells = []
        self.cell_adjacency_map = {}
        self.final_map = {}
        self.visited_cells = []
        self.unmapped_cells = []
        self.frontier_and_path = None
        self.potential_exit = None
        self.potential_exit_direction = None

        self.state = None
        self.mapping_mode = True
        self.travelled_distance = 0
        self.left_speed = 0.0
        self.right_speed = 0.0
        self.torque_left = 0.0
        self.torque_right = 0.0

        self.MAX_TORQUE = 0.1
        self.TICKS_PER_ROTATION = 508.8
        self.CELL_SIZE = 0.615

        self.kp = 0.08
        self.ki = 0.33
        self.kd = 0.0

        self.previous_error_left = 0
        self.integral_left = 0
        self.previous_left_ticks = 0

        self.previous_error_right = 0
        self.integral_right = 0
        self.previous_right_ticks = 0

        self.current_time = self.robot.get_time() or 0.0
        self.previous_time = self.current_time

    def forward_speed(self, left_target: float, right_target: float) -> None:
        """Set the forward speed."""
        self.left_speed = left_target * 3
        self.right_speed = right_target * 3

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
        """Update both wheel speeds."""
        delta_time = self.current_time - self.previous_time

        current_left_ticks = self.robot.get_left_motor_encoder_ticks()
        current_right_ticks = self.robot.get_right_motor_encoder_ticks()

        delta_left = current_left_ticks - self.previous_left_ticks
        delta_right = current_right_ticks - self.previous_right_ticks
        self.travelled_distance += self.ticks_to_distance((delta_left + delta_right) / 2)

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

    def align_to_orientation(self) -> None:
        """Adjust speeds of both wheels to align robot's position to the robot orientation."""
        if self.state == "drive":
            theta = self.orientation % (2 * math.pi)

            closest_ideal = round(theta / (math.pi / 2)) * (math.pi / 2)
            error = theta - closest_ideal

            if error > math.pi:
                error -= 2 * math.pi
            elif error < -math.pi:
                error += 2 * math.pi

            tolerance = 0.01 * (math.pi / 2)
            if abs(error) < tolerance:
                error = 0

            self.left_speed += error
            self.right_speed -= error

    def calculate_delta_rotation(self, orientation, previous_orientation) -> float:
        """Calculate the total rotation of the robot."""
        delta = orientation - previous_orientation
        if delta > math.pi:
            delta -= 2 * math.pi
        elif delta < -math.pi:
            delta += 2 * math.pi
        return delta

    def get_traversable_cells(self) -> list:
        """Get a list of all known traversable cells in the map.

        This method returns the grid cells that the robot knows to be traversable,
        starting from the initial position (0, 0). A traversable cell is a cell
        that the robot can go to.

        Returns:
            [(int, int), ...]: A list of tuples, where each tuple (x, y)
            represents the coordinates of a traversable cell.
        """
        return list(set(self.traversable_cells))

    def get_unmapped_cells(self) -> list:
        """Get a list of all unmapped cells that the robot has discovered so far.

        This method identifies grid cells that have been found but not yet fully mapped,
        starting with the initial position (0, 0).
        A cell is considered mapped when the robot has gathered a LIDAR reading while
        located in that cell. Then it can be removed from unmapped cells.
        This method returns a subset of all traversable cells.

        Returns:
            [(int, int), ...]: A list of tuples, where each tuple (x, y)
            represents the coordinates of an unmapped cell.
        """
        return self.unmapped_cells

    def get_map(self) -> dict:
        """Get the map representation as a dictionary of adjacency.

        The map is represented as a dictionary where each key is a grid cell
        (represented as a tuple of coordinates), and the corresponding value
        is a list of adjacent cells.
        Adjacency is defined as orthogonal movement meaning: up, down, left, or right.

        Returns:
            {(int, int): [(int, int), ...]}: A dictionary where keys are cells (x, y)
            and values are lists of neighboring cells (x, y).
        """
        return self.cell_adjacency_map

    def get_frontier_and_path(self) -> list:
        """Identify next frontier for exploration and calculate the path to reach it.

        The frontier is the boundary between the known (mapped) and unknown (unmapped)
        regions of the map. This method determines the most suitable frontier to
        explore next and computes the path from the robot's current position to that
        frontier. Formula for choosing the next frontier to explore: Manhattan distance

        Returns:
            [(int, int), [(int, int), ...]]:
            - The first element is a tuple (x, y) representing the coordinates of the
              selected frontier.
            - The second element is a list of tuples [(x1, y1), (x2, y2), ...]
              representing the sequence of grid cells (coordinates) the robot should
              traverse in order to reach the frontier.

        Example:
            Suppose the robot's current position is (0, 0), and it detects a frontier
            at (3, 0). The function might return:
                [(3, 0), [(0, 0), (1, 0), (2, 0), (3, 0)]]
            This means the robot should travel through the listed cells to reach the
            frontier at (3, 0).
        """
        return self.frontier_and_path

    def orientation_to_direction(self, orientation):
        """Convert the robot's orientation angle to direction (up, left, down, right)."""
        threshold = 0.1

        if abs(orientation) < threshold:
            return "up"
        elif abs(orientation - math.pi / 2) < threshold:
            return "left"
        elif abs(orientation + math.pi) < threshold or abs(orientation - math.pi) < threshold:
            return "down"
        elif abs(orientation + math.pi / 2) < threshold:
            return "right"
        else:
            return "unknown"

    def get_lidar_index_for_direction(self, robot_direction, side_to_check):
        """Determine the LIDAR data index based on the robot's direction and side to check."""
        lidar_indices = {"up": 480, "left": 320, "down": 160, "right": 639}

        directions = ["up", "left", "down", "right"]

        shift = directions.index(robot_direction)

        rotated_index = (directions.index(side_to_check) - shift) % 4

        return list(lidar_indices.values())[rotated_index]

    def is_traversable_in_all_directions(self, allow_inf=False) -> dict:
        """Check whether the robot can move in all four directions (up, left, down, right)."""
        sides = ["up", "left", "down", "right"]
        result = {}

        if self.lidar_data:
            robot_direction = self.orientation_to_direction(self.orientation)

            for side_to_check in sides:
                index = self.get_lidar_index_for_direction(robot_direction, side_to_check)
                filtered_values = [v for v in self.lidar_data[index - 10: index + 10]
                                   if not math.isinf(v) and not math.isnan(v)]
                wide_infinity_values = [v for v in self.lidar_data[index - 50: index + 50]
                                        if not math.isinf(v) and not math.isnan(v)]

                if len(wide_infinity_values) == 0 and not allow_inf:
                    self.potential_exit = self.position
                    if side_to_check == "up":
                        self.potential_exit_direction = (self.position[0], self.position[1] + 1)
                    if side_to_check == "left":
                        self.potential_exit_direction = (self.position[0] - 1, self.position[1])
                    if side_to_check == "down":
                        self.potential_exit_direction = (self.position[0], self.position[1] - 1)
                    if side_to_check == "right":
                        self.potential_exit_direction = (self.position[0] + 1, self.position[1])
                    distance = 0

                elif not filtered_values:

                    distance = 10 * self.CELL_SIZE if allow_inf else 1
                else:
                    max_val = max(filtered_values)
                    distance = max([x for x in filtered_values if x != max_val], default=0)

                result[side_to_check] = int(distance / self.CELL_SIZE)

        return result

    def populate_adjacent_cells(self, result, robot_x, robot_y):
        """Populate the list of adjacent cells based on the LIDAR results."""
        existing = set(self.cell_adjacency_map.get((robot_x, robot_y), []))

        if result["up"]:
            existing.add((robot_x, robot_y + 1))
        if result["left"]:
            existing.add((robot_x - 1, robot_y))
        if result["down"]:
            existing.add((robot_x, robot_y - 1))
        if result["right"]:
            existing.add((robot_x + 1, robot_y))

        self.cell_adjacency_map[(robot_x, robot_y)] = list(existing)
        if (robot_x, robot_y) not in self.visited_cells:
            self.visited_cells.append((robot_x, robot_y))

        for adj_cell in existing:
            self.cell_adjacency_map.setdefault(adj_cell, [])
            if (robot_x, robot_y) not in self.cell_adjacency_map[adj_cell]:
                self.cell_adjacency_map[adj_cell].append((robot_x, robot_y))

    def manhattan_distance(self, a, b):
        """Calculate Manhattan distance between two points."""
        return abs(a[0] - b[0]) + abs(a[1] - b[1])

    def find_path(self, start, end):
        """Find a path."""
        adjacent_cells = self.get_map()

        open_set = [start]
        came_from = {}
        g_score = {start: 0}
        f_score = {start: self.manhattan_distance(start, end)}

        while open_set:

            current = min(open_set, key=lambda p: f_score.get(p, float('inf')))

            if current == end:
                path = [current]
                while current in came_from:
                    current = came_from[current]
                    path.append(current)
                path.reverse()
                return path

            open_set.remove(current)

            for neighbor in adjacent_cells[current]:
                tentative_g = g_score[current] + 1
                if tentative_g < g_score.get(neighbor, float('inf')):
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g
                    f_score[neighbor] = tentative_g + self.manhattan_distance(neighbor, end)
                    if neighbor not in open_set:
                        open_set.append(neighbor)

        return []

    def find_frontier_and_path(self):
        """Find frontier and path."""
        self.frontier_and_path = None

        frontiers = []
        for cell in self.cell_adjacency_map.keys():
            for neighbor in self.cell_adjacency_map.get(cell):
                if neighbor in self.unmapped_cells and neighbor != self.potential_exit:
                    frontiers.append(neighbor)
                    break

        if frontiers:
            selected_frontier = min(
                frontiers,
                key=lambda point: len(self.find_path(self.position, point)) if self.find_path(self.position, point)
                else float('inf')
            )
            self.frontier_and_path = [selected_frontier, self.find_path(self.position, selected_frontier)]

    def find_traversable_cells(self, result, robot_x, robot_y):
        """Find traversable cells."""
        if (robot_x, robot_y) not in self.traversable_cells:
            self.traversable_cells.append((robot_x, robot_y))

        for i in range(result["up"]):
            self.traversable_cells.append((robot_x, robot_y + i + 1))
        for i in range(result["left"]):
            self.traversable_cells.append((robot_x - i - 1, robot_y))
        for i in range(result["down"]):
            self.traversable_cells.append((robot_x, robot_y - i - 1))
        for i in range(result["right"]):
            self.traversable_cells.append((robot_x + i + 1, robot_y))

    def rotate_robot(self, current_pos, target_pos, orientation):
        """Calculate the angle of robot."""
        x, y = current_pos
        a, b = target_pos

        dx = a - x
        dy = b - y

        if dx == 1 and dy == 0:
            move = "right"
        elif dx == -1 and dy == 0:
            move = "left"
        elif dx == 0 and dy == 1:
            move = "up"
        elif dx == 0 and dy == -1:
            move = "down"
        else:
            move = "unknown"

        directions = ["up", "right", "down", "left"]

        facing = self.orientation_to_direction(orientation)
        facing_idx = directions.index(facing)
        move_idx = directions.index(move)

        diff = move_idx - facing_idx

        angle = diff * math.pi / 2
        if angle > math.pi:
            angle -= 2 * math.pi
        elif angle < -math.pi:
            angle += 2 * math.pi
        return angle

    def is_within_stop_range(self):
        """Check the object is within a stop range."""
        if self.lidar_data and self.lidar_data[480] < self.CELL_SIZE:
            if self.lidar_data and self.lidar_data[480] < self.CELL_SIZE / 2:
                return True
        else:
            if self.travelled_distance >= self.CELL_SIZE:
                return True
        return False

    def get_status_on_map(self, cell, robot_pos, target_pos):
        """Get cell's map content based on robot's position."""
        if robot_pos == cell:
            return "🤖"
        elif cell in self.unmapped_cells:
            return " ? "
        else:
            return "   "

    def choose_path(self):
        """Choose a next path."""
        if self.get_frontier_and_path():
            result = self.is_traversable_in_all_directions(not self.mapping_mode)
            count = sum(1 for v in result.values() if v >= 10)
            if count >= 3:
                self.forward_speed(0, 0)
                self.state = "stop"
                self.unmapped_cells = []
                self.potential_exit = None
                print("✨Final✨")
                self.draw_map(self.final_map, self.position, None)
            else:
                if not self.target_position:
                    frontier, new_path = self.get_frontier_and_path()
                    if not new_path:
                        self.forward_speed(0, 0)
                        self.state = "stop"
                        return
                    self.target_position = new_path[1] if len(new_path) > 1 else frontier

                self.turn_angle = self.rotate_robot(self.position, self.target_position, self.orientation)

                if self.turn_angle == 0:
                    self.state = "drive"
                    self.travelled_distance = 0
                else:
                    self.state = "turn"
                    self.total_rotation = 0
                    if self.turn_angle > 0:
                        self.forward_speed(1, -1)
                    else:

                        self.forward_speed(-1, 1)
        else:
            self.final_map = self.get_map().copy()

            if self.potential_exit:
                self.visited_cells.remove(self.potential_exit)
                self.potential_exit = None
                self.mapping_mode = False

    def draw_map(self, graph, robot_pos=None, target_pos=None):
        """Draw a map."""
        all_x = [x for x, y in graph.keys()]
        all_y = [y for x, y in graph.keys()]
        min_x, max_x = min(all_x), max(all_x)
        min_y, max_y = min(all_y), max(all_y)

        col_header = ""
        for x in range(min_x, max_x + 1):
            col_header += ""
        print(col_header)

        maze_rows = []
        for y in range(max_y, min_y - 1, -1):  # flipped vertically
            top_row = ""
            mid_row = ""

            for x in range(min_x, max_x + 1):
                cell = (x, y)
                neighbors = graph.get(cell, [])

                top_row += "*"
                if (x, y + 1) in neighbors or self.potential_exit_direction == (x, y + 1):
                    top_row += "   "
                else:
                    top_row += "---"

                if (x - 1, y) in neighbors or self.potential_exit_direction == (x - 1, y):
                    mid_row += " "
                else:
                    mid_row += "|"

                mid_row += self.get_status_on_map(cell, robot_pos, target_pos)

            top_row += "*"
            if self.potential_exit_direction == (x + 1, y):
                mid_row += " "
            else:
                mid_row += "|"
            maze_rows.append(top_row)
            maze_rows.append(mid_row)

        last_row = "*"
        for x in range(min_x, max_x + 1):
            if (x, y - 1) in neighbors or self.potential_exit_direction == (x, y - 1):
                last_row += "*"
            else:
                last_row += "---*"

        maze_rows.append(last_row)

        print("\n".join(maze_rows))

    def sense(self) -> None:
        """Sense."""
        self.current_time = self.robot.get_time()
        self.orientation = self.robot.get_orientation()
        self.total_rotation += self.calculate_delta_rotation(self.orientation, self.previous_orientation)
        self.previous_orientation = self.orientation
        self.lidar_data = self.robot.get_lidar_range_list()

        if self.lidar_data and self.state == "decide":
            result = self.is_traversable_in_all_directions(not self.mapping_mode)
            self.find_traversable_cells(result, self.position[0], self.position[1])

            # Update the adjacency map with the newly discovered cells
            self.populate_adjacent_cells(result, self.position[0], self.position[1])
            self.unmapped_cells = list(
                set([unmapped for unmapped in self.traversable_cells if unmapped not in self.visited_cells]))

            if self.mapping_mode:
                self.draw_map(self.get_map(), self.position, self.target_position)
                print("=================================================================")

            # find a path to the nearest frontier
            self.find_frontier_and_path()

    def plan(self) -> None:
        """Plan."""
        if not self.state:
            self.state = "decide"

        elif self.state == "decide":
            self.forward_speed(0, 0)
            self.choose_path()

        elif self.state == "drive":
            self.forward_speed(1, 1)
            if self.is_within_stop_range():
                self.forward_speed(0, 0)
                self.position = self.target_position
                self.target_position = None
                self.state = "decide"

        elif self.state == "turn":
            if abs(self.total_rotation) > abs(self.turn_angle):
                self.forward_speed(0, 0)
                self.state = "decide"

        self.align_to_orientation()
        self.update_both_wheel_speeds()
        self.previous_time = self.robot.get_time()

    def act(self) -> None:
        """Act."""
        if self.robot.get_realistic():
            self.robot.set_left_motor_torque(self.get_pid_corrected_left_wheel_speed())
            self.robot.set_right_motor_torque(self.get_pid_corrected_right_wheel_speed())
        else:
            self.robot.set_left_motor_velocity(self.left_speed)
            self.robot.set_right_motor_velocity(self.right_speed)

    def spin(self) -> None:
        """Spin."""
        self.sense()
        self.plan()
        self.act()
