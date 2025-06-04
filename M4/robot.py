"""M4."""

from __future__ import annotations

import math
import random


class Robot:
    """Turtlebot robot."""

    def __init__(self, robot: object) -> None:
        """Class initializer."""
        self.robot = robot
        self.lidar_data = None
        self.position = (0, 0)
        self.target_position = None
        self.orientation = 0.0
        self.previous_orientation = 0.0
        self.previous_direction = "up"
        self.total_rotation = 0.0
        self.turn_angle = 0
        self.initial_direction = None  # for the north

        # Calculated values for robot's environment and state
        self.traversable_cells = []  # List of all cells the robot can move to
        self.cell_adjacency_map = {}  # Adjacency map for cell connections (neighbors)
        self.final_map = {}  # Final state f the map
        self.visited_cells = []  # List of cells the robot has already visited
        self.unmapped_cells = []
        self.frontier_and_path = None
        self.northwest_cell = None
        self.potential_exit = None
        self.potential_exit_direction = None
        self.particles = None
        self.times_since_last_count_change = 0
        self.localize_initialized = False
        self.last_traversable_result = {}
        self.last_known_orientation = 0.0

        # State management
        self.state = None  # options: "decide", "turn", "drive", "stop"
        self.phase = "map"  # options: "map", "exit", "localize"
        self.travelled_distance = 0
        self.left_speed = 0.0
        self.right_speed = 0.0
        self.torque_left = 0.0
        self.torque_right = 0.0

        # Constants
        self.SPEED_FACTOR = 1.5  # change this in range 0.5-2.5 to set robot's speed. Other factors will require adjustment to PID control.
        self.MAX_TORQUE = 0.1  # maximum torque to apply to the wheel
        self.TICKS_PER_ROTATION = 508.8  # Constants for encoder ticks to velocity conversion
        self.CELL_SIZE = 0.615  # Size of a grid cell in meters

        # Constants
        self.directions = {
            'up': (0, 1),
            'down': (0, -1),
            'left': (-1, 0),
            'right': (1, 0)
        }

        # Map and its adjacent cells have absolute coordinates and poses
        # according to Y (vertical and up) and X (horizontal to right) axis.
        # Use this dict to remap absolute directions to relative for robot orientation.
        self.ROTATION = {
            "up": {"up": "up", "down": "down", "left": "left", "right": "right"},
            "right": {"up": "right", "down": "left", "left": "up", "right": "down"},
            "down": {"up": "down", "down": "up", "left": "right", "right": "left"},
            "left": {"up": "left", "down": "right", "left": "down", "right": "up"},
        }

        self.DIRECTIONS = ["up", "right", "down", "left"]

        # PID Control
        self.kp = 0.12  # Proportional gain
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
        """Set the target speeds for the robot's wheels.

        Note that higher speed multiplier will affect accuracy in calculating travelled
        distance and will require adjustments in PID control coefficients. More or less
        tested for values between 0.5 to 2.5.

        Args:
            left_target (float): Target speed for the left wheel.
            right_target (float): Target speed for the right wheel.
        """
        self.left_speed = left_target * self.SPEED_FACTOR
        self.right_speed = right_target * self.SPEED_FACTOR

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

        # ignore too big deltas, it means drive was stopped and started from the beginning
        if delta_left > 100 or delta_right > 100:
            delta_left = 0
            delta_right = 0

        self.travelled_distance += self.ticks_to_distance((delta_left + delta_right) / 2)

        # Compute current angular velocity (rad/s)
        left_angular_velocity = self.ticks_to_velocity(delta_left, delta_time)
        right_angular_velocity = self.ticks_to_velocity(delta_right, delta_time)

        self.previous_left_ticks = current_left_ticks
        self.previous_right_ticks = current_right_ticks

        # PID for left motor
        error_left = self.left_speed - left_angular_velocity
        derivative_left = (error_left - self.previous_error_left) / delta_time if delta_time > 0 else 0
        self.integral_left += error_left * delta_time
        self.integral_left = max(min(self.integral_left, self.MAX_TORQUE), -self.MAX_TORQUE)
        self.torque_left = (self.kp * error_left + self.ki * self.integral_left + self.kd * derivative_left)
        self.previous_error_left = error_left

        # PID for right motor
        error_right = self.right_speed - right_angular_velocity
        derivative_right = (error_right - self.previous_error_right) / delta_time if delta_time > 0 else 0
        self.integral_right += error_right * delta_time
        self.integral_right = max(min(self.integral_right, self.MAX_TORQUE), -self.MAX_TORQUE)
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

    def align_to_orientation(self) -> None:
        """Adjust speeds of both wheels to align robot's position to the robot orientation."""
        if self.state == "drive":
            theta = self.orientation % (2 * math.pi)

            # Calculate angle error to nearest cardinal direction
            closest_ideal = round(theta / (math.pi / 2)) * (math.pi / 2)
            error = theta - closest_ideal

            # Normalize to [-π, π] to prevent spin-over
            if error > math.pi:
                error -= 2 * math.pi
            elif error < -math.pi:
                error += 2 * math.pi

            tolerance = 0.01 * (math.pi / 2)
            if abs(error) < tolerance:
                error = 0  # No correction needed

                # check LIDAR values on left and right side of the robot to detect walls
                left_side = self.lidar_data[320]
                right_side = self.lidar_data[639]

                if left_side < self.CELL_SIZE and right_side < self.CELL_SIZE:
                    # adjust robot only when both walls are present
                    distance_error = right_side - left_side
                    # just some magic numbers associated with current SPEED_FACTOR and PID control parameters
                    if distance_error > 0.05:
                        error += 0.5
                    elif distance_error < -0.05:
                        error -= 0.5

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
        """Convert the robot's orientation angle to a cardinal direction (up, left, down, right).

        Args:
            orientation (float): The robot's orientation angle in radians.

        Returns:
            str: A string representing the robot's direction ("up", "left", "down", "right", or "unknown").
        """
        threshold = 0.78  # Tolerance for angle matching

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
        """Determine the LIDAR data index based on the robot's direction and side to check.

        Args:
            robot_direction (str): The direction the robot is facing.
            side_to_check (str): The direction to check (up, left, down, or right).

        Returns:
            int: The LIDAR data index corresponding to the side to check.
        """
        lidar_indices = {"up": 480, "left": 320, "down": 160, "right": 639}

        # Ordered directions to allow index-based rotation
        directions = ["up", "left", "down", "right"]

        # Find shift based on robot's direction
        shift = directions.index(robot_direction)

        # Compute the new index after rotation
        rotated_index = (directions.index(side_to_check) - shift) % 4

        return list(lidar_indices.values())[rotated_index]

    def is_traversable_in_all_directions(self, allow_inf=False) -> dict:
        """Check whether the robot can move in all four directions (up, left, down, right).

        Uses LIDAR data to determine the distance the robot can travel in each direction.

        Returns:
            dict: A dictionary with directions as keys and the number of traversable cells as values.
        """
        sides = ["up", "left", "down", "right"]
        result = {}

        if self.lidar_data:
            robot_direction = self.orientation_to_direction(self.orientation)

            for side_to_check in sides:
                index = self.get_lidar_index_for_direction(robot_direction, side_to_check)
                # only back sensor requires extended range of readings, other sensors provide proper coverage
                lidar_range = self.lidar_data[index - 11: index + 11] if index == 160 \
                    else self.lidar_data[index - 2: index + 2]
                filtered_values = [v for v in lidar_range if not math.isinf(v) and not math.isnan(v)]
                wide_infinity_values = [v for v in self.lidar_data[index - 50: index + 50] if
                                        not math.isinf(v) and not math.isnan(v)]

                if len(wide_infinity_values) == 0 and not allow_inf:
                    # scanning 100 LIDAR values - if all of them gives infinity then robot stands at the edge of the maze (exit is on this side)
                    if self.phase == "map":
                        # we can detect potential exit during mapping phase
                        self.potential_exit = self.position

                        # will record direction of the exit to properly show it on the map
                        # not needed for M4 but decided to keep it here for better map visualization
                        if side_to_check == "up":
                            self.potential_exit_direction = (self.position[0], self.position[1] + 1)
                        if side_to_check == "left":
                            self.potential_exit_direction = (self.position[0] - 1, self.position[1])
                        if side_to_check == "down":
                            self.potential_exit_direction = (self.position[0], self.position[1] - 1)
                        if side_to_check == "right":
                            self.potential_exit_direction = (self.position[0] + 1, self.position[1])

                    # to prevent exiting the maze at mapping phase we mark this size as a wall
                    distance = 0
                elif not filtered_values:
                    # All values in the observed direction are infinite or NaN
                    # set 10 cell size indicating exit from maze on this side
                    distance = 10 * self.CELL_SIZE if allow_inf else 1
                else:
                    # Get the maximum distance in the LIDAR data, ignoring extremes values
                    max_val = max(filtered_values)
                    distance = max([x for x in filtered_values if max_val - x < 0.1], default=0)

                # Convert the LIDAR distance into the number of cells the robot can move
                result[side_to_check] = int(distance / self.CELL_SIZE)

        return result

    def populate_adjacent_cells(self, result, robot_x, robot_y):
        """Populate the list of adjacent cells based on the LIDAR results.

        Adds neighboring cells to the adjacency map and ensures bidirectional mapping.

        Args:
            result (dict): The dictionary containing the distances the robot can move in each direction.
            robot_x (int): The current x-coordinate of the robot.
            robot_y (int): The current y-coordinate of the robot.
        """
        existing = set(self.cell_adjacency_map.get((robot_x, robot_y), []))

        if result["up"]:
            existing.add((robot_x, robot_y + 1))
        if result["left"]:
            existing.add((robot_x - 1, robot_y))
        if result["down"]:
            existing.add((robot_x, robot_y - 1))
        if result["right"]:
            existing.add((robot_x + 1, robot_y))

        # Store the adjacent cells in the map
        self.cell_adjacency_map[(robot_x, robot_y)] = list(existing)
        if (robot_x, robot_y) not in self.visited_cells:
            self.visited_cells.append((robot_x, robot_y))

        # Ensure bidirectional mapping by adding the current cell as an adjacent cell to its neighbors
        for adj_cell in existing:
            self.cell_adjacency_map.setdefault(adj_cell, [])
            if (robot_x, robot_y) not in self.cell_adjacency_map[adj_cell]:
                self.cell_adjacency_map[adj_cell].append((robot_x, robot_y))

    def manhattan_distance(self, a, b):
        """Calculate Manhattan distance between two points."""
        return abs(a[0] - b[0]) + abs(a[1] - b[1])

    def find_path(self, start, end):
        """A-star algorithm to find a path between start and end points."""
        adjacent_cells = self.get_map()

        open_set = [start]
        came_from = {}
        g_score = {start: 0}
        f_score = {start: self.manhattan_distance(start, end)}

        while open_set:
            # Find the node in open_set with the lowest f_score
            current = min(open_set, key=lambda p: f_score.get(p, float('inf')))

            if current == end:
                # Reconstruct path
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

        return []  # No path found

    def find_frontier_and_path(self):
        """Identify the next nearest frontier for exploration and calculate the path to reach it."""
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
                key=lambda point: len(self.find_path(self.position, point)) if self.find_path(self.position,
                                                                                              point) else float('inf')
            )

            self.frontier_and_path = [selected_frontier, self.find_path(self.position, selected_frontier)]

    def find_traversable_cells(self, result, robot_x, robot_y):
        """Identify all traversable cells based on found distances."""
        if (robot_x, robot_y) not in self.traversable_cells:
            self.traversable_cells.append((robot_x, robot_y))

        # Generate a list of potential unmapped cells based on the distances in each direction
        for i in range(result["up"]):
            self.traversable_cells.append((robot_x, robot_y + i + 1))
        for i in range(result["left"]):
            self.traversable_cells.append((robot_x - i - 1, robot_y))
        for i in range(result["down"]):
            self.traversable_cells.append((robot_x, robot_y - i - 1))
        for i in range(result["right"]):
            self.traversable_cells.append((robot_x + i + 1, robot_y))

    def get_rotation_degrees(self, current_pos, target_pos, orientation):
        """Calculate the angle to turn the robot to the direction of the target."""
        x, y = current_pos
        a, b = target_pos

        dx = a - x
        dy = b - y

        # Determine movement direction
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

        # Define directions in clockwise order
        directions = ["up", "right", "down", "left"]

        # Get indexes
        facing = self.orientation_to_direction(orientation)
        facing_idx = directions.index(facing)
        move_idx = directions.index(move)

        # Calculate angle difference (in 90° increments)
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
            # use LIDAR data for more precise stop, add 0.05 meters to let robot stop in realistic mode
            stop_distance = self.CELL_SIZE / 2 + 0.05 if self.robot.get_realistic() else self.CELL_SIZE / 2
            if self.lidar_data and self.lidar_data[480] < stop_distance:
                return True
        else:
            if self.travelled_distance > self.CELL_SIZE:
                return True

        return False

    def get_cell_content(self, cell, robot_pos, target_pos):
        """Get cell's map content based on robot's position."""
        if robot_pos == cell:
            return "🤖"
        elif cell == self.potential_exit:
            return "♿ "
        elif cell in self.unmapped_cells:
            return " ? "
        else:
            return "   "

    def print_ascii_maze(self, graph, robot_pos=None, target_pos=None):
        """Use ASCII art to draw the current state of the maze and robot's position in it."""
        all_x = [x for x, y in graph.keys()]
        all_y = [y for x, y in graph.keys()]
        min_x, max_x = min(all_x), max(all_x)
        min_y, max_y = min(all_y), max(all_y)

        col_header = "     "  # left padding for row numbers
        for x in range(min_x, max_x + 1):
            col_header += f"{x:^3} "
        print(col_header)

        maze_rows = []
        for y in range(max_y, min_y - 1, -1):  # flipped vertically
            top_row = "    "  # left padding for walls
            mid_row = f"{y:>3} "  # row number label

            for x in range(min_x, max_x + 1):
                cell = (x, y)
                neighbors = graph.get(cell, [])

                top_row += "+"
                top_row += "   " if (x, y + 1) in neighbors or self.potential_exit_direction == (x, y + 1) else "---"
                mid_row += " " if (x - 1, y) in neighbors or self.potential_exit_direction == (x - 1, y) else "|"
                mid_row += self.get_cell_content(cell, robot_pos, target_pos)

            top_row += "+"
            # right wall
            mid_row += " " if self.potential_exit_direction == (x + 1, y) else "|"
            maze_rows.append(top_row)
            maze_rows.append(mid_row)

        # Bottom wall
        last_row = "    +"
        for x in range(min_x, max_x + 1):
            last_row += "   +" if (x, y - 1) in neighbors or self.potential_exit_direction == (x, y - 1) else "---+"

        maze_rows.append(last_row)
        print("\n".join(maze_rows))

    def can_traverse(self, start, direction, steps):
        """Check if cell is traversable by number of steps in the given direction."""
        current = start
        for _ in range(steps):
            dx, dy = self.directions[direction]
            next_cell = (current[0] + dx, current[1] + dy)
            if next_cell in self.cell_adjacency_map.get(current, []):
                current = next_cell
            else:
                return False
        return True

    def has_neighbor(self, cell, direction):
        """Check if cell has any neighbour in given direction."""
        dx, dy = self.directions[direction]
        neighbor = (cell[0] + dx, cell[1] + dy)
        return neighbor in self.cell_adjacency_map.get(cell, [])

    def filter_bad_particles(self):
        """Remove all particles that do not match current view of the world."""
        observed = self.is_traversable_in_all_directions()

        # remap to relative robot's orientation, e.g. "up" for right-facing robot is right in absolution position.
        direction = self.orientation_to_direction(self.orientation)
        observed = {k: observed[self.ROTATION[direction][k]] for k in ["up", "left", "down", "right"]}
        filtered_particles = []

        for particle in self.particles:
            cell = (particle["x"], particle["y"])
            orientation = particle["direction"]
            match = True

            for relative_dir, observed_steps in observed.items():
                global_dir = self.ROTATION[orientation][relative_dir]

                if observed_steps == 0:
                    # Should NOT be able to move in this direction
                    if self.has_neighbor(cell, global_dir):
                        match = False
                        break
                else:
                    # Should be able to go exactly `observed_steps` in this direction
                    if not self.can_traverse(cell, global_dir, observed_steps):
                        match = False
                        break

            if match:
                filtered_particles.append(particle)

        # count filtering attempt to prevent hanging at the same state forever
        # we will use this counter to randomize moving direction after N attempts
        # without changes in potential particles.
        if len(self.particles) != len(self.particles):
            self.times_since_last_count_change = 0
        else:
            self.times_since_last_count_change += 1

        # remove potential duplicates from dict
        seen = set()
        unique = [d for d in filtered_particles if (t := tuple(sorted(d.items()))) not in seen and not seen.add(t)]

        self.particles = unique

    def initialize_particles(self) -> None:
        """Initialize particles for localization.

        This method initializes the robot's internal representation of particles
        based on the known map information.
        Directions are categorized as: up, right, down and left, each corresponding
        to orientation of robot as 0.0 ; -1.57 ; pi an -pi ; 1.57.

        Example:
            Particles are represented as a list of dictionaries:
            [
                {"x": 2, "y": 3, "direction": "up"},
                {"x": 5, "y": 1, "direction": "down"},
                ...
            ]
        """
        self.print_ascii_maze(self.cell_adjacency_map)

        self.times_since_last_count_change = 0
        self.particles = []
        for x, y in self.traversable_cells:
            for direction in self.directions:
                # Create a particle for each cell and direction
                self.particles.append({
                    "x": x,
                    "y": y,
                    "direction": direction
                })

    def adjust_particles_for_movement(self):
        """Adjust known particles according to the new movement."""
        direction = self.orientation_to_direction(self.orientation)
        for particle in self.particles:
            dx, dy = self.directions[direction]
            particle["x"] += dx
            particle["y"] += dy

    def get_rotation_steps(self, prev_dir, curr_dir):
        """Calculate number of steps to rotate from one orientation to another."""
        prev_idx = self.DIRECTIONS.index(prev_dir)
        curr_idx = self.DIRECTIONS.index(curr_dir)
        # Compute rotation steps (mod 4 ensures it wraps correctly)
        return (curr_idx - prev_idx) % 4

    def rotate_particle_direction(self, prev_particle_dir, rotation_steps):
        """Rotate particle direction by given number of steps."""
        idx = self.DIRECTIONS.index(prev_particle_dir)
        new_idx = (idx + rotation_steps) % 4
        return self.DIRECTIONS[new_idx]

    def adjust_particles_for_rotation(self):
        """Adjust known particles according to the new orientation."""
        direction = self.orientation_to_direction(self.orientation)

        if self.previous_direction != direction:
            # detect how much robot got rotated and apply same rotation to each particle
            rotation_steps = self.get_rotation_steps(self.previous_direction, direction)
            for particle in self.particles:
                particle["direction"] = self.rotate_particle_direction(particle["direction"], rotation_steps)

            self.previous_direction = direction

    def things_went_wrong_full_stop(self):
        """Emergency stop."""
        self.set_target_speeds(0, 0)
        self.state = "stop"

    def get_northwest_traversable_cell(self, direction):
        """Find the most north-west cell in a maze."""
        if direction == "up":
            # Top-left: highest y, then smallest x
            return max(self.traversable_cells, key=lambda cell: (cell[1], -cell[0]))

        elif direction == "right":
            # Top-right: largest x (north), then highest x (west)
            return max(self.traversable_cells, key=lambda cell: (cell[0], cell[1]))

        elif direction == "down":
            # Bottom-right: min y, then max x
            return min(self.traversable_cells, key=lambda cell: (cell[1], cell[0]))

        elif direction == "left":
            # Bottom-left: min x, then min y
            return min(self.traversable_cells, key=lambda cell: (cell[0], -cell[1]))
        return None

    def drive(self):
        """Turn robot's motor and drive straight."""
        self.set_target_speeds(1, 1)
        self.state = "drive"
        self.travelled_distance = 0

    def turn(self, angle):
        """Turn robot."""
        self.state = "turn"
        self.total_rotation = 0
        if angle > 0:
            self.set_target_speeds(1, -1)
        else:
            self.set_target_speeds(-1, 1)

    def make_exit_decision(self):
        """Make a decision whether to stop, drive or rotate the robot."""
        if self.position == self.northwest_cell:
            print(f"🦊 Standing at north-west position {self.position}.")
            self.set_target_speeds(0, 0)
            print("👽 Waiting to robot teleportation to a new unknown location.")
            self.phase = "localize"
            self.state = "init"

        else:
            # drive to north-west cell
            new_path = self.find_path(self.position, self.northwest_cell)

            if not new_path:
                self.things_went_wrong_full_stop()
                return

            self.target_position = new_path[1] if len(new_path) > 1 else new_path[0]

            # should decide if to drive or turn
            self.turn_angle = self.get_rotation_degrees(self.position, self.target_position, self.orientation)

            if self.turn_angle == 0:
                self.drive()
            else:
                self.turn(self.turn_angle)

    def make_map_decision(self):
        """Make decisions during mapping phase."""
        if self.get_frontier_and_path():
            if not self.target_position:
                frontier, new_path = self.get_frontier_and_path()

                if not new_path:
                    self.things_went_wrong_full_stop()
                    return

                self.target_position = new_path[1] if len(new_path) > 1 else frontier

            # should decide if to drive or turn
            self.turn_angle = self.get_rotation_degrees(self.position, self.target_position, self.orientation)

            if self.turn_angle == 0:
                self.drive()
            else:
                self.turn(self.turn_angle)
        else:
            # switch to localize mode
            print(f"✨Final map: {self.get_map()}")

            self.final_map = self.get_map().copy()
            self.set_target_speeds(0, 0)

            print("👽 Waiting to robot teleportation to a new unknown location")
            self.phase = "localize"
            self.state = "init"

    def has_anything_changed(self, old_result, new_result, old_orientation, new_orientation):
        """Check if the adjacent environment has changed."""
        if ((old_result["up"] != new_result["up"])
                or (old_result["right"] != new_result["right"])
                or (old_result["down"] != new_result["down"])
                or (old_result["left"] != new_result["left"])
                or (abs(old_orientation - new_orientation) > 0.78)):
            return True
        return False

    def make_localize_decision(self):
        """Make decision during localization phase."""
        if not self.localize_initialized:
            if self.has_anything_changed(self.last_traversable_result, self.is_traversable_in_all_directions(),
                                         self.last_known_orientation, self.orientation):
                self.localize_initialized = True
                self.previous_direction = self.orientation_to_direction(self.orientation)
                self.initialize_particles()
                self.filter_bad_particles()

        if self.localize_initialized:
            # decide where to go next
            if not self.particles:
                self.localize_initialized = False
                self.turn_angle = math.pi / 2
                self.turn(self.turn_angle)

            elif len(self.particles) == 1:
                # position is known
                self.position = (self.particles[0]['x'], self.particles[0]['y'])
                self.northwest_cell = self.get_northwest_traversable_cell(self.initial_direction)
                print("🐸 Direct is: ", self.initial_direction, " so the north cell is ", self.northwest_cell)
                self.phase = "exit"
                self.state = "decide"

            else:
                direction = self.orientation_to_direction(self.orientation)
                result = self.is_traversable_in_all_directions()

                if result[direction] > 0:
                    self.drive()
                else:
                    if self.times_since_last_count_change > 3:
                        self.turn_angle = random.choice([-1, 1]) * math.pi / 2
                    else:
                        self.turn_angle = math.pi / 2
                    self.turn(self.turn_angle)

    def sense(self) -> None:
        """Gather sensor data."""
        self.current_time = self.robot.get_time()
        self.orientation = self.robot.get_orientation()
        self.total_rotation += self.calculate_delta_rotation(self.orientation, self.previous_orientation)
        self.previous_orientation = self.orientation
        self.lidar_data = self.robot.get_lidar_range_list()

        if not self.initial_direction:
            self.initial_direction = self.orientation_to_direction(self.orientation)

        if self.lidar_data and self.phase == "map" and self.state == "decide":
            result = self.is_traversable_in_all_directions(self.phase == "exit")
            self.find_traversable_cells(result, self.position[0], self.position[1])

            # Update the adjacency map with the newly discovered cells
            self.populate_adjacent_cells(result, self.position[0], self.position[1])
            self.unmapped_cells = list(
                set([unmapped for unmapped in self.traversable_cells if unmapped not in self.visited_cells]))

            print("------Sense------")
            self.print_ascii_maze(self.get_map(), self.position, self.target_position)
            print(f"Position: {self.position}")

            self.find_frontier_and_path()

    def plan(self) -> None:
        """Plan the robot's actions in the simplest possible structure."""
        # Initial setup
        if not self.state:
            self.phase = "map"
            self.state = "decide"

        # State handlers
        elif self.state == "init" and self.phase == "localize":
            self._handle_init_localize()
        elif self.state == "decide":
            self._handle_decide()
        elif self.state == "drive":
            self._handle_drive()
        elif self.state == "turn":
            self._handle_turn()

        # Final updates (always runs)
        self.align_to_orientation()
        self.update_both_wheel_speeds()
        self.previous_time = self.robot.get_time()

    def _handle_init_localize(self) -> None:
        """Do Special case for initialization during localization."""
        self.visited_cells = []
        self.unmapped_cells = []
        self.phase = "localize"
        self.state = "decide"
        self.position = None
        self.localize_initialized = False
        self.last_traversable_result = self.is_traversable_in_all_directions()
        self.last_known_orientation = self.orientation

    def _handle_decide(self) -> None:
        """Make decisions based on current phase."""
        self.set_target_speeds(0, 0)
        {
            "map": self.make_map_decision,
            "localize": self.make_localize_decision,
            "exit": self.make_exit_decision
        }[self.phase]()

    def _handle_drive(self) -> None:
        """Handle driving logic."""
        self.set_target_speeds(1, 1)
        if not self.is_within_stop_range():
            return

        if self.phase in ("map", "exit"):
            self.position = self.target_position
            self.target_position = None
        elif self.phase == "localize":
            self.adjust_particles_for_movement()
            self.filter_bad_particles()

        self.set_target_speeds(0, 0)
        self.state = "decide"

    def _handle_turn(self) -> None:
        """Handle turning completion."""
        if abs(self.total_rotation) > abs(self.turn_angle):
            self.set_target_speeds(0, 0)
            self.state = "decide"
            if self.phase == "localize":
                self.adjust_particles_for_rotation()

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
