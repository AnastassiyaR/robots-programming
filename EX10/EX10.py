"""EX10."""

import math
from typing import List, Dict, Tuple, Optional, Any


class Robot:
    """
    Represent navigation and localization logic for a Turtlebot-style robot.

    The robot uses a known map, LIDAR, encoders, and heading to localize its position.
    """

    def __init__(self, robot: object) -> None:
        """
        Initialize the Robot object with default state.

        Args:
            robot (object): A hardware or simulation interface providing sensor, encoder, and time data.
        """
        # Hardware/simulation interface
        self.robot = robot

        # Map-related properties
        self.free_cells: Optional[List[Tuple[int, int]]] = None  # where can move
        self.adjacency_map: Optional[Dict[Tuple[int, int], List[Tuple[int, int]]]] = None  # Graph of connected cells

        # Sensor readings
        self.lidar_ranges: Optional[List[float]] = None  # Current LIDAR distance measurements
        self.heading: Optional[float] = None  # Current orientation in radians

        # State tracking
        self.last_direction: str = "up"  # Last known cardinal direction
        self.position_hypotheses: Optional[List[Dict[str, Any]]] = None  # Possible robot states (particles)
        self.distance_traveled: float = 0.0  # Accumulated distance since last cell transition
        self.last_left_ticks: int = 0  # Previous left encoder tick count
        self.last_right_ticks: int = 0  # Previous right encoder tick count
        self.current_time: float = self.robot.get_time() or 0.0  # Current timestamp
        self.previous_time: float = self.current_time  # Previous measurement timestamp

        # Movement vectors for each cardinal direction (dx, dy)
        self.move_vectors: Dict[str, Tuple[int, int]] = {
            'up': (0, 1),     # +Y direction
            'down': (0, -1),  # -Y direction
            'left': (-1, 0),   # -X direction
            'right': (1, 0)    # +X direction
        }

        # Direction transformation matrix for relative directions
        # Maps (current_direction × relative_direction) → absolute_direction
        self.ROTATION: Dict[str, Dict[str, str]] = {
            "up": {"up": "up", "down": "down", "left": "left", "right": "right"},
            "right": {"up": "right", "down": "left", "left": "up", "right": "down"},
            "down": {"up": "down", "down": "up", "left": "right", "right": "left"},
            "left": {"up": "left", "down": "right", "left": "down", "right": "up"},
        }

        # Hardware constants
        self.TICKS_PER_ROTATION: float = 508.8  # Encoder ticks per wheel revolution
        self.CELL_SIZE: float = 0.615
        self.WHEEL_DIAMETER: float = 0.0715  # Wheel diameter in meters

    # The robot doesn't know where it is, so it creates a lot of hypotheses (particles).
    def initialize_particles(self) -> None:
        """
        Generate initial hypotheses by placing particles in all free cells facing all possible directions.

        Each particle represents a possible (x, y, direction) state hypothesis.
        """
        if not self.free_cells or not self.adjacency_map:
            raise ValueError("Must set free_cells and adjacency_map before initializing particles.")

        # Create particles for every possible (cell × direction) combination
        self.position_hypotheses = []
        for x, y in self.free_cells:
            for direction in self.move_vectors:
                self.position_hypotheses.append({
                    "x": x,
                    "y": y,
                    "direction": direction
                })

    def set_known_map_traversable_cells(self, free_cells: List[Tuple[int, int]]) -> None:
        """Set the list of traversable cells on the map."""
        self.free_cells = free_cells

    def set_known_map_neighbour_list(self, adjacency_map: Dict[Tuple[int, int], List[Tuple[int, int]]]) -> None:
        """Set the adjacency graph indicating connectivity between cells."""
        self.adjacency_map = adjacency_map

    def get_potential_positions(self) -> List[Dict[str, Any]]:
        """Return the current list of position hypotheses (particles)."""
        return self.position_hypotheses if self.position_hypotheses else []

    def orientation_to_direction(self, heading: float) -> str:
        """
        Convert a heading in radians to a cardinal direction (with tolerance).

        Args:
            heading: Orientation angle in radians

        Returns:
            Cardinal direction ("up", "down", "left", "right") or "unknown"
        """
        threshold = 0.05  # Angular tolerance for direction classification
        if abs(heading) < threshold:
            return "up"
        elif abs(heading - math.pi / 2) < threshold:
            return "left"
        elif abs(heading + math.pi) < threshold or abs(heading - math.pi) < threshold:
            return "down"
        elif abs(heading + math.pi / 2) < threshold:
            return "right"
        return "unknown"

    def get_lidar_index_for_direction(self, current_facing: str, relative_check: str) -> int:
        """
        Return the LIDAR index corresponding to a given relative direction.

        LIDAR has 640 values (0-639), covering 180° field of view.
        Front is index 480, left is 320, back is 160, right is 639.

        Args:
            current_facing: Robot's current cardinal direction
            relative_check: Direction to check relative to current facing

        Returns:
            LIDAR array index for the specified direction
        """
        # Base indices for each direction when facing up
        lidar_indices = {"up": 480, "left": 320, "down": 160, "right": 639}
        directions = ["up", "left", "down", "right"]

        # Calculate rotational offset
        shift = directions.index(current_facing)
        rotated_index = (directions.index(relative_check) - shift) % 4
        return list(lidar_indices.values())[rotated_index]

    def is_traversable_in_all_directions(self) -> Dict[str, int]:
        """
        Estimate how many cells are free in each direction based on LIDAR ranges.

        Returns:
            Dictionary mapping directions to number of free cells
        """
        result = {}
        if self.lidar_ranges and self.heading is not None:
            current_facing = self.orientation_to_direction(self.heading)
            for side in ["up", "left", "down", "right"]:
                # Get LIDAR index for this relative direction
                idx = self.get_lidar_index_for_direction(current_facing, side)

                # Average nearby readings to reduce noise
                values = [v for v in self.lidar_ranges[idx - 10: idx + 10] if not math.isinf(v)]
                distance = max(values) if values else 0

                # Convert distance to cell count
                result[side] = int(distance / self.CELL_SIZE)
        return result

    # It translates the "ticks" (clicks) of the sensor on the wheel into meters traveled by the robot.
    def ticks_to_distance(self, ticks: int) -> float:
        """
        Convert encoder ticks to linear distance in meters.

        Args:
            ticks: Encoder tick count

        Returns:
            Distance traveled in meters
        """
        circumference = math.pi * self.WHEEL_DIAMETER
        rotations = ticks / self.TICKS_PER_ROTATION
        return rotations * circumference

    def find_travelled_distance(self) -> None:
        """Update the distance traveled using motor encoder readings."""
        # Get current encoder values
        current_left = self.robot.get_left_motor_encoder_ticks()
        current_right = self.robot.get_right_motor_encoder_ticks()

        # Calculate delta from last reading
        delta_left = current_left - self.last_left_ticks
        delta_right = current_right - self.last_right_ticks

        # Update accumulated distance (average of both wheels)
        self.distance_traveled += self.ticks_to_distance((delta_left + delta_right) / 2)

        # Store current values for next iteration
        self.last_left_ticks = current_left
        self.last_right_ticks = current_right

    def can_traverse(self, start: Tuple[int, int], direction: str, steps: int) -> bool:
        """
        Check if a path exists in the given direction for specified steps.

        Args:
            start: Starting (x,y) cell
            direction: Cardinal direction to check
            steps: Number of cells to check

        Returns:
            True if path exists, False otherwise
        """
        current = start
        for _ in range(steps):
            dx, dy = self.move_vectors[direction]
            next_cell = (current[0] + dx, current[1] + dy)
            if next_cell in self.adjacency_map.get(current, []):
                current = next_cell
            else:
                return False
        return True

    def has_neighbor(self, cell: Tuple[int, int], direction: str) -> bool:
        """
        Check if a neighboring cell exists in the specified direction.

        Args:
            cell: Base (x,y) cell
            direction: Neighbor direction to check

        Returns:
            True if neighbor exists, False otherwise
        """
        dx, dy = self.move_vectors[direction]
        return (cell[0] + dx, cell[1] + dy) in self.adjacency_map.get(cell, [])

    # This method filters out the robot's incorrect guesses about its location.
    def filter_bad_particles(self) -> None:
        """
        Filter out hypotheses that do not match observed LIDAR structure.

        Compares each particle's expected visibility with actual LIDAR readings.
        """
        if not self.position_hypotheses or not self.adjacency_map:
            return

        # Get observed free cells in each direction
        observed = self.is_traversable_in_all_directions()
        if not observed:
            return

        # Align observations with robot's current facing
        current_facing = self.orientation_to_direction(self.heading)
        rotated = {k: observed[self.ROTATION[current_facing][k]] for k in ["up", "left", "down", "right"]}

        # Filter particles
        valid = []
        for particle in self.position_hypotheses:
            cell = (particle["x"], particle["y"])
            facing = particle["direction"]
            keep = True

            # Check each direction's consistency
            for rel, steps in rotated.items():
                absolute = self.ROTATION[facing][rel]
                # Case 1: LIDAR sees wall but particle expects path
                if steps == 0 and self.has_neighbor(cell, absolute):
                    keep = False
                    break
                # Case 2: LIDAR sees path but particle expects wall
                if steps > 0 and not self.can_traverse(cell, absolute, steps):
                    keep = False
                    break

            if keep:
                valid.append(particle)

        self.position_hypotheses = valid
        print(f"Remaining particles: {len(valid)}")

    def adjust_for_movement(self) -> None:
        """
        Move particles one cell in the direction the robot moved.

        Updates all hypotheses based on estimated movement.
        """
        if not self.position_hypotheses:
            return

        moved = []
        for p in self.position_hypotheses:
            original = p["direction"]
            # Adjust for any discrepancy between particle and actual facing
            actual = self.ROTATION[original][self.last_direction] if original != self.last_direction else original
            dx, dy = self.move_vectors[actual]
            moved.append({
                "x": p["x"] + dx,
                "y": p["y"] + dy,
                "direction": actual
            })

        self.position_hypotheses = moved
        self.distance_traveled = 0  # Reset after movement

    def sense(self) -> None:
        """Sense."""
        # Update timing
        self.current_time = self.robot.get_time()
        self.find_travelled_distance()

        # Update orientation
        self.heading = self.robot.get_orientation()
        current_dir = self.orientation_to_direction(self.heading)

        # Handle direction changes
        if self.last_direction != current_dir:
            self.distance_traveled = 0  # Reset distance counter
            print(f"Direction changed from {self.last_direction} to {current_dir}")
            self.last_direction = current_dir

        # Get LIDAR data
        self.lidar_ranges = self.robot.get_lidar_range_list()

        if self.lidar_ranges:
            # Debug output
            print(f"Time: {self.current_time}")
            print(f"Distance: {self.distance_traveled:.2f}")
            print(f"Heading: {self.heading:.2f}")
            print(f"Dir: {current_dir}")

            # Check if moved approximately one cell
            if abs(self.distance_traveled - self.CELL_SIZE) < 0.1:
                self.adjust_for_movement()

            # Update particle filter
            self.filter_bad_particles()

        self.previous_time = self.current_time

    def plan(self) -> None:
        """Planning phase (stub)."""
        pass

    def act(self) -> None:
        """Act."""
        pass

    def spin(self) -> None:
        """Spin."""
        self.sense()
        self.plan()
        self.act()
