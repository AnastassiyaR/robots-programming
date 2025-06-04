"""EX08: Mapping the Environment."""

import math


class Robot:
    """Turtlebot robot mapping the environment using LIDAR.

    Attributes:
        CELL_SIZE (float): Grid cell width in meters (0.615m per cell).
        robot (object): Interface to the physical robot.
        lidar_data (list): Raw LIDAR distance readings.
        orientation (float): Current facing direction in radians.
        traversable_cells (list): All known reachable grid coordinates.
        map (dict): Adjacency dictionary representing the environment graph.
        visited_map (dict): Tracks visited cells and their neighbors.
    """

    CELL_SIZE = 0.615  # Grid cell width in meters

    def __init__(self, robot: object) -> None:
        """Initialize the robot mapping interface.

        Args:
            robot (object): A Turtlebot-like robot interface instance.
        """
        self.robot = robot               # Physical robot interface
        self.lidar_data = None           # Raw LIDAR measurements
        self.orientation = 0             # Current facing direction (radians)

        # Mapping data structures
        self.traversable_cells = []     # All known reachable cells
        self.map = {}                    # Adjacency map (graph representation)
        self.visited_map = {}            # Visited cells tracking

    # === Public Interface ===
    def get_traversable_cells(self) -> list:
        """Get all known traversable grid cells.

        Returns:
            list: Unique (x,y) coordinates of reachable cells.
        """
        return list(set(self.traversable_cells))

    def get_unmapped_cells(self) -> list:
        """Get traversable cells not yet visited by the robot.

        Returns:
            list: Unvisited (x,y) coordinates from traversable cells.
        """
        return [cell for cell in self.traversable_cells if cell not in self.visited_map]

    def get_map(self) -> dict:
        """Get the robot's internal map as an adjacency dictionary.

        Returns:
            dict: Mapping of each cell to its connected neighbors.
        """
        return self.map

    def sense(self) -> None:
        """Perform environment sensing using LIDAR and update maps.

        1. Gets current LIDAR readings
        2. Updates robot position and orientation
        3. Calculates visible traversable cells
        4. Updates internal maps with new information
        """
        self.lidar_data = self.robot.get_lidar_range_list()
        if not self.lidar_data:
            return  # Skip if no LIDAR data available

        # Get current position and orientation
        x, y = self.robot.get_current_position()
        self.orientation = self.robot.get_orientation()

        # Calculate visible cells in all directions
        direction_map = self._get_traversable_directions()
        newly_detected = self._calculate_visible_cells(x, y, direction_map)

        # Update traversable cells with new discoveries
        self.traversable_cells.extend(newly_detected)
        self.traversable_cells.append((x, y))  # Current position is always traversable

        # Update adjacency relationships
        self._update_map_with_neighbors(x, y, direction_map)

    def plan(self) -> None:
        """Plan actions based on sensed data (to be implemented)."""
        pass

    def act(self) -> None:
        """Execute planned actions (to be implemented)."""
        pass

    def spin(self) -> None:
        """Execute one complete sense-plan-act cycle."""
        self.sense()
        self.plan()
        self.act()

    # === Internal Helpers ===
    def _get_direction(self) -> str:
        """Convert orientation angle to cardinal direction.

        Returns:
            str: Current facing direction ('up', 'down', 'left', 'right').
        """
        angle = self.orientation
        threshold = 0.05  # Angle tolerance for direction detection

        if abs(angle) < threshold:
            return "up"
        if abs(angle - math.pi / 2) < threshold:
            return "left"
        if abs(angle + math.pi) < threshold or abs(angle - math.pi) < threshold:
            return "down"
        if abs(angle + math.pi / 2) < threshold:
            return "right"
        return "unknown"

    def _get_traversable_directions(self) -> dict:
        """Calculate traversable distances in all cardinal directions.

        Returns:
            dict: Number of traversable cells in each direction.
        """
        direction_labels = ["up", "left", "down", "right"]
        lidar_indices = {"up": 480, "left": 320, "down": 160, "right": 639}
        direction = self._get_direction()
        result = {}

        for target in direction_labels:
            # Get relevant LIDAR indices for this direction
            idx = self.calculate_lidar_index(direction, target, lidar_indices, direction_labels)

            # Process LIDAR data (ignore infinite values)
            valid_readings = [d for d in self.lidar_data[idx - 10: idx + 10] if not math.isinf(d)]
            distance = max(valid_readings, default=0)  # Take farthest valid reading

            # Convert distance to number of cells
            result[target] = int(distance / self.CELL_SIZE)

        return result

    def calculate_lidar_index(self, orientation: str, check_side: str, index_map: dict, labels: list) -> int:
        """Calculate LIDAR array index for a given direction relative to robot orientation.

        Args:
            orientation: Current facing direction ('up', 'down', etc.)
            check_side: Direction to calculate index for
            index_map: Base indices for each cardinal direction
            labels: Ordered list of direction labels

        Returns:
            int: Adjusted LIDAR array index for the requested direction
        """
        shift = labels.index(orientation)
        rotated_index = (labels.index(check_side) - shift) % 4
        return list(index_map.values())[rotated_index]

    def _calculate_visible_cells(self, x: int, y: int, directions: dict) -> list:
        """Generate coordinates of visible cells in all directions.

        Args:
            x: Current x position
            y: Current y position
            directions: Number of traversable cells in each direction

        Returns:
            list: (x,y) coordinates of all visible cells
        """
        cells = []
        # Calculate cells in each direction
        for i in range(directions["up"]):
            cells.append((x, y + i + 1))
        for i in range(directions["left"]):
            cells.append((x - i - 1, y))
        for i in range(directions["down"]):
            cells.append((x, y - i - 1))
        for i in range(directions["right"]):
            cells.append((x + i + 1, y))
        return cells

    def _update_map_with_neighbors(self, x: int, y: int, directions: dict) -> None:
        """Update adjacency maps with neighboring cell relationships.

        Args:
            x: Current x position
            y: Current y position
            directions: Number of traversable cells in each direction
        """
        neighbors = []
        # Add immediate neighbors (1 cell away in each direction)
        if directions["up"]:
            neighbors.append((x, y + 1))
        if directions["left"]:
            neighbors.append((x - 1, y))
        if directions["down"]:
            neighbors.append((x, y - 1))
        if directions["right"]:
            neighbors.append((x + 1, y))

        # Update maps with new connections
        self.map[(x, y)] = neighbors
        self.visited_map[(x, y)] = neighbors

        # Ensure bidirectional connections
        for cell in neighbors:
            self.map.setdefault(cell, [])
            if (x, y) not in self.map[cell]:
                self.map[cell].append((x, y))
