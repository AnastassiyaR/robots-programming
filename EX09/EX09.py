"""EX09."""

import math


class Robot:
    """Class for a frontier-exploring Turtlebot-like robot."""

    def __init__(self, robot: object) -> None:
        """Initialize robot state and exploration-related data structures."""
        self.robot = robot  # Interface to the robot hardware or simulator

        # Constants
        self.CELL_SIZE = 0.615  # Size of one cell in meters
        self.LIDAR_TOTAL = 640  # Total number of LIDAR points per full scan

        # Sensor and current state
        self.lidar_data = None  # Latest list of distances from LIDAR
        self.orientation = 0    # Robot's angle in radians (0 = facing up)
        self.position = None    # Current (x, y) position on the grid

        # Map and movement memory
        self.traversable_cells = []  # Cells the robot knows it can move to
        self.adjacency_map = {}      # Map of connected cells (graph)
        self.mapped_cells = set()    # All cells that have been visited

        # Frontier exploration
        self.unmapped_cells = []             # Known cells that haven't been visited yet
        self.was_part_of_frontier_path = []  # Cells that were already part of planned paths
        self.frontier_and_path = None        # Next target and how to get there

    def get_traversable_cells(self) -> list:
        """Return list of all known safe-to-enter cells."""
        return list(set(self.traversable_cells))  # Remove duplicates

    def get_unmapped_cells(self) -> list:
        """Return cells that are safe but haven’t been visited (frontier)."""
        return self.unmapped_cells

    def get_map(self) -> dict:
        """Return current adjacency map (graph of explored area)."""
        return self.adjacency_map

    def get_mapped(self):
        """Return cells that have been sensed and mapped."""
        return self.mapped_cells

    def get_frontier_and_path(self) -> list:
        """Return the current exploration goal and the path to it."""
        return self.frontier_and_path

    def manhattan_distance(self, a, b):
        """Calculate Manhattan (grid-based) distance between two points."""
        return abs(a[0] - b[0]) + abs(a[1] - b[1])

    def orientation_to_direction(self, orientation):
        """Convert orientation (in radians) into one of 4 directions."""
        threshold = 0.05  # Allow small angle errors

        if abs(orientation) < threshold:
            return "up"
        elif abs(orientation - math.pi / 2) < threshold:
            return "left"
        elif abs(orientation + math.pi) < threshold or abs(orientation - math.pi) < threshold:
            return "down"
        elif abs(orientation + math.pi / 2) < threshold:
            return "right"
        else:
            return "unknown"  # Can't determine direction

    def get_lidar_index_for_direction(self, robot_direction, side_to_check):
        """Find which LIDAR index corresponds to a direction (e.g., left of robot)."""
        self.LIDAR_TOTAL = len(self.lidar_data)  # Update LIDAR point count if needed

        # Indices for each direction assuming robot faces up
        lidar_indices = {
            "up": int(self.LIDAR_TOTAL * 0.75),
            "left": int(self.LIDAR_TOTAL * 0.5),
            "down": int(self.LIDAR_TOTAL * 0.25),
            "right": self.LIDAR_TOTAL - 1,
        }

        # Rotate directions depending on which way robot is facing
        directions = ["up", "left", "down", "right"]
        shift = directions.index(robot_direction)
        rotated_index = (directions.index(side_to_check) - shift) % 4

        return list(lidar_indices.values())[rotated_index]

    def is_traversable_in_all_directions(self) -> dict:
        """Check how many cells are free in each direction using LIDAR."""
        sides = ["up", "left", "down", "right"]
        result = {}
        if self.lidar_data:
            robot_direction = self.orientation_to_direction(self.orientation)
            for side_to_check in sides:
                index = self.get_lidar_index_for_direction(robot_direction, side_to_check)
                nearby_vals = self.lidar_data[index - 10: index + 10]  # Use a small range
                valid_vals = [v for v in nearby_vals if not math.isinf(v)]
                distance = max(valid_vals, default=0)
                result[side_to_check] = int(distance / self.CELL_SIZE)  # Convert meters to cells
        return result

    def populate_adjacent_cells(self, result, robot_x, robot_y):
        """Update map with reachable neighbor cells based on LIDAR data."""
        adjacent_cells = []
        if result["up"]:
            adjacent_cells.append((robot_x, robot_y + 1))
        if result["left"]:
            adjacent_cells.append((robot_x - 1, robot_y))
        if result["down"]:
            adjacent_cells.append((robot_x, robot_y - 1))
        if result["right"]:
            adjacent_cells.append((robot_x + 1, robot_y))

        self.adjacency_map[(robot_x, robot_y)] = adjacent_cells
        self.mapped_cells.add((robot_x, robot_y))  # Mark current cell as visited

        # Also update neighbors so they point back to this cell
        for adj_cell in adjacent_cells:
            self.adjacency_map.setdefault(adj_cell, [])
            if (robot_x, robot_y) not in self.adjacency_map[adj_cell]:
                self.adjacency_map[adj_cell].append((robot_x, robot_y))

    def find_path(self, start, end):
        """Find shortest path from start to end using A* algorithm."""
        cell_adjacency_map = self.get_map()

        nodes_to_explore = [start]
        path_origins = {}  # For building the path
        movement_costs = {start: 0}  # g cost
        total_estimated_costs = {start: self.manhattan_distance(start, end)}  # f = g + h

        while nodes_to_explore:
            # Get node with lowest estimated cost
            current = min(nodes_to_explore, key=lambda p: total_estimated_costs.get(p, float('inf')))

            if current == end:
                # Reconstruct the path from end to start
                path = [current]
                while current in path_origins:
                    current = path_origins[current]
                    path.append(current)
                return path[::-1]  # Return reversed path

            nodes_to_explore.remove(current)

            for neighbor in cell_adjacency_map.get(current, []):
                tentative_g = movement_costs[current] + 1  # Assume cost = 1 per move
                if tentative_g < movement_costs.get(neighbor, float('inf')):
                    path_origins[neighbor] = current
                    movement_costs[neighbor] = tentative_g
                    total_estimated_costs[neighbor] = tentative_g + self.manhattan_distance(neighbor, end)
                    if neighbor not in nodes_to_explore:
                        nodes_to_explore.append(neighbor)

        return []  # No path found

    def find_frontier_and_path(self):
        """Pick the closest unmapped cell and find path to reach it."""
        frontiers = [x for x in self.get_traversable_cells() if x not in self.was_part_of_frontier_path]
        if frontiers:
            selected_frontier = min(frontiers, key=lambda point: self.manhattan_distance(self.position, point))
            path = self.find_path(self.position, selected_frontier)
            self.frontier_and_path = [selected_frontier, path]
            self.was_part_of_frontier_path.extend(path)

    def find_traversable_cells(self, result, robot_x, robot_y):
        """Add reachable cells from current location based on LIDAR data."""
        self.traversable_cells.append((robot_x, robot_y))  # Include current cell

        for i in range(result["up"]):
            self.traversable_cells.append((robot_x, robot_y + i + 1))
        for i in range(result["left"]):
            self.traversable_cells.append((robot_x - i - 1, robot_y))
        for i in range(result["down"]):
            self.traversable_cells.append((robot_x, robot_y - i - 1))
        for i in range(result["right"]):
            self.traversable_cells.append((robot_x + i + 1, robot_y))

    def sense(self) -> None:
        """Sense the environment and update map and path planning."""
        self.lidar_data = self.robot.get_lidar_range_list()

        if self.lidar_data:
            self.position = self.robot.get_current_position()
            self.orientation = self.robot.get_orientation()

            if not self.was_part_of_frontier_path:
                self.was_part_of_frontier_path.append(self.position)

            result = self.is_traversable_in_all_directions()
            self.find_traversable_cells(result, self.position[0], self.position[1])
            self.populate_adjacent_cells(result, self.position[0], self.position[1])

            # Determine cells that are reachable but not yet visited
            self.unmapped_cells = list(set(
                [unmapped for unmapped in self.traversable_cells if unmapped not in self.mapped_cells]
            ))

            # Plan to move toward unexplored area
            self.find_frontier_and_path()

    def plan(self) -> None:
        """Plan."""
        pass

    def act(self) -> None:
        """Act."""
        pass

    def spin(self) -> None:
        """Spin."""
        self.sense()
        self.plan()
        self.act()
