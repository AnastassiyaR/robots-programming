import numpy as np
import time
import math

# Try to import OpenCV
try:
    import cv2

    OPENCV_AVAILABLE = True
except ImportError:
    OPENCV_AVAILABLE = False
    print("OpenCV not available - using simpler color detection")


class Robot:
    """Complete robot implementation with fixed color detection."""

    def __init__(self, robot: object) -> None:
        self.robot = robot
        self.debug = True
        self.show_vision = True  # Enable vision debug windows

        # Motion parameters
        self.base_turn_speed = 0.6
        self.base_move_speed = 2.0
        self.slow_move_speed = 0.5

        # Target thresholds
        self.stop_threshold = 50000
        self.slow_threshold = 30000
        self.min_distance = 0.3

        # Target management
        self.target_color = "red"
        self.color_order = ["red", "blue", "cyan"]
        self.current_target = None
        self.approached_targets = []

        # Motor states
        self.left_motor_speed = 0
        self.right_motor_speed = 0

        # Scanning state
        self.scan_start_angle = 0
        self.scan_complete = False
        self.last_scan_check = 0
        self.scan_check_interval = 0.2

        # Color detection ranges (HSV)
        self.color_ranges = {
            'red': {
                'lower': [0, 150, 50],
                'upper': [10, 255, 255],
                'lower2': [160, 150, 50],
                'upper2': [180, 255, 255],
                'display_color': (0, 0, 255)  # BGR for red
            },
            'blue': {
                'lower': [100, 150, 50],
                'upper': [140, 255, 255],
                'display_color': (255, 0, 0)  # BGR for blue
            },
            'cyan': {
                'lower': [85, 150, 50],
                'upper': [95, 255, 255],
                'display_color': (255, 255, 0)  # BGR for cyan
            }
        }

    def log(self, message):
        """Debug logging function."""
        if self.debug:
            print(f"[DEBUG {time.time():.1f}] {message}")

    def get_current_angle(self):
        """Estimate current angle based on wheel movement."""
        left_ticks = self.robot.get_left_motor_encoder_ticks()
        right_ticks = self.robot.get_right_motor_encoder_ticks()
        return (right_ticks - left_ticks) * 0.001

    def detect_color(self, np_image, color):
        """Fixed color detection that always returns 4 values."""
        if not OPENCV_AVAILABLE:
            return None, 0, None, None

        hsv = cv2.cvtColor(np_image, cv2.COLOR_RGB2HSV)
        color_range = self.color_ranges[color]

        # Primary mask
        mask = cv2.inRange(hsv, np.array(color_range['lower']), np.array(color_range['upper']))

        # Secondary mask for red
        if 'lower2' in color_range:
            mask2 = cv2.inRange(hsv, np.array(color_range['lower2']), np.array(color_range['upper2']))
            mask = cv2.bitwise_or(mask, mask2)

        # Find contours
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return None, 0, None, mask

        # Get largest valid contour
        largest = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(largest)

        if w == 0 or h / w < 1.5:  # Not pole-like
            return None, 0, None, mask

        return (x + w // 2, y + h // 2), w * h, (x, y, x + w, y + h), mask

    def sense(self) -> None:
        """Fixed sensing with proper return value handling."""
        try:
            if time.time() - self.last_scan_check < self.scan_check_interval:
                return

            self.last_scan_check = time.time()
            self.lidar_data = self.robot.get_lidar_range_list()
            self.image = self.robot.get_camera_rgb_image()

            if self.image is None:
                self.log("No image available")
                return

            np_image = np.asarray(self.image, dtype=np.uint8)
            if np_image.ndim != 3:
                return

            # Get all 4 return values
            center, size, bbox, mask = self.detect_color(np_image, self.target_color)

            if center is not None:
                self.current_target = {
                    'center': center,
                    'size': size,
                    'bbox': bbox,
                    'color': self.target_color
                }
                self.scan_complete = True
                self.log(f"Found {self.target_color} (size: {size})")
            else:
                self.current_target = None
                current_angle = self.get_current_angle()
                if abs(current_angle - self.scan_start_angle) >= 2 * math.pi:
                    self.scan_complete = True
                    self.log("Scan complete - no target found")

            # Visualization
            if self.show_vision and OPENCV_AVAILABLE and mask is not None:
                self._draw_vision(np_image, mask)

        except Exception as e:
            self.log(f"Sense error: {str(e)}")

    def _draw_vision(self, np_image, mask):
        """Draw real-time vision debug."""
        debug_img = cv2.cvtColor(np_image, cv2.COLOR_RGB2BGR)

        # Draw detection
        if self.current_target:
            x1, y1, x2, y2 = self.current_target['bbox']
            color = self.color_ranges[self.target_color]['display_color']
            cv2.rectangle(debug_img, (x1, y1), (x2, y2), color, 2)
            cv2.putText(debug_img, f"{self.target_color} ({self.current_target['size']}px)",
                        (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        # Show windows
        cv2.imshow("Camera View", debug_img)
        cv2.imshow("Color Mask", mask)
        cv2.waitKey(1)

    def plan(self) -> None:
        """Decision making logic."""
        if self.current_target:
            self.approach_target()
        elif self.scan_complete:
            self.switch_color()
        else:
            self.search_pattern()

    def approach_target(self):
        """Approach target with LIDAR safety checks."""
        target = self.current_target
        img_center = self.image.shape[1] // 2
        error = (target['center'][0] - img_center) / img_center

        # Get front distance
        front_dist = float('inf')
        if self.lidar_data:
            sector = self.lidar_data[len(self.lidar_data) // 3:2 * len(self.lidar_data) // 3]
            front_dist = min([d for d in sector if d > 0] or [front_dist])

        # Control logic
        if target['size'] > self.stop_threshold or front_dist < self.min_distance:
            self.left_motor_speed = self.right_motor_speed = 0
            self.approached_targets.append(target)
            self.switch_color()
        elif target['size'] > self.slow_threshold or front_dist < self.min_distance + 0.2:
            speed = self.slow_move_speed
            steer = 1.0 * error
            self.left_motor_speed = speed + steer
            self.right_motor_speed = speed - steer
        else:
            speed = self.base_move_speed
            steer = 1.5 * error
            self.left_motor_speed = speed + steer
            self.right_motor_speed = speed - steer

    def search_pattern(self):
        """Systematic search behavior."""
        self.left_motor_speed = self.base_turn_speed
        self.right_motor_speed = -self.base_turn_speed

        if self.scan_start_angle == 0:
            self.scan_start_angle = self.get_current_angle()
            self.scan_complete = False
            self.log(f"Scanning for {self.target_color}")

    def switch_color(self):
        """Cycle through target colors."""
        current_idx = self.color_order.index(self.target_color)
        next_idx = (current_idx + 1) % len(self.color_order)
        self.target_color = self.color_order[next_idx]
        self.scan_start_angle = 0
        self.scan_complete = False
        self.current_target = None
        self.log(f"Switched to {self.target_color}")

    def act(self) -> None:
        """Execute motor commands."""
        max_speed = 3.0
        left = np.clip(self.left_motor_speed, -max_speed, max_speed)
        right = np.clip(self.right_motor_speed, -max_speed, max_speed)
        self.robot.set_left_motor_velocity(left)
        self.robot.set_right_motor_velocity(right)

    def spin(self) -> None:
        """Main control loop."""
        self.sense()
        self.plan()
        self.act()
