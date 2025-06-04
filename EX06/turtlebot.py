"""
To use this helper file, please add it to the same directory as your EX06.py file along with the data files.

The directory should have the following files in it:
.
├── EX06.py
├── turtlebot.py
├── local.py
└── {Data_file}.pkl

To use with data file set the value in local.py main to the task related .pkl file you want to use.
"""


class Robot:
    def __init__(self):
        self.fov = None
        self.image = None
        self.time = None
        self.orientation = None
        self.range_list = None
        self.point_cloud = None
        self.enc_l = None
        self.enc_r = None

    def get_time(self):
        return self.time

    def get_orientation(self):
        return self.orientation

    def get_camera_rgb_image(self):
        return self.image

    def get_camera_params(self):
        return self.fov

    def get_lidar_range_list(self):
        return self.range_list

    def get_lidar_point_cloud(self):
        return self.point_cloud

    def get_left_motor_encoder_ticks(self):
        return self.enc_l

    def get_right_motor_encoder_ticks(self):
        return self.enc_r

    def _set_data(self, data_at_time_step: list) -> None:
        self.time = data_at_time_step[0]
        self.orientation = data_at_time_step[1]
        self.enc_l = data_at_time_step[2]
        self.enc_r = data_at_time_step[3]
        self.range_list = data_at_time_step[4]
        self.point_cloud = data_at_time_step[5]
        self.image = data_at_time_step[6]
        self.fov = data_at_time_step[7]
