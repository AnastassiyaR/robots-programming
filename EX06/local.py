"""
To use this helper file, please add it to the same directory as your EX06.py file along with the data files.

The directory should have the following files in it:
.
├── EX06.py
├── turtlebot.py
├── {Data_file}.pkl
└── local.py

To use with data file set the value in local.py main to the task related .pkl file you want to use.
"""


import pickle
import EX06 as student
import turtlebot as turtlebot


def load_dataset(filename):
    with open(filename, 'rb') as file:
        loaded_data = pickle.load(file)
    return loaded_data


if __name__ == "__main__":
    ex_data = load_dataset("data_1.pkl")
    turtlebot_interface = turtlebot.Robot()
    student_robot = student.Robot(turtlebot_interface)
    for i, data_at_time_step in enumerate(ex_data):
        turtlebot_interface._set_data(data_at_time_step)
        student_robot.spin()
