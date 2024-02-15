import numpy as np
from haptic_exploration.util import Pose, GlanceAreaBB


MYRMEX_BODY = "myrmex_body"
MYRMEX_MOCAP_BODY = "myrmex_mocap_body"


glance_sum_pressure_threshold = 6000
glance_cell_pressure_threshold = 2000
glance_velocity_threshold = 0.001
glance_velocity_threshold_count = 15
glance_mocap_distance_threshold = 0.03
mocap_velocity = 0.1


### SIMPLE SIM CONFIG ###

simple_inactive_object_x = 0.4
simple_glance_object_pose = Pose(np.array([0, 0, 0.36]), np.array([0, 0, 0, 0]))

basic_objects_glance_area = GlanceAreaBB((-0.075, 0.075), (-0.075, 0.075), (0.39, 0.54))
basic_objects_z_buffer = 0.05

basic_objects = {
    0: "cube",
    1: "circle1",
    2: "triangle",
    3: "circle2"
}


### YCB SIM CONFIG ###

ycb_inactive_object_y = -0.4
ycb_glance_object_pose = Pose(np.array([0, 0, 0.36]), np.array([0, 0, 0, 0]))

ycb_objects_glance_area = GlanceAreaBB((-0.075, 0.075), (-0.075, 0.075), (0.39, 0.54))
ycb_objects_z_buffer = 0.05

ycb_objects = {
    2: "002_master_chef_can",
    3: "003_cracker_box",
    4: "004_sugar_box",
    5: "005_tomato_soup_can",
    6: "006_mustard_bottle"
}

ycb_objects_custom_rotation = {
    6: {"z": -67}
}

ycb_objects_custom_position = {
    2: [-0.07, 0.01, 0.033],
    3: [-0.1, 0.01, 0.022],
    4: [-0.085, 0.02, 0.015],
    5: [-0.05, -0.084, 0.024],
    6: [-0.085, 0, 0]
}


### COMPOSITE SIM CONFIG ###

composite_active_relative_positions = [
    np.asarray([-0.1, 0.1, 0.0]),
    np.asarray([0.1, 0.1, 0.0]),
    np.asarray([-0.1, -0.1, 0.0]),
    np.asarray([0.1, -0.1, 0.0])
]
composite_inactive_base_pose = Pose(np.asarray([0.5, 0.0, 0.0]), np.asarray([0, 0, 0, 0]))
composite_active_base_pose = Pose(np.asarray([0.0, 0.0, 0.36]), np.asarray([0, 0, 0, 0]))

composite_glance_area = GlanceAreaBB((-0.15, 0.15), (-0.15, 0.15), (0.38, 0.425))

composite_objects = [
    [2, 2, 3, 0],
    [1, 2, 3, 5],
    [1, 2, 4, 4],
    [1, 2, 4, 5],
    [2, 2, 4, 4],
    [2, 2, 3, 5],
    [2, 3, 4, 1],
    [2, 3, 4, 4],
    [3, 3, 3, 4],
    [3, 2, 3, 2],
    [2, 3, 3, 5],
    [2, 2, 4, 5],
    [3, 3, 3, 2],
]
