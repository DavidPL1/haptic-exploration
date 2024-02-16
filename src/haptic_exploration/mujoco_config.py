import numpy as np
from enum import Enum

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

ycb_glance_area = GlanceAreaBB((-0.18, 0.18), (-0.14, 0.14), (0.39, 0.51))
ycb_z_buffer = 0.05

ycb_names = [
    #"001_chips_can",
    "002_master_chef_can",
    "003_cracker_box",
    "004_sugar_box",
    "005_tomato_soup_can",
    "006_mustard_bottle",
    "007_tuna_fish_can",
    "008_pudding_box",
    "009_gelatin_box",
    "010_potted_meat_can",
    "011_banana",
    "012_strawberry",
    "013_apple",
    "014_lemon",
    "015_peach",
    "016_pear",
    "017_orange",
    "018_plum",
    #"019_pitcher_base",
    #"021_bleach_cleanser",
    #"022_windex_bottle",
    #"023_wine_glass",
    #"024_bowl",
    #"025_mug",
    #"026_sponge",
    #"027-skillet",
    #"028_skillet_lid",
    #"029_plate",
    #"030_fork",
    #"031_spoon",
    #"032_knife",
    #"033_spatula",
    #"035_power_drill",
    #"036_wood_block",
    #"037_scissors",
    #"038_padlock",
    #"039_key",
    #"040_large_marker",
    #"041_small_marker",
    #"042_adjustable_wrench",
    #"043_phillips_screwdriver",
    #"044_flat_screwdriver",
    #"046_plastic_bolt",
    #"047_plastic_nut",
    #"048_hammer",
    #"049_small_clamp",
    #"050_medium_clamp",
    #"051_large_clamp",
    #"052_extra_large_clamp",
    #"053_mini_soccer_ball",
    #"054_softball",
    #"055_baseball",
    #"056_tennis_ball",
    #"057_racquetball",
    #"058_golf_ball",
    #"059_chain",
    #"061_foam_brick",
    #"062_dice",
]

ycb_objects = {int(name.split("_")[0]): name for name in ycb_names}


ycb_objects_custom_rotation = {
    2: {"y": 90},
    3: {"y": 90},
    4: {"y": 90},
    5: {"y": 90},
    6: {"z": -67},
    8: {"z": -28},
    9: {"z": 77},
    10: {"x": 90, "y": 90, "z": -3},
    11: {"z": -70},
    12: {"z": -45},
    14: {"z": -42},
    15: {"y": -43, "z": 32},
    16: {"z": -86},
    17: {"y": 180},
    18: {"z": 60},
}

ycb_objects_custom_position = {
    2: [-0.07, 0.01, 0.033],
    3: [-0.1, 0.01, 0.022],
    4: [-0.085, 0.02, 0.015],
    5: [-0.05, -0.084, 0.024],
    6: [-0.085, 0, 0],
    7: [0.026, 0.02, 0],
    8: [-0.008, -0.018, 0],
    9: [0, 0.023, 0],
    10: [-0.04, 0.033, 0.051],
    11: [0, -0.01, 0],
    12: [0, -0.01, 0],
    13: [0, 0.005, 0],
    14: [-0.005, -0.0205, 0],
    15: [0.03, 0.005, 0.02],
    16: [-0.014, -0.031, 0],
    17: [-0.008, 0.02, 0.0705],
    18: [0.02, -0.002, 0]
}


class YCBMesh(Enum):
    TSDF = "tsdf"
    Poisson = "poisson"
    Google16k = "google_16k"
    Google64k = "google_64k"
    Google512k = "google_64k"


ycb_objects_default_mesh = YCBMesh.Google16k
ycb_objects_custom_meshes = {
    1: YCBMesh.TSDF
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
