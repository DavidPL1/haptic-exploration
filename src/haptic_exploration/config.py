from enum import Enum
from pathlib import Path
from rospkg import RosPack

from haptic_exploration import mujoco_config


# GENERAL

WAIT_PLOTS = False
TRANSLATION_STD_M = 0.002
ROTATION_STD_DEG = 1.5


# OBJECT SETS

class ObjectSet(Enum):
    Basic = "basic"
    Composite = "composite"
    YCB = "ycb"
    YCB_rot = "ycb_rot"


try:
    BASE_DIR = Path(RosPack().get_path("haptic_exploration"))
except:
    BASE_DIR = Path("..")

MODEL_SAVE_PATH = BASE_DIR / "saved_models"
EXPERIMENTS_DIR = BASE_DIR / "experiments"

OBJECT_DIR = BASE_DIR / "datasets"
OBJECT_PATHS = {
    ObjectSet.Basic: OBJECT_DIR / "basic_61x61",
    ObjectSet.Composite: OBJECT_DIR / "composite_61x61",
    ObjectSet.YCB: OBJECT_DIR / "ycb_21x16x41",
    ObjectSet.YCB_rot: OBJECT_DIR / "ycb_41_31_rot"
}


GLANCE_AREA = {
    ObjectSet.Basic: mujoco_config.basic_objects_glance_area,
    ObjectSet.Composite: mujoco_config.composite_glance_area,
    ObjectSet.YCB: mujoco_config.ycb_glance_area,
    ObjectSet.YCB_rot: mujoco_config.ycb_glance_area
}
