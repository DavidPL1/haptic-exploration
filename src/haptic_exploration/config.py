from enum import Enum
from pathlib import Path
from rospkg import RosPack

from haptic_exploration import mujoco_config


# GENERAL

MODEL_SAVE_PATH = "../saved_models/"
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
    base_dir = Path(RosPack().get_path("haptic_exploration"))
except:
    base_dir = Path("..")

object_dir = base_dir / "datasets"
OBJECT_PATHS = {
    ObjectSet.Basic: object_dir / "basic_61x61",
    ObjectSet.Composite: object_dir / "composite_61x61",
    ObjectSet.YCB: object_dir / "ycb_21x16x41",
    ObjectSet.YCB_rot: object_dir / "ycb_41_31_rot"
}


GLANCE_AREA = {
    ObjectSet.Basic: mujoco_config.basic_objects_glance_area,
    ObjectSet.Composite: mujoco_config.composite_glance_area,
    ObjectSet.YCB: mujoco_config.ycb_glance_area,
    ObjectSet.YCB_rot: mujoco_config.ycb_glance_area
}
