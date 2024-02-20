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


#object_dir = Path(RosPack().get_path("haptic_exploration"), "datasets")
object_dir = Path("../datasets/")
OBJECT_PATHS = {
    ObjectSet.Basic: object_dir / "basic_61x61",
    ObjectSet.Composite: object_dir / "composite_61x61",
    ObjectSet.YCB: object_dir / "ycb_21x16x41"
}


GLANCE_AREA = {
    ObjectSet.Basic: mujoco_config.basic_objects_glance_area,
    ObjectSet.Composite: mujoco_config.composite_glance_area,
    ObjectSet.YCB: mujoco_config.ycb_glance_area
}
