from pathlib import Path
from rospkg import RosPack

from haptic_exploration.object_sets import ObjectSet
from haptic_exploration import mujoco_config
from haptic_exploration.object_controller import SimpleObjectController, YCBObjectController, CompositeObjectController

# GENERAL

WAIT_PLOTS = False
TRANSLATION_STD_M = 0.002
ROTATION_STD_DEG = 1.5


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

SAMPLING_SPEC = {
    ObjectSet.Basic: (
        SimpleObjectController,
        mujoco_config.basic_objects,
        90,
        mujoco_config.basic_objects_z_buffer,
        [("x", 61), ("y_angle", 61)],
        {}
    ),
    ObjectSet.Composite: (
        CompositeObjectController,
        mujoco_config.composite_objects_dict,
        0, # max angle is not used for composite objects
        mujoco_config.composite_objects_z_buffer,
        [("x", 61), ("y", 61)],
        dict(composite_objects=mujoco_config.composite_objects)
    ),
    ObjectSet.YCB: (
        YCBObjectController,
        mujoco_config.ycb_objects,
        45,
        mujoco_config.ycb_z_buffer,
        [("x", 3), ("x_angle", 3)],
        {}
        # [("x", 3), ("y", 3), ("x_angle", 3), ("y_angle", 3)]
    )
}