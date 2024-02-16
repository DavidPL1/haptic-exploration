from enum import Enum


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


OBJECT_PATHS = {
    ObjectSet.Basic: "../datasets/basic_61x61",
    ObjectSet.Composite: "../datasets/composite_61x61"
}
