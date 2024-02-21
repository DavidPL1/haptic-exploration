from haptic_exploration.ml_util import ModelType
from haptic_exploration.config import ObjectSet


def get_pretrained_cls_weights(object_set, model_type):
    spec = {
        (ObjectSet.YCB, ModelType.Transformer):       "ycb/cls/weights_2024_02_21_17-13-34_HapticTransformer_cls_pretrained_random_0_8",
    }
    return spec[(object_set, model_type)]


def get_trained_ac(object_set, model_type, action_type="hybrid", shared_architecture=True, init_pretrained=True, freeze_core=False, n_glances=-1):
    if action_type != "glance":
        n_glances = -1
    spec = {
        (ObjectSet.YCB, ModelType.Transformer, "hybrid", True, True, False, -1): [
            "ycb/rl/2024_02_19_13-06-32_HapticTransformer_rl_hybrid_shared_pretrained.pkl"
        ],
    }
    return spec[(object_set, model_type, action_type, shared_architecture, init_pretrained, freeze_core, n_glances)]

