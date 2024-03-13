from haptic_exploration.ml_util import ModelType
from haptic_exploration.config import ObjectSet


def get_pretrained_cls_weights(object_set, model_type):
    spec = {
        (ObjectSet.Basic, ModelType.Transformer):       "basic/cls/weights_2024_01_24_23-23-44_HapticTransformer_cls_pretrained_random_0_4_HD32",
        (ObjectSet.Composite, ModelType.Transformer):   "composite/cls/weights_2024_01_29_21-12-04_HapticTransformer_cls_pretrained_random_0_8",
        (ObjectSet.Composite, ModelType.LSTM):          "composite/cls/weights_2024_01_29_21-32-03_HapticLSTM_cls_pretrained_random_0_8",
    }
    return spec[(object_set, model_type)]


def get_trained_ac(object_set, model_type, action_type="hybrid", shared_architecture=True, init_pretrained=True, freeze_core=False, n_glances=-1):
    if action_type != "glance":
        n_glances = -1
    spec = {
        # GLANCE (FIXED N)
        (ObjectSet.Composite, ModelType.Transformer, "glance", True, True, False, 1): [
            "composite/rl/2024_01_31_04-28-32_HapticTransformer_rl_glance_1_shared_pretrained.pkl"
        ],
        (ObjectSet.Composite, ModelType.Transformer, "glance", True, True, False, 2): [
            "composite/rl/2024_01_31_05-14-32_HapticTransformer_rl_glance_2_shared_pretrained.pkl"
        ],
        (ObjectSet.Composite, ModelType.Transformer, "glance", True, True, False, 3): [
            "composite/rl/2024_01_31_06-14-25_HapticTransformer_rl_glance_3_shared_pretrained.pkl"
        ],
        (ObjectSet.Composite, ModelType.Transformer, "glance", True, True, False, 4): [
            "composite/rl/2024_01_31_07-30-52_HapticTransformer_rl_glance_4_shared_pretrained.pkl"
        ],
        (ObjectSet.Composite, ModelType.Transformer, "glance", True, True, False, 5): [
            "composite/rl/2024_01_31_08-57-43_HapticTransformer_rl_glance_5_shared_pretrained.pkl"
        ],
        (ObjectSet.Composite, ModelType.LSTM, "glance", True, True, False, 1): [
            "composite/rl/2024_01_31_09-14-17_HapticLSTM_rl_glance_1_shared_pretrained.pkl"
        ],
        (ObjectSet.Composite, ModelType.LSTM, "glance", True, True, False, 2): [
            "composite/rl/2024_01_31_09-38-26_HapticLSTM_rl_glance_2_shared_pretrained.pkl"
        ],
        (ObjectSet.Composite, ModelType.LSTM, "glance", True, True, False, 3): [
            "composite/rl/2024_01_31_10-09-54_HapticLSTM_rl_glance_3_shared_pretrained.pkl"
        ],
        (ObjectSet.Composite, ModelType.LSTM, "glance", True, True, False, 4): [
            "composite/rl/2024_01_31_10-50-24_HapticLSTM_rl_glance_4_shared_pretrained.pkl"
        ],
        (ObjectSet.Composite, ModelType.LSTM, "glance", True, True, False, 5): [
            #"composite/rl/2024_01_31_08-57-43_HapticTransformer_rl_glance_5_shared_pretrained.pkl"
        ],
        # HYBRID
        (ObjectSet.Composite, ModelType.Transformer, "hybrid", True, True, False, -1): [
            "composite/rl/2024_01_31_02-31-28_HapticTransformer_rl_hybrid_shared_pretrained.pkl",
            "composite/rl/2024_01_31_03-24-24_HapticTransformer_rl_hybrid_shared_pretrained.pkl",
            "composite/rl/2024_01_31_05-15-30_HapticTransformer_rl_hybrid_shared_pretrained.pkl",
            "composite/rl/2024_01_31_06-08-08_HapticTransformer_rl_hybrid_shared_pretrained.pkl",
        ],
        (ObjectSet.Composite, ModelType.Transformer, "parameterized", True, True, False, -1): [
            "composite/rl/2024_01_31_07-04-48_HapticTransformer_rl_parameterized_shared_pretrained.pkl",
            "composite/rl/2024_01_31_07-56-01_HapticTransformer_rl_parameterized_shared_pretrained.pkl",
            "composite/rl/2024_01_31_08-54-07_HapticTransformer_rl_parameterized_shared_pretrained.pkl",
            "composite/rl/2024_01_31_09-41-20_HapticTransformer_rl_parameterized_shared_pretrained.pkl",
            "composite/rl/2024_01_31_10-30-03_HapticTransformer_rl_parameterized_shared_pretrained.pkl",
        ],
        (ObjectSet.Composite, ModelType.LSTM, "hybrid", True, True, False, -1): [
            "composite/rl/2024_01_31_11-17-45_HapticLSTM_rl_hybrid_shared_pretrained.pkl",
            "composite/rl/2024_01_31_11-54-19_HapticLSTM_rl_hybrid_shared_pretrained.pkl",
            "composite/rl/2024_01_31_12-26-59_HapticLSTM_rl_hybrid_shared_pretrained.pkl",
        ],
        (ObjectSet.Composite, ModelType.LSTM, "parameterized", True, True, False, -1): [
            "composite/rl/2024_01_31_13-00-34_HapticLSTM_rl_parameterized_shared_pretrained.pkl",
            "composite/rl/2024_01_31_13-36-09_HapticLSTM_rl_parameterized_shared_pretrained.pkl",
            "composite/rl/2024_01_31_14-11-25_HapticLSTM_rl_parameterized_shared_pretrained.pkl",
        ],
    }
    return spec[(object_set, model_type, action_type, shared_architecture, init_pretrained, freeze_core, n_glances)]

