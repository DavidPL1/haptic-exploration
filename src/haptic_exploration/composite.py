from collections import defaultdict
from copy import deepcopy
from operator import itemgetter
from dataclasses import dataclass
from typing import Dict, Tuple, List, Optional

import haptic_exploration.mujoco_config as mujoco_config


@dataclass
class Knowledge:
    hint: Tuple[Optional[int], ...]                             # (1, None, 2, None)
    remaining_objects: List[int]                                # [0, 3, 5, 6]
    next_dict: Dict[int, Dict[int, Tuple[float, 'Knowledge']]]  # { 1 -> {2 -> (1.0, K1)}, 3 -> {1 -> (0.5, K2), 2 -> (0.5, K3)}}
    next_expected_lengths: Dict[int, float]                     # { 1 -> 2.0, 3 -> 1.5}
    best_expected_length: float                                 # 2.5
    best_min_length: float
    best_max_length: float

    def __str__(self):
        return f"(Hint={self.hint}, N={len(self.remaining_objects)}, E={self.best_expected_length:.2f}, MIN={self.best_min_length:.2f}, MAX={self.best_max_length:.2f})"

    def print_probs(self):
        for position, feature_dict in self.next_dict.items():
            position_best_min_length = min(knowledge.best_min_length for prob, knowledge in feature_dict.values())
            position_best_max_length = max(knowledge.best_max_length for prob, knowledge in feature_dict.values())
            print(f"POS={position}: E={self.next_expected_lengths[position]:.2f} (MIN={position_best_min_length:.2f}, MAX={position_best_max_length:.2f})")
            for feature, (prob, knowledge) in feature_dict.items():
                print(" " * 3, f"F={feature}: P={prob:.2f} N={len(knowledge.remaining_objects)} E={knowledge.best_expected_length:.2f} (MIN={knowledge.best_min_length:.2f}, MAX={knowledge.best_max_length:.2f})")


def match_hint(hint, composite_object) -> bool:
    return all(known_feature == object_feature for known_feature, object_feature in zip(hint, composite_object) if known_feature is not None)


def possible_object_ids(hint, composite_objects):
    return [object_id for object_id, object_features in enumerate(composite_objects) if all(feature_hint == object_feature for feature_hint, object_feature in zip(hint, object_features) if feature_hint is not None)]


def construct_knowledge_table(composite_objects):

    assert len(composite_objects) > 0, "there must be composite objects"
    assert len(composite_objects) == len({tuple(obj) for obj in composite_objects}), "duplicate objects!"

    n_positions = len(composite_objects[0])

    # construct masks for hint
    masks = [[()]]
    for n_unknown_positions in range(1, n_positions + 1):
        next_masks = []
        for mask in masks[-1]:
            highest_idx = -1 if len(mask) == 0 else mask[-1]
            for next_mask_idx in range(highest_idx + 1, n_positions):
                next_masks.append(mask + (next_mask_idx,))
        masks.append(next_masks)

    # construct all possible hints
    all_hints = []
    for n_unknown_positions in range(0, n_positions + 1):
        hints = []
        for composite_object in composite_objects:
            for mask in masks[n_unknown_positions]:
                hints.append(tuple(None if idx in mask else feature for idx, feature in enumerate(composite_object)))
        all_hints.append(set(hints))

    # iteratively construct table
    knowledge_table: Dict[Tuple[Optional[int], ...], Knowledge] = dict()
    for hints in all_hints:
        for hint in hints:
            remaining_object_ids = possible_object_ids(hint, composite_objects)

            if None in hint:
                next_dict = dict()
                for position_idx, feature_hint in enumerate(hint):
                    if feature_hint is None:
                        possible_features = set(composite_objects[object_id][position_idx] for object_id in remaining_object_ids)
                        next_knowledges = []
                        for feature in sorted(possible_features):
                            next_hint = list(hint)
                            next_hint[position_idx] = feature
                            next_hint = tuple(next_hint)
                            next_knowledges.append(knowledge_table[next_hint])
                        next_dict[position_idx] = {knowledge.hint[position_idx]: (len(knowledge.remaining_objects)/len(remaining_object_ids), knowledge) for knowledge in next_knowledges}
                next_expected_lengths = {position_idx: sum(prob * knowledge.best_expected_length for prob, knowledge in next_knowledges.values()) for position_idx, next_knowledges in next_dict.items()}
                if len(remaining_object_ids) == 1:
                    best_expected_length = 0
                    best_max_length = 0
                    best_min_length = 0
                else:
                    best_expected_length = 1 + min(next_expected_lengths.values())
                    best_expected_positions = [position for position, expected_length in next_expected_lengths.items() if abs(expected_length - (best_expected_length-1)) < 1e-3]
                    best_min_length = 1 + min(min([next_knowledge.best_min_length for prob, next_knowledge in next_dict[position].values()]) for position in best_expected_positions)
                    best_max_length = 1 + max(max([next_knowledge.best_max_length for prob, next_knowledge in next_dict[position].values()]) for position in best_expected_positions)
                knowledge = Knowledge(hint, remaining_object_ids, next_dict, next_expected_lengths, best_expected_length, best_min_length, best_max_length)
            else:
                knowledge = Knowledge(hint, remaining_object_ids, dict(), dict(), 0, 0, 0)
            knowledge_table[hint] = knowledge

    return knowledge_table


def get_feature_position_param(position_idx, zero_centered=False):
    x_offset, y_offset = mujoco_config.composite_active_relative_positions[position_idx][:2]
    _, max_x = mujoco_config.composite_glance_area.x_limits
    x, y = x_offset/max_x, y_offset/max_x
    if zero_centered:
        return x, y
    else:
        return (1 + x)/2, (1 + y)/2


def calculate_object_n():
    knowledge_table = construct_knowledge_table(mujoco_config.composite_objects)

    object_n_dict = dict()
    feature_dict = {f: 0 for f in range(6)}
    for i, obj in enumerate(mujoco_config.composite_objects):
        knowledge = knowledge_table[(None, None, None, None)]
        N = 0
        while True:
            if knowledge.best_min_length == 0:
                break
            next_position = sorted(knowledge.next_expected_lengths.items(), key=lambda v: v[1])[0][0]
            knowledge = knowledge.next_dict[next_position][obj[next_position]][1]
            N += 1
            feature_dict[obj[next_position]] += 1
        object_n_dict[i] = N

    knowledge = knowledge_table[(None, None, None, None)]
    print(knowledge)
    knowledge.print_probs()

    print("Optimal Object N Hist", object_n_dict)
    print("Optimal Feature Hist", feature_dict)
    print("Optimal Expected N", sum(object_n_dict.values()) / len(object_n_dict))

    return object_n_dict, feature_dict


def main():
    calculate_object_n()


if __name__ == "__main__":
    main()
