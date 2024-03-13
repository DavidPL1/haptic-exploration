import sys

from haptic_exploration.data import GlanceTable
from haptic_exploration.environment import HapticExplorationTableEnv, HapticExplorationSimEnv
from haptic_exploration.generation import generate_dataset
from haptic_exploration.visualization import summarize_training
from haptic_exploration.config import OBJECT_PATHS, ObjectSet
from haptic_exploration.train_cls import train_cls_random
from haptic_exploration.ml_util import set_seeds, print_summary, get_model_params_dataset, get_model_params_env, save_best_model_weights, \
    load_model_weights, get_device, save_rl, load_rl, ModelType, get_best_checkpoints
from haptic_exploration.model import TransformerParameters, build_model_cls, LSTMParameters, build_model_cls, \
    build_model_value, HapticTransformer, HapticLSTM, HapticTransformer, \
    build_model_action, build_model_shared
from haptic_exploration.actor_critic import ActorCritic, ActorCriticHyperparameters
from haptic_exploration.actions import HybridActionSpace, DiscreteActionSpace, ContinuousActionSpace, ActionSpaceConfig, \
    ParameterizedActionSpace
from haptic_exploration.glance_controller import MocapGlanceController
from haptic_exploration.object_controller import SimpleObjectController, CompositeObjectController, YCBObjectController
import haptic_exploration.mujoco_config as mujoco_config

from spec import get_pretrained_cls_weights, get_trained_ac


### MODEL CONFIG ###
POSITION_EMBEDDED_DIM = 32
PRESSURE_EMBEDDED_DIM = 32
BASE_OUTPUT_DIM = 64
N_TRANSFORMER_HEADS = 2
N_TRANSFORMER_ENCODER_LAYERS = 2
N_LSTM_STACKED_LAYERS = 1
NUM_GLANCES = [1]

### CLS CONFIG ###
NUM_CLS_SAMPLES_TRAIN = int(1e4)
NUM_CLS_SAMPLES_TEST = NUM_CLS_SAMPLES_TRAIN//10
NUM_CLS_EPOCHS = 5
CLS_LR = 0.0003
CLS_BATCH_SIZE = 64

### RL CONFIG ###
RL_BATCH_SIZE = 1
NUM_RL_EPOCHS = {
    ObjectSet.Basic: 25,
    ObjectSet.Composite: 200,
    ObjectSet.YCB: 500,
    ObjectSet.YCB_rot: 200
}
NUM_RL_EPISODES = 1000
RL_STORE_WEIGHT_INTERVAL = 20


def get_model(model_type):
    if model_type == ModelType.Transformer:
        core_class = HapticTransformer
        core_params = TransformerParameters(N_TRANSFORMER_ENCODER_LAYERS, N_TRANSFORMER_HEADS, True, False)
    elif model_type == ModelType.LSTM:
        core_class = HapticLSTM
        core_params = LSTMParameters(N_LSTM_STACKED_LAYERS)
    else:
        raise Exception("unsupported model type")
    return core_class, core_params


def get_sim_env(object_set):
    if object_set == ObjectSet.Basic:
        glc = MocapGlanceController(SimpleObjectController(), mujoco_config.basic_objects_glance_area)
    elif object_set == ObjectSet.Composite:
        glc = MocapGlanceController(CompositeObjectController(mujoco_config.composite_objects), mujoco_config.composite_glance_area)
    elif object_set in (ObjectSet.YCB, ObjectSet.YCB_rot):
        def id_mapping(object_id):
            name = mujoco_config.ycb_names[object_id]
            return {v: k for k, v in mujoco_config.ycb_objects.items()}[name]
        glc = MocapGlanceController(YCBObjectController(id_mapping=id_mapping), mujoco_config.ycb_glance_area, 0.349066, mujoco_config.ycb_z_buffer)
    else:
        raise Exception("unsupported object set")
    return HapticExplorationSimEnv(glc)


def train_cls(fixed_glances=False, save=True):

    glance_table = GlanceTable(OBJECT_SET)

    train_sets = []
    validation_sets = []
    dataset_names = []
    dataset_properties = []

    if fixed_glances:
        dataset_train, dataset_test, data_properties = generate_dataset("random", glance_table, NUM_GLANCES[0], NUM_CLS_SAMPLES_TRAIN, NUM_CLS_SAMPLES_TEST, add_noise=ADD_NOISE)
        train_sets.append(dataset_train)
        validation_sets.append(dataset_test)
        dataset_names.append(f"r_{NUM_GLANCES[0]}")
        dataset_properties.append(data_properties)
    else:

        # generate dataset for classification
        random_n_glances = list(range(5)) if OBJECT_SET == ObjectSet.Basic else list(range(9))
        for n in random_n_glances:
            dataset_train, dataset_test, data_properties = generate_dataset("random", glance_table, n, NUM_CLS_SAMPLES_TRAIN, NUM_CLS_SAMPLES_TEST)
            train_sets.append(dataset_train)
            validation_sets.append(dataset_test)
            dataset_names.append(f"r_{n}")
            dataset_properties.append(data_properties)
            #analyse_dataset(dataset_test, data_properties)

        # perfect glances just for validation of composite objects
        if OBJECT_SET == ObjectSet.Composite:
            for n in range(1, 5):
                dataset_train, dataset_test, data_properties = generate_dataset("position", glance_table, n, NUM_CLS_SAMPLES_TRAIN, NUM_CLS_SAMPLES_TEST)
                validation_sets.append(dataset_test)
                dataset_names.append(f"p_{n}")
                dataset_properties.append(data_properties)

    # construct actual model
    model_params = get_model_params_dataset(dataset_properties[0], POSITION_EMBEDDED_DIM, PRESSURE_EMBEDDED_DIM, BASE_OUTPUT_DIM)
    core_class, core_params = get_model(MODEL_TYPE)
    model = build_model_cls(core_class, core_params)(model_params)
    print_summary(model, model_params)
    print(model)

    # perform training
    training_monitor = train_cls_random(train_sets, validation_sets, dataset_names, model, NUM_CLS_EPOCHS, batch_size=CLS_BATCH_SIZE, lr=CLS_LR)

    if save:
        if fixed_glances:
            save_suffix = '_'.join(str(n) for n in NUM_GLANCES)
        else:
            save_suffix = f"random_{random_n_glances[0]}_{random_n_glances[-1]}"
        save_best_model_weights(model, training_monitor, OBJECT_SET, f"{core_class.__name__}_cls_pretrained_{save_suffix}")


def run_rl(save=True, init_pretrained=True, freeze_core=False, test=False):

    glance_table = GlanceTable(OBJECT_SET)
    table_env = HapticExplorationTableEnv(glance_table, first_obs="empty", add_noise=ADD_NOISE, add_offset=ADD_OFFSET)

    model_params = get_model_params_env(table_env, POSITION_EMBEDDED_DIM, PRESSURE_EMBEDDED_DIM, BASE_OUTPUT_DIM, NUM_GLANCES)
    core_class, core_params = get_model(MODEL_TYPE)

    space_config = ActionSpaceConfig(log_std_init=-1.3)
    decision_space = DiscreteActionSpace(2, net_dims=(32,))
    glance_space = ContinuousActionSpace(glance_table.n_params, net_dims=(32,), space_config=space_config)
    hybrid_space = HybridActionSpace([decision_space, glance_space], space_config=space_config)
    parameterized_space = ParameterizedActionSpace(decision_space, {0: glance_space}, space_config=space_config)

    # GLANCE Model
    action_type = RL_ACTION_TYPE
    if action_type == "decision":
        action_space = decision_space
        latent_dim = model_params.core_hidden_dim
        action_net, action_params = action_space.build_action_net(latent_dim)
    elif action_type == "glance":
        action_space = glance_space
        latent_dim = model_params.core_hidden_dim
        action_net, action_params = action_space.build_action_net(latent_dim)
        table_env.glance_reward = 0
    elif action_type == "hybrid":
        action_space = hybrid_space
        latent_dim = [model_params.core_hidden_dim, model_params.core_hidden_dim]
        action_net, action_params = action_space.build_action_net(latent_dim)
    elif action_type == "parameterized":
        action_space = parameterized_space
        latent_dim = [model_params.core_hidden_dim, [model_params.core_hidden_dim]]
        action_net, action_params = action_space.build_action_net(latent_dim)
    else:
        raise Exception("invalid action type")

    if SHARED_ARCHITECTURE:
        shared_model = build_model_shared(core_class, core_params, action_type, action_net)(model_params)
    else:
        cls_model = build_model_cls(core_class, core_params)(model_params)
        glance_model = build_model_action(core_class, core_params, action_type, action_net)(model_params)
        value_model = build_model_value(core_class, core_params)(model_params)

    if test: # load and evaluate configuration
        n_glances = NUM_GLANCES[0] if action_type == "glance" else -1
        trained_ac_files = get_trained_ac(OBJECT_SET, MODEL_TYPE, action_type=action_type, n_glances=n_glances)

        if not isinstance(trained_ac_files, list):
            trained_ac_files = [trained_ac_files]

        for trained_ac_file in trained_ac_files:
            space, hp, checkpoints = load_rl(trained_ac_file)
            #visualize_training(checkpoints, optimal_n=2.58)

            best_cps = get_best_checkpoints(checkpoints)
            best_cp = next(cp for cp in reversed(best_cps) if cp.shared_model_weights is not None)
            #best_cp = checkpoints[-1]
            #best_cp = best_cp
            print("INITIALIZE BEST CHECKPOINT:")
            print("Epoch", best_cp.i_epoch)
            print("Accuracy:", best_cp.validation_stats.accuracy)
            print("Reward:", best_cp.validation_stats.avg_reward)
            print("N_glances:", best_cp.validation_stats.avg_n_glances)

            if SHARED_ARCHITECTURE:
                shared_model.load_state_dict(best_cp.shared_model_weights)
                model_spec = shared_model
            else:
                cls_model.load_state_dict(best_cp.cls_model_weights)
                glance_model.load_state_dict(best_cp.action_model_weights)
                value_model.load_state_dict(best_cp.value_model_weights)
                model_spec = cls_model, glance_model, value_model

            action_params = best_cp.action_params
            table_env.verbose = False
            table_ac = ActorCritic(table_env, SHARED_ARCHITECTURE, model_spec, action_params, action_space, hyperparameters=hp)
            stats = table_ac.evaluate(eval_desc=f"TABLE EVAL (epoch {best_cp.i_epoch})", deterministic=True)

            sim_env = get_sim_env(OBJECT_SET)
            sim_env.verbose = True
            sim_ac = ActorCritic(sim_env, SHARED_ARCHITECTURE, model_spec, action_params, action_space, hyperparameters=hp)
            stats = sim_ac.evaluate(eval_desc=f"SIM EVAL (epoch {best_cp.i_epoch})")

    else:
        if init_pretrained:
            # Load model weights
            pretrained_cls_weights = get_pretrained_cls_weights(OBJECT_SET, MODEL_TYPE)
            if SHARED_ARCHITECTURE:
                cls_model_tmp = build_model_cls(core_class, core_params)(model_params)
                load_model_weights(cls_model_tmp, pretrained_cls_weights)
                shared_model.model.model[0].load_state_dict(cls_model_tmp.model.model[0].state_dict())
                shared_model.model.model[1].classification_output.load_state_dict(cls_model_tmp.model.model[1].state_dict())
            else:
                load_model_weights(cls_model, pretrained_cls_weights)
                glance_model.model.model[0].load_state_dict(cls_model.model.model[0].state_dict())
                value_model.model.model[0].load_state_dict(cls_model.model.model[0].state_dict())

        if freeze_core:
            if SHARED_ARCHITECTURE:
                for param in shared_model.model.model[0]:
                    param.requires_grad = False
            else:
                for param in list(cls_model.model.model[0].parameters()) + list(glance_model.model.model[0].parameters()) + list(value_model.model.model[0].parameters()):
                    param.requires_grad = False

        hp = ActorCriticHyperparameters(method="reinforce", n_glances=NUM_GLANCES[0])
        if SHARED_ARCHITECTURE:
            model_spec = shared_model
        else:
            model_spec = cls_model, glance_model, value_model
        ac = ActorCritic(table_env, SHARED_ARCHITECTURE, model_spec, action_params, action_space, hyperparameters=hp, store_weight_interval=RL_STORE_WEIGHT_INTERVAL)
        checkpoints = ac.train(NUM_RL_EPOCHS[OBJECT_SET], NUM_RL_EPISODES, draw_plots=False)

        if save:
            action_type_str = action_type if action_type != "glance" else f"{action_type}_{hp.n_glances}"
            architecture_str = "_shared" if SHARED_ARCHITECTURE else "_split"
            pretrained_suffix = "_pretrained" if init_pretrained else ""
            freeze_core_suffix = "_frozen" if freeze_core else ""
            save_rl((action_space, hp, checkpoints), OBJECT_SET, f"{core_class.__name__}_rl_{action_type_str}" + architecture_str + pretrained_suffix + freeze_core_suffix)

        summarize_training(checkpoints)


def train_rl_all():
    global OBJECT_SET, MODEL_TYPE, NUM_GLANCES, RL_ACTION_TYPE

    object_spec = [
        #ObjectSet.Basic,
        ObjectSet.Composite
    ]
    model_spec = [
        ModelType.Transformer,
        ModelType.LSTM
    ]
    pretrained_freeze_spec = [
        (True, False),
        #(True, True),
    ]
    n_glances_spec = [
        1,
        #2,
        #3,
        #4,
    ]
    action_type_spec = [
        "hybrid",
        "parameterized"
    ]

    repeat = 5

    total_runs = repeat * len(object_spec) * len(model_spec) * len(pretrained_freeze_spec) * len(n_glances_spec) * len(action_type_spec)
    print("TOTAL RUNS:", total_runs)

    for object_set in object_spec:
        for model_type in model_spec:
            for init_pretrained, freeze_core in pretrained_freeze_spec:
                for n_glances in n_glances_spec:
                    for action_type in action_type_spec:
                        for _ in range(repeat):
                            OBJECT_SET = object_set
                            MODEL_TYPE = model_type
                            NUM_GLANCES = [n_glances]
                            RL_ACTION_TYPE = action_type
                            print(f"*** RUNNING: {object_set}, {model_type}, n_glances={n_glances}, action_type={RL_ACTION_TYPE},init_pretrained={init_pretrained}, freeze_core={freeze_core}")
                            run_rl(init_pretrained=init_pretrained, freeze_core=freeze_core)


def main():
    set_seeds(0)

    if RUN_TYPE == "train_cls":
        train_cls()
    elif RUN_TYPE == "train_rl":
        run_rl()
    elif RUN_TYPE == "test_rl":
        run_rl(test=True)
    elif RUN_TYPE == "train_rl_all":
        train_rl_all()
    else:
        raise Exception("invalid run type")


RUN_TYPE = "train_cls"
OBJECT_SET = ObjectSet.YCB_rot
MODEL_TYPE = ModelType.Transformer
RL_ACTION_TYPE = "hybrid"
SHARED_ARCHITECTURE = True
ADD_NOISE = False
ADD_OFFSET = True

if __name__ == "__main__":
    if len(sys.argv) > 1:
        RUN_TYPE = sys.argv[1]
    if len(sys.argv) > 2:
        NUM_GLANCES = [int(sys.argv[2])]

    main()
