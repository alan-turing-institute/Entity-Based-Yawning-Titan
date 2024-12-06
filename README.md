
<img src="media/ebyt.png" alt="Entity-Based Yawning Titan" width="150">

# Entity-Based Yawning Titan
This repository contains an adaptation of the <a href="https://github.com/dstl/YAWNING-TITAN">Yawning Titan</a> cyber simulator, as released by DSTL, to the <a href="https://github.com/entity-neural-network/entity-gym">Entity Gym</a> interface. This adaptation enables <a href="https://clemenswinter.com/2023/04/14/entity-based-reinforcement-learning/">entity-based reinforcement learning</a>, which is designed to improve the generalisability of reinforcement learning agents across varying network topologies.

  The repository provides training and evaluation scripts that can reproduce the experiments described in the workshop paper <em><a href="https://arxiv.org/abs/2410.17647">Entity-based Reinforcement Learning for Autonomous Cyber Defence</a></em>. This paper introduces and motivates the use of entity-based reinforcement learning in the context of autonomous cyber defence.

  The training process leverages the <a href="https://github.com/entity-neural-network/enn-trainer">Entity-Neural-Net Trainer</a> package and its default Proximal Policy Optimization (PPO) implementation. It also uses the <a href="https://github.com/entity-neural-network/rogue-net">RogueNet</a> Transformer policy, specifically designed for Entity Gym environments, applied to a modified entity-based Yawning Titan environment.

  For comparison, baseline scripts utilize the <a href="https://github.com/DLR-RM/stable-baselines3">Stable Baselines 3</a> PPO trainer with Multilayer Perceptron policy parameterisation. These baselines are trained on an equivalent Yawning Titan environment that retains the <a href="https://github.com/openai/gym">OpenAI Gym</a> interface.

  If you use this repository in your research, please consider citing our <a href="https://arxiv.org/abs/2410.17647">companion paper</a>:
</p>

```bibtex
@article{SymesThompson2024EntityACD,
  title={Entity-based Reinforcement Learning for Autonomous Cyber Defence},
  author={Symes Thompson, Isaac and Caron, Alberto and Hicks, Chris and Mavroudis, Vasilios},
  journal={arXiv preprint arXiv:2410.17647v2},
  year={2024}
}
```

Please also cite the original Yawning Titan work, with citation specified in the repository [here](https://github.com/dstl/YAWNING-TITAN). If using the Entity Gym package, consider citing their [repository](https://github.com/entity-neural-network/entity-gym) or the original [blog post](https://clemenswinter.com/2023/04/14/entity-based-reinforcement-learning/) by the authors, as we do in the paper.
## Setup
### Docker
This repository contains a Dockerfile and docker-compose.yaml file for use with Docker compose. This might be the simplest way to get the scripts working.

1. First, clone the repository, then navigate to the root directory.
2. Make sure Docker is running, and then depending on your Docker/ Docker Compose version. Run:
```bash
docker-compose run --rm entity-yt
```
Or:
```bash
docker compose run --rm -it entity-yt
```

This should build and open terminal access to a docker container, with a volume mounted to the project directory, with all relevant packages installed.

NB: A dependency of the Entity-Neural-Net Trainer is the *[pyron](https://pypi.org/project/python-ron/)* library, for which wheels are provided only for x86 architectures. Therefore it is necessary to use emulation if running on non-x86 (e.g Arm Apple Silicon). This is specified on the docker-compose file.

### Manual Installation

This is tested with python 3.9 on Ubuntu 22.04, CPU only. With models of default sizes, initial experiments with GPU acceleration were found to hamper performance. 

#### Step 1: Install Gym 0.21.0
Installing Gym 0.21.0 using pip requires specific versions of setuptools and wheel, otherwise installation is likely to fail.
```bash
pip install setuptools==66 wheel==0.38.4
```
```bash
pip install gym==0.21.0
```

#### Step 2: Install Entity Neural Network dependencies

Install `typing_extensions`, `hyperstate`, and `enn_trainer` with the following command:

```bash
pip install typing_extensions hyperstate enn_trainer
```

#### Step 3: Install PyTorch and Torch scatter

For a CPU-only version of PyTorch, run:

```bash
pip install torch==1.12.0+cpu --extra-index-url https://download.pytorch.org/whl/cpu
```
```bash
pip install --no-cache-dir --no-index torch-scatter -f https://data.pyg.org/whl/torch-1.12.0+cpu.html
```

#### Step 4: Install Yawning Titan and dependencies
```bash
pip install stable_baselines3==1.6.2 wandb platformdirs networkx tinydb tabulate
```
Make sure you are in the parent directory, and installing the Yawning Titan version present in this repository.
```bash
pip install -e yawning_titan
```
#### Step 5: Install ebyt as a package
```bash
pip install -e .
```
## Weights & Biases setup
Logging with Weights & Biases is supported. When training an entity-based agent, this involves modifying the training config file (in `src/ebg/configs/yawning_titan/train_config/EntitYT ... .ron`) to add a wandb user and project, and enabling tracking. For example:
```rust
TrainConfig(
    ...
    track: true,
    wandb_project_name: "EntityYT"
    wandb_entity: "John_Smith"
    ...
)
```
For logging the training of the Stable Baselines 3 agents, you must modify the arguments of the `wandb.init` inside the `Sb3_PPO_YT_train.py` script:
```python
wandb.init(
    ...
    project='EntityYT',
    entity='John_Smith',
    sync_tensorboard=True,
)
```

For logging the evaluation of either kind of agent, you must modify the `wandb.init` call in the relevant evaluation script.
```python
eval_config.capture_videos = False
wandb.init(
    project='EntityYT',
    entity='John_Smith',
    ...
)
```
## Overview of Entity-based Yawning Titan environment
The `EntityYT` environment class inside `src/ebg/envs/entity_cyber/entity_yawning_titan_generic.py` contains the Entity-based version of the Yawning Titan environment. It is derived from the `GenericNetworkEnv` environment, which can be found in `yawning_titan/src/yawning_titan/envs/generic/generic_env.py` for reference.

Broadly speaking, the main difference between an 'entity-based' approach and the approach used in the Yawning Titan environment is that the observation and action spaces are treated as collections of distinct objects or entities (nodes), whereas in the default Gym environment all features are concatenated into a unified observation vector.

For example, the observation space is defined as:
```python 
def obs_space(self) -> ObsSpace:
    return ObsSpace(
        entities={
            'Generic_Node': Entity(features=self.feature_list),
            'Defender': Entity(features=[]),
        }
    )
```
This specifies the different possible 'entity types' in the environment, as well as a list of named features that entities of that type might have. In our case, we have a single generic entity type for all the nodes in the environment, with the feature list built from all those that are enabled in the blue agent observation space in the game mode config file. We also have a dummy 'Defender' entity type with no specified features - this is to allow for a composite action space where the blue agent first chooses an action type, and then a node to act on. As shown:
```python
def action_space(self) -> Dict[ActionName, ActionSpace]:
    return {
        'High_Level': GlobalCategoricalActionSpace(self.action_list),
        'Target': SelectEntityActionSpace(),
    }
```
Here a 'GlobalCategoricalActionSpace' is defined, containing all of the types of actions the blue agent may perform. This list is built based on the game mode configuration file. This is similar to the standard discrete action space found in Gym. A 'SelectEntityActionSpace' is also specified, which is used for selecting a particular node to execute the action that has been selected by the 'Global' action space.

Note that both the observation and action space provide only the types of possible observations and actions, with the specific compositions potentially varying with each timestep or between episodes. This allows for the initialisation of a policy network with the environment whilst still being flexible to environment variation.

The actual observations and actions are made available to an agent in an `Observation` object at a particular timestep, upon calling the `observe` function.
```python
def observe(self, done=False, reward=0) -> Observation:
    return Observation(
        entities={
            'Generic_Node': (
                self.get_node_features(),
                [('Generic_Node', i) for i in range(len(self.nodes))],
            ),
            'Defender': (np.zeros([1, 0], dtype=np.float32), [('Defender', 0)]),
        },
        actions={
            'High_Level': GlobalCategoricalActionMask(),
            'Target': SelectEntityActionMask(
                actor_types=['Defender'], actee_types=['Generic_Node']
            ),
        },
        done=done,
        reward=reward,
    )
```
Here, a potentially variable length list of node features is provided to the agent, returned by the `get_node_features` function. The length of this list depends on the number of nodes in the environment. The defender entity is given a dummy feature vector.

The actions available to the agent are defined by action masks over the previously specified action spaces. There is no action mask on the high level action type space. The action mask on the SelectEntityActionSpace determines that the 'Defender' entity is to act on the 'Generic_Node' entities. This prompts the policy network to use information in the embedding of the Defender entity to decide which node to execute the defensive action on. If there were multiple Defender entities then an action would be chosen for each Defender, since the mask is based on entity types rather than specific entities.
## Overview of Scripts

All scripts are present in the `src/ebg/train_eval/` directory.
### 1. `EntityYT_train.py`
This script facilitates training using the `EntityYT` entity-based yawning titan environment. It trains a version of PPO 

#### Arguments:
 - "*--n-nodes*" Specifies the number of nodes to be used in the Yawning Titan network environment. Default 10
 - "*--edge-prob*" Specifies the edge probability parameter when generating the network (using Erdős-Renyi model). Default 0.1
 - "*--episode-length*" Specifies the length of each episode. Default 100
 - "*--red-skill*" Specifies the 'skill' of the red agent in Yawning Titan. Between 0.0 and 1.0. Default 0.7
 - "*--random*" Specifies whether the network environment is re-generated between episode resets. If absent then the same network is used throughout the training run.
 - "*--render*" Specifies whether the environment is rendered on logging and evaluation.
 - "*--seed*" Random seed, default 0
 - "*--total-timesteps*" Total number of environment timesteps to run the training for.
 - "*--eval-freq*" Interval (in environment timesteps) between evaluations and logging of the policy.
 - "*--game-mode*" Specifies which Yawning Titan gamemode config file to use in the game_modes config directory. Default "fixed_episode_base.json"
 - "*--config*" Specifies the training configuration file, in the train_config directory. "EntityYT_base.ron" by default.
 - "*--checkpoint-dir*" Specifies a directory to save checkpoints. By default this is generated automatically from other arguments and placed in a "checkpoints" parent directory.

#### Example Usage:
```bash
python EntityYT_train.py --n-nodes 10 --edge-prob 0.1 --episode-length 100 \
    --total-timesteps 1000000 --eval-freq 10000 --random --config EntityYT_base.ron
```

### 2. `EntityYT_evaluate.py`
Evaluates a trained agent in the `EntityYT` environment, given a checkpoint.

#### Arguments
- "*--n-nodes*" Specifies the number of nodes to be used in the evaluation environment. Default 10
 - "*--edge-prob*" Specifies the edge probability parameter when generating networks (using Erdős-Renyi model). Default 0.1
 - "*--episode-length*" Specifies the length of each episode. Default 100
 - "*--red-skill*" Specifies the 'skill' of the red agent in Yawning Titan. Between 0.0 and 1.0. Default 0.7
 - "*--random*" Specifies whether the network environment is re-generated between episode resets. If absent then the same network is used throughout evaluation.
 - "*--render*" Specifies whether the environment is rendered during evaluation.
 - "*--seed*" Randoom seed, default 0
 - "*--game-mode*" Specifies which Yawning Titan gamemode config file to use in the game_modes config directory. Default "fixed_episode_base.json"
 - "*--num-evals*" Specifies the number of episodes to evaluate the agent over, default 30. In the [paper](https://arxiv.org/abs/2410.17647), evaluations over 1000 episodes are used.
 - "*--checkpoint-dir*" Specifies a directory to load checkpoints from. By default, the most recent checkpoint beginning with "EntityYT" in the checkpoints directory will be loaded.
#### Example Usage:
```bash
python EntityYT_evaluate.py --n-nodes 10 --edge-prob 0.1 --episode-length 100 --num-evals 1000
```

### 3. `Sb3_PPO_YT_train.py`
This script trains an agent using `Stable-Baselines3` PPO in an OpenAI Gym version of the Yawning Titan environment. This version uses fully 'concatenated' observation and action spaces, and the policy is parameterised using a multilayer perceptron.
This script has the same arguments as the Entity-based Yawning Titan training script, apart from the rendering and logging function, which proved harder to implement correctly.
#### Arguments:
 - "*--n-nodes*" Specifies the number of nodes to be used in the Yawning Titan network environment. Default 10
 - "*--edge-prob*" Specifies the edge probability parameter when generating the network (using Erdős-Renyi model). Default 0.1
 - "*--episode-length*" Specifies the length of each episode. Default 100
 - "*--red-skill*" Specifies the 'skill' of the red agent in Yawning Titan. Between 0.0 and 1.0. Default 0.7
 - "*--random*" Specifies whether the network environment is re-generated between episode resets. If absent then the same network is used throughout the training run.
 - "*--seed*" Random seed, default 0
 - "*--total-timesteps*" Total number of environment timesteps to run the training for.
 - "*--eval-freq*" Interval (in environment timesteps) between evaluations and logging of the policy.
 - "*--game-mode*" Specifies which Yawning Titan gamemode config file to use in the game_modes config directory. Default "fixed_episode_base.json"
 - "*--config*" Specifies the training configuration file, in the train_config directory. "Sb3_PPO_base.ron" by default.
 - "*--checkpoint-dir*" Specifies a directory to save checkpoints. By default this is generated automatically from other arguments and placed in a "checkpoints" parent directory.
#### Example Usage:
```bash
python Sb3_PPO_YT_train.py --n-nodes 10 --edge-prob 0.1 --episode-length 100 \
    --total-timesteps 1000000 --eval-freq 10000 --random --config Sb3_PPO_base.ron
```

### 4. `Sb3_PPO_YT_evaluate.py`
Evaluates a checkpointed Stable Baselines 3 PPO agent on a Yawning Titan environment. Uses the same argumnts as the Entity-based Yawning Titan evaluation script.

#### Arguments
- "*--n-nodes*" Specifies the number of nodes to be used in the evaluation environment. Default 10
 - "*--edge-prob*" Specifies the edge probability parameter when generating networks (using Erdős-Renyi model). Default 0.1
 - "*--episode-length*" Specifies the length of each episode. Default 100
 - "*--red-skill*" Specifies the 'skill' of the red agent in Yawning Titan. Between 0.0 and 1.0. Default 0.7
 - "*--random*" Specifies whether the network environment is re-generated between episode resets. If absent then the same network is used throughout evaluation.
 - "*--render*" Specifies whether the environment is rendered during evaluation.
 - "*--seed*" Randoom seed, default 0
 - "*--game-mode*" Specifies which Yawning Titan gamemode config file to use in the game_modes config directory. Default "fixed_episode_base.json"
 - "*--num-evals*" Specifies the number of episodes to evaluate the agent over, default 30. In the [paper](https://arxiv.org/abs/2410.17647), evaluations over 1000 episodes are used.
 - "*--checkpoint-dir*" Specifies a directory to load checkpoints from. By default, the most recent checkpoint beginning with "EntityYT" in the checkpoints directory will be loaded.
#### Example Usage:
```bash
python Sb3_PPO_YT_evaluate.py --n-nodes 10 --edge-prob 0.1 --episode-length 100 \
    --checkpoint-dir checkpoints/Sb3_PPO_YT_checkpoint_dir --num-evals 30
```

## Overview of Configuration files
### Game Mode Configuration
Game modes are defined in JSON files located in `configs/yawning_titan/game_modes/`. Modify these files to adjust environment parameters, blue and red capabilities, and the reward function used by the environment.

:rotating_light: ! Important: Currently the Entity-based Yawning Titan environment does not support all of the Game mode options provided in Yawning Titan. So experimentation outside of the provided configuration may be unreliable. In particular, episodes with early termination conditions, and Deceptive node functionality are not supported.

The arguments to the training scripts '*episode-length*', '*red-skill*' override the values in the game_mode config, to allow for quicker experimentation

### Training Configuration
Training settings are specified in `.ron` files within `configs/yawning_titan/train_config/`.
- The EntityYT ... .ron configs follow the structure provided in the Entity Neural Network project. They specify the configuration of the *Roguenet* Transformer policy, the optimiser and PPO hyperparameters, the evaluation configuration as well as higher level options such as wandb settings and overall training length. `EntityYT_base.ron` contains the parameters used for the experiments in the paper, which are largely the defaults for the Entity-Neural-Network Trainer package. `EntityYT_alt.ron` contains PPO and optimiser parameters closer to the Stable Baselines 3 defaults. The training script arguments *total-timesteps* and *eval-freq* override the 'total_timesteps' and 'eval.interval' values present in these files.
- The Sb3_PPO ... .ron configs specify the hyperparameters for Stable Baselines 3 PPO. `Sb3_PPO_default.ron` contains the default hyperparameters as specified in the Stable Baselines 3 documentation, and as used for the experiments in the paper. `Sb3_PPO_alt.ron` contains hyperparameters closer to the defaults used for the PPO optimiser in the Entity-Neural-Network Trainer.

---


## ***Supported by the National Cyber Security Centre (NCSC)***
![](https://github.com/alan-turing-institute/Entity-Based-Generalisation/blob/main/media/example_network.gif)
