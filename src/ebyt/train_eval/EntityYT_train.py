import sys
import time
import argparse
import hyperstate
import json

from pathlib import Path
from typing import Type
from enn_trainer import State, TrainConfig, init_train_state, train
from entity_gym.env import Environment
from yawning_titan.game_modes.game_mode import GameMode
from yawning_titan.networks.network import Network

from ebyt.envs.entity_cyber.entity_yawning_titan_generic import EntityYT
script_dir = Path(__file__).resolve().parent
parent_dir = script_dir.parent

def network_env_cls(network: Network, game_mode: GameMode, random : bool, random_seed: int, n_nodes: int, edge_prob: float
    ) -> Type[Environment]:
    class env_cls(EntityYT):
        def __init__(self):
            super().__init__(network=network, game_mode=game_mode, random=random, random_seed=random_seed, n_nodes=n_nodes, edge_prob=edge_prob)

    return env_cls


def main(args):
    with open(f"{parent_dir}/configs/yawning_titan/game_modes/{args.game_mode}") as game_mode_config:
        game_mode = GameMode.create(dict=json.load(game_mode_config))
    
    game_mode.red.agent_attack.skill.value = args.red_skill
    game_mode.game_rules.max_steps = args.episode_length

    #clear cl arguments for hyperstate - to avoid conflicts with hyperstate argparsing
    sys.argv = sys.argv[:1]
    sys.argv.extend([f"eval.steps={args.episode_length}"])
    sys.argv.extend([f"eval.capture_videos={args.render}"])
    sys.argv.extend([f"eval.interval={args.eval_freq}"])
    sys.argv.extend([f"total_timesteps={args.total_timesteps}"])  
    sys.argv.extend([f"name=EntityYT_{args.n_nodes}_nodes_fixed_length_{args.episode_length}_skill_{args.red_skill}_{'random' if args.random else 'static'}"])
    sys.argv.extend([f"env.id={'EntityYT_random' if args.random else 'EntityYT_static'}"])
    sys.argv.extend(["--config", f"{parent_dir}/configs/yawning_titan/train_config/{args.config}"])
    sys.argv.extend(["--checkpoint-dir", args.checkpoint_dir])
    
    random_seed = args.seed
    network = EntityYT.generate_random_network(n=args.n_nodes, p=args.edge_prob)
    env_cls = network_env_cls(network, game_mode, args.random, random_seed, args.n_nodes, args.edge_prob)
    
    
    @hyperstate.stateful_command(TrainConfig, State, init_train_state)
    def start_training(state_manager: hyperstate.StateManager) -> None:
        train(state_manager=state_manager, env=env_cls)


    start_training()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-nodes", type=int, default=10)
    parser.add_argument("--edge-prob", type=float, default=0.1)
    parser.add_argument("--episode-length", type=int, default=100)
    parser.add_argument("--red-skill", type=float, default=0.7)
    parser.add_argument("--random", action="store_true")
    parser.add_argument("--render", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--total-timesteps", type=int, default=1000000)
    parser.add_argument("--eval-freq", type=int, default=10000)
    parser.add_argument("--game-mode", type=str, default="fixed_episode_base.json")
    parser.add_argument(
        "--checkpoint-dir",
        type=str,
        help="Directory to save checkpoints"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="EntityYT_base.ron",
        help="Path to the training configuration file"
    )
    args = parser.parse_known_args()[0]

    if not args.checkpoint_dir:
        if args.random:
            episode_type = "random"
        else:
            episode_type = "static"
        
        args.checkpoint_dir = f"checkpoints/EntityYT_{args.n_nodes}_nodes_fixed_length_{args.episode_length}_skill_{args.red_skill}_{episode_type}_{time.time()}"

    main(args)
