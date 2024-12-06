from __future__ import annotations
import json
import wandb
import argparse
import numpy as np
from pathlib import Path
from re import split
from yawning_titan.game_modes.game_mode import GameMode
from yawning_titan.networks.network import Network
from typing import Type
from entity_gym.env import Environment
from ebyt.train_eval.Sb3_PPO_YT_runner import YawningTitanRun
from ebyt.envs.entity_cyber.yawning_titan_generic_baseline_env import YTGenericNetworkEnv
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.monitor import Monitor
from stable_baselines3 import PPO
script_dir = Path(__file__).resolve().parent
parent_dir = script_dir.parent

PPO_TENSORBOARD_LOGS_DIR = f"{parent_dir}/train_eval/tensorboard_logs"

def network_env_cls(network: Network, game_mode: GameMode, random : bool, random_seed: int, n_nodes: int, edge_prob: float
    ) -> Type[Environment]:
    class env_cls(YTGenericNetworkEnv):
        def __init__(self):
            super().__init__(network=network, game_mode=game_mode, random=random, random_seed=random_seed, n_nodes=n_nodes, edge_prob=edge_prob)

    return env_cls


def main(args):
    game_mode_path = parent_dir / f"configs/yawning_titan/game_modes/{args.game_mode}"
    if not game_mode_path.exists():
        raise FileNotFoundError(f"Game mode config file not found: {game_mode_path}")
    with open(game_mode_path) as game_mode_config:
        game_mode = GameMode.create(dict=json.load(game_mode_config))
    
    game_mode.red.agent_attack.skill.value = args.red_skill
    game_mode.game_rules.max_steps = args.episode_length
    if not args.checkpoint_dir:
        p = Path(f'{parent_dir}/train_eval/checkpoints')
        subdirectories = [split(r'_(random|static)_',x.name) for x in p.iterdir() if x.is_dir() and x.name.startswith('Sb3_PPO_YT')]
        subdirectories = [x for x in subdirectories if len(x) == 3]
        recent = sorted(subdirectories, key=lambda x: float(x[2]), reverse=True)[0]
        print(recent)
        checkpoint_dir = '_'.join(recent)
    else:
        checkpoint_dir = args.checkpoint_dir
    
    network = YTGenericNetworkEnv.generate_random_network(n=args.n_nodes, p=args.edge_prob)
    eval_env = YTGenericNetworkEnv(network, game_mode, args.random, args.seed, args.n_nodes, args.edge_prob)
    wandb.init(
        project='',
        entity='',
        name=f"eval_{checkpoint_dir}_on_{'random' if args.random else 'static'}_{args.n_nodes}_nodes_fixed_length_{args.episode_length}_skill_{args.red_skill}",
        sync_tensorboard=True,
    )
    model = PPO.load(
            f"{parent_dir}/train_eval/checkpoints/{checkpoint_dir}/ppo.zip",
            eval_env,
            verbose=1,
            tensorboard_log=str(PPO_TENSORBOARD_LOGS_DIR),
            seed=args.seed,
        )
    eval_env = Monitor(eval_env)

    # Inspired by stable baselines 3 evaluation callback
    wandb.watch(model.policy)
    num_timesteps = 0
    for _ in range(args.num_evals):
        episode_rewards, episode_lengths = evaluate_policy(
                                            model,
                                            eval_env,
                                            n_eval_episodes=1,
                                            render=args.render,
                                            deterministic=True,
                                            return_episode_rewards=True,
                                            warn=True,
                                            callback=None,
                                        )
        num_timesteps += args.episode_length
        assert isinstance(episode_rewards, list)
        assert isinstance(episode_lengths, list)
        mean_reward=np.mean(episode_rewards)
        mean_ep_length= np.mean(episode_lengths)
  
        print(f"Eval num_timesteps={num_timesteps}, " f"episode_reward={mean_reward:.2f}")
        print(f"Episode length: {mean_ep_length:.2f}")
        # Add to current Logger
        wandb.log({"eval/mean_reward": float(mean_reward), "eval/mean_ep_length": mean_ep_length, "time/total_timesteps": num_timesteps})






if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-nodes", type=int, default=10)
    parser.add_argument("--edge-prob", type=float, default=0.1)
    parser.add_argument("--episode-length", type=int, default=100)
    parser.add_argument("--red-skill", type=float, default=0.7)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--random", action="store_true")
    parser.add_argument("--render", action="store_true")

    parser.add_argument("--game-mode", type=str, default="fixed_episode_base.json")
    parser.add_argument(
        "--checkpoint-dir",
        type=str,
        help="Directory from which to load checkpoints"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="Sb3_PPO_base.ron",
        help="Training configuration filename"
    )
    parser.add_argument(
        '--num-evals', type=int, default=30, help='Number of evaluation episodes'
    )
    args = parser.parse_known_args()[0]

    main(args)