import os
import wandb
import glob
import json
import torch
import argparse

from pathlib import Path
from enn_trainer import load_checkpoint
from enn_trainer.config import EnvConfig
from enn_trainer.train import load_rogue_net_opponent
from enn_trainer.eval import run_eval
from entity_gym.env import Environment, ParallelEnvList, ValidatingEnv, EnvList
from entity_gym.env.vec_env import VecEnv
from entity_gym.simple_trace import Tracer
from dataclasses import asdict
from typing import Any, Callable, Dict, Type
from torch.utils.tensorboard import SummaryWriter
from re import split
from yawning_titan.game_modes.game_mode import GameMode
from yawning_titan.networks.network import Network

from ebyt.envs.entity_cyber.entity_yawning_titan_generic import EntityYT

script_dir = Path(__file__).resolve().parent
parent_dir = script_dir.parent

def _env_factory(
    env_cls: Type[Environment],
) -> Callable[[EnvConfig, int, int, int], VecEnv]:
    def _create_env(
        cfg: EnvConfig, num_envs: int, processes: int, first_env_index: int
    ) -> VecEnv:
        if cfg.validate:
            create_env = lambda: ValidatingEnv(env_cls())
        else:
            create_env = lambda: env_cls()  # type: ignore
        if processes > 1:
            return ParallelEnvList(create_env, num_envs, processes)
        else:
            return EnvList(create_env, num_envs)

    return _create_env


def flatten(config: Dict[str, Any], prefix: str = '') -> Dict[str, Any]:
    """
    Flattens a nested dictionary.

    Args:
        config (Dict[str, Any]): The dictionary to flatten.
        prefix (str): The prefix to add to the keys.

    Returns:
        Dict[str, Any]: The flattened dictionary.
    """
    flattened = {}
    for k, v in config.items():
        if isinstance(v, dict):
            flattened.update(flatten(v, k if prefix == '' else f'{prefix}.{k}'))
        else:
            flattened[prefix + k] = v
    return flattened

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
    
    if not args.checkpoint_dir:
        p = Path(f'{parent_dir}/train_eval/checkpoints')
        subdirectories = [split(r'_(random|static)_',x.name) for x in p.iterdir() if x.is_dir() and x.name.startswith('EntityYT')]
        subdirectories = [x for x in subdirectories if len(x) == 3]
        recent = sorted(subdirectories, key=lambda x: float(x[2]), reverse=True)[0]
        print(recent)
        checkpoint_dir = '_'.join(recent)
    else:
        checkpoint_dir = args.checkpoint_dir

    network = EntityYT.generate_random_network(n=args.n_nodes, p=args.edge_prob)
    eval_env_cls = network_env_cls(network, game_mode, args.random, args.seed, args.n_nodes, args.edge_prob)
    
    latest_checkpoint = glob.glob(f"{parent_dir}/train_eval/checkpoints/{checkpoint_dir}/latest*")
    print('LATEST CHECKPOINT GET', latest_checkpoint)
    checkpoint = load_checkpoint(latest_checkpoint[0])
    # redefine EnvConfig
    env_config = EnvConfig(
        kwargs='\{\}',
        id=f"EntityYT_{'random' if args.random else 'static'}",
        validate=True,
    )
    cfg = checkpoint.config

    rollout_config = checkpoint.config.rollout
    eval_config = checkpoint.config.eval
    create_env = _env_factory(eval_env_cls)
    data_path = Path(cfg.data_dir).absolute()
    data_path.mkdir(parents=True, exist_ok=True)
    data_dir = str(data_path)
    cuda = torch.cuda.is_available()
    device = torch.device('cuda' if cuda else 'cpu')
    agent = checkpoint.state.agent.to(device)

    eval_config.capture_videos = args.render
    wandb.init(
        project='',
        entity='',
        sync_tensorboard=True,
        config=asdict(eval_config),
        name=f"eval_{checkpoint_dir}_on_{'random' if args.random else 'static'}_{args.n_nodes}_nodes_length_{args.episode_length}_skill_{args.red_skill}",
        save_code=True,
        dir=data_dir,
        monitor_gym=True,
    )
    wandb.watch(agent)

    writer = SummaryWriter(os.path.join(data_dir, 'runs/eval_test/'))

    writer.add_text(
        'hyperparameters',
        '|param|value|\n|-|-|\n%s'
        % (
            '\n'.join(
                [
                    f'|{key}|{value}|'
                    for key, value in flatten(asdict(eval_config)).items()
                ]
            )
        ),
    )

    tracer = Tracer(cuda=cuda)

    global_step = 0
    with tracer.span('eval'):
        for _ in range(args.num_evals):
            run_eval(
                cfg=eval_config,
                env_cfg=env_config,
                rollout=rollout_config,
                create_env=create_env,
                create_opponent=load_rogue_net_opponent,
                agent=agent,
                device=device,
                tracer=tracer,
                writer=writer,
                global_step=global_step,
                rank=0,
                parallelism=1,
            )
            global_step += 1


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-nodes", type=int, default=10)
    parser.add_argument("--edge-prob", type=float, default=0.1)
    parser.add_argument("--episode-length", type=int, default=100)
    parser.add_argument("--red-skill", type=float, default=0.7)
    parser.add_argument("--random", action="store_true")
    parser.add_argument("--render", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    
    parser.add_argument("--game-mode", type=str, default="fixed_episode_base.json")
    parser.add_argument(
        "--checkpoint-dir",
        type=str,
        help="Directory from which to load checkpoint for evaluation"
    )
    parser.add_argument(
        '--num-evals', type=int, default=30, help='Number of evaluation episodes'
    )

    args = parser.parse_known_args()[0]

  
    
    return args


if __name__ == '__main__':
    args = parse_args()
    main(args)
