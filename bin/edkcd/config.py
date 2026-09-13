import os
import argparse
from pathlib import Path
import yaml


def parse_args():
    parser = argparse.ArgumentParser(description='EDKCD: Edge-Decoupled Koopman Causal Discovery')
    parser.add_argument('--experiment', type=str, default="lorenz96_2")
    parser.add_argument('--num-hidden-layers', type=int, default=6)
    parser.add_argument('--hidden-layer-size', type=int, default=10)
    parser.add_argument('--num-epochs-1', type=int, default=1000)
    parser.add_argument('--num-epochs-2', type=int, default=2000)
    parser.add_argument('--initial-lr', type=float, default=0.01)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--num-sim', type=int, default=5)
    parser.add_argument('--use-cuda', type=bool, default=True)
    parser.add_argument('--cuda-i', type=int, default=0)
    parser.add_argument('--lag', type=int, default=1)
    parser.add_argument('--pred-weight', type=float, default=1.0)
    parser.add_argument('--recon-weight', type=float, default=1.0)
    parser.add_argument('--l1-weight', type=float, default=0.5)
    parser.add_argument('--lin-weight', type=float, default=0.1)
    parser.add_argument('--encoder-type', type=str, default='mlp',
                        choices=['kan', 'mlp'])
    parser.add_argument('--kan-embed-dim', type=int, default=10)
    parser.add_argument('--kan-hidden', type=int, nargs='+', default=[64, 64])
    parser.add_argument('--channel-independent', action='store_true', default=False)
    parser.add_argument('--base-dir', type=str, default=None)
    parser.add_argument('--fast-mode', action='store_true', default=False)
    parser.add_argument('--output-root', type=str, default='logs')
    parser.add_argument('--config', type=str, default=None)
    args = parser.parse_args()

    if args.config is not None:
        _load_yaml_config(args)

    args.base_dir = args.base_dir if args.base_dir else os.path.join(
        os.path.dirname(os.path.abspath(__file__)), '..', '..', 'datasets')
    return args


def _load_yaml_config(args):
    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = Path(__file__).parent.resolve() / args.config
    if not config_path.exists():
        config_path = Path(__file__).parent.parent.parent.resolve() / "config" / args.config
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {args.config}")

    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)

    arg_map = {
        'experiment': 'experiment', 'num_hidden_layers': 'num_hidden_layers',
        'hidden_layer_size': 'hidden_layer_size', 'num_epochs_1': 'num_epochs_1',
        'num_epochs_2': 'num_epochs_2', 'initial_lr': 'initial_lr', 'seed': 'seed',
        'num_sim': 'num_sim', 'use_cuda': 'use_cuda', 'cuda_i': 'cuda_i', 'lag': 'lag',
        'pred_weight': 'pred_weight', 'recon_weight': 'recon_weight',
        'l1_weight': 'l1_weight', 'lin_weight': 'lin_weight',
        'encoder_type': 'encoder_type', 'kan_embed_dim': 'kan_embed_dim',
        'kan_hidden': 'kan_hidden', 'channel_independent': 'channel_independent',
        'base_dir': 'base_dir', 'fast_mode': 'fast_mode',
    }
    for yaml_key, arg_key in arg_map.items():
        if yaml_key in config:
            setattr(args, arg_key, config[yaml_key])
    args._config_loaded = str(config_path)