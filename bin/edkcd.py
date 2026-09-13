import os
import sys
import warnings
import numpy as np
import pandas as pd
import torch
from loguru import logger
from edkcd import (parse_args, load_dataset, run_grid_search)

warnings.filterwarnings("ignore")


def main():
    args = parse_args()

    # resolve paths
    script_dir = os.path.dirname(os.path.abspath(__file__))
    base_dir = args.base_dir if args.base_dir else os.path.join(
        os.path.dirname(script_dir), 'datasets')
    os.makedirs(base_dir, exist_ok=True)

    # load datasets
    datasets, structures = load_dataset(args.experiment, args.num_sim, base_dir=base_dir)

    # run grid search
    run_grid_search(datasets, structures, args)


if __name__ == "__main__":
    main()