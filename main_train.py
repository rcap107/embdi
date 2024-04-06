"""
This script is used to train the embeddings starting from the base table, avoiding
additional data integration operation, or tests.

Author: Riccardo Cappuzzo
"""

import argparse
import tomllib
from pathlib import Path

from embdi.pipeline import training_driver
from embdi.utils import check_config_validity


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "config_file",
        action="store",
        type=argparse.FileType("rb"),
        help="Path to the configuration file to use.",
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    config = tomllib.load(args.config_file)

    configuration = check_config_validity(config)
    training_driver(configuration)
