import argparse
import datetime
from pathlib import Path

import warnings

with warnings.catch_warnings():
    warnings.simplefilter('ignore')
    from embdi.embeddings import learn_embeddings
    from embdi.sentence_generation_strategies import random_walks_generation
    from embdi.utils import *

    from embdi.testing_functions import test_driver, match_driver
    from embdi.graph import graph_generation
    # TODO: implement proper logging
    from embdi.logging import *


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--unblocking", action="store_true", default=False)
    group = parser.add_mutually_exclusive_group()
    group.add_argument("-f", "--config_file", action="store", default=None)
    group.add_argument("-d", "--config_dir", action="store", default=None)
    parser.add_argument("--no_info", action="store_true", default=False)
    return parser.parse_args()


def main(file_path=None, dir_path=None, args=None):

    # Building dir tree required to run the code.
    os.makedirs("data/dump", exist_ok=True)
    os.makedirs("data/walks", exist_ok=True)
    os.makedirs("data/embeddings", exist_ok=True)
    os.makedirs("data/generated-matches", exist_ok=True)
    os.makedirs("data/logging", exist_ok=True)

    # Finding the configuration file paths.
    if args:
        if args.config_dir:
            config_dir = args.config_dir
            config_file = None
        else:
            config_dir = None
            config_file = args.config_file
        unblocking = args.unblocking
    else:
        config_dir = dir_path
        config_file = file_path
        unblocking = False

    # Extracting valid files
    if config_dir:
        # TODO: clean this up, use Path
        valid_files = [
            _
            for _ in os.listdir(config_dir)
            if not _.startswith("default") and not os.path.isdir(config_dir + "/" + _)
        ]
        n_files = len(valid_files)
        print("Found {} files".format(n_files))
    elif config_file:
        if args:
            valid_files = [os.path.basename(args.config_file)]
            config_dir = os.path.dirname(args.config_file)
        else:
            valid_files = [os.path.basename(config_file)]
            config_dir = os.path.dirname(config_file)

    else:
        raise ValueError("Missing file_path or config_path.")

    if unblocking:
        print("######## IGNORING EXCEPTIONS ########")
        for idx, file in enumerate(sorted(valid_files)):
            try:
                print("#" * 80)
                print("# File {} out of {}".format(idx + 1, len(valid_files)))
                print("# Configuration file: {}".format(file))
                t_start = datetime.datetime.now()
                print(
                    OUTPUT_FORMAT.format("Starting run.", t_start.strftime(TIME_FORMAT))
                )
                print()

                full_run(config_dir, file)

                t_end = datetime.datetime.now()
                print(OUTPUT_FORMAT.format("Ending run.", t_end.strftime(TIME_FORMAT)))
                dt = t_end - t_start
                print("# Time required: {:.2} s".format(dt.total_seconds()))
            except Exception as e:
                print(f"Run {file} has failed. ")
                print(e)
    else:
        for idx, file in enumerate(sorted(valid_files)):
            print("#" * 80)
            print("# File {} out of {}".format(idx + 1, len(valid_files)))
            print("# Configuration file: {}".format(file))
            t_start = datetime.datetime.now()
            print(OUTPUT_FORMAT.format("Starting run.", t_start.strftime(TIME_FORMAT)))
            print()

            full_run(config_dir, file)

            t_end = datetime.datetime.now()
            print(OUTPUT_FORMAT.format("Ending run.", t_end.strftime(TIME_FORMAT)))
            dt = t_end - t_start
            print("# Time required: {:.2f} s".format(dt.total_seconds()))

    # clean_dump()


if __name__ == "__main__":
    args = parse_args()
    main(args=args)
