"""
Author: Riccardo Cappuzzo

Main EmbDI script.

Invoke by passing either a single config file, or by passing a directory that 
contains a batch of config files to run.
```
python main.py -f path/to/config/file
python main.py -d path/to/config/directory
```


"""
import argparse
import datetime
import os
import warnings

import pandas as pd

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    import EmbDI.utils as eutils
    from EmbDI.embeddings import learn_embeddings
    from EmbDI.graph import graph_generation
    from EmbDI.logging import log_params, metrics, params
    from EmbDI.sentence_generation_strategies import random_walks_generation
    from EmbDI.testing_functions import match_driver, test_driver
    from EmbDI.utils import (
        OUTPUT_FORMAT,
        TIME_FORMAT,
        check_config_validity,
        dict_compression_edgelist,
        read_edgelist,
    )


def parse_args():
    """Simple argument parser invoked on startup.

    Returns:
        Namespace: Argument parser
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--unblocking", action="store_true", default=False)
    group = parser.add_mutually_exclusive_group()
    group.add_argument("-f", "--config_file", action="store", default=None)
    group.add_argument("-d", "--config_dir", action="store", default=None)
    parser.add_argument("--no_info", action="store_true", default=False)
    built_args = parser.parse_args()
    return built_args


def embeddings_generation(walks, configuration, dictionary):
    """
    Take the generated walks and train embeddings using the walks as training corpus.
    :param walks:
    :param configuration:
    :param dictionary:
    :return:
    """
    start_time = datetime.datetime.now()
    output_file = configuration["run-tag"]

    print(OUTPUT_FORMAT.format("Training embeddings", start_time.strftime(TIME_FORMAT)))
    output_file_name = "pipeline/embeddings/" + output_file + ".emb"
    print("# Writing embeddings in file: {}".format(output_file_name))
    learn_embeddings(
        output_file_name,
        walks,
        write_walks=configuration["write_walks"],
        dimensions=int(configuration["n_dimensions"]),
        window_size=int(configuration["window_size"]),
        training_algorithm=configuration["training_algorithm"],
        learning_method=configuration["learning_method"],
        sampling_factor=configuration["sampling_factor"],
    )
    if configuration["compression"]:
        newf = eutils.clean_embeddings_file(output_file_name, dictionary)
    else:
        newf = output_file_name
    end_time = datetime.datetime.now()
    duration = end_time - start_time
    str_ttime = end_time.strftime(TIME_FORMAT)
    print(OUTPUT_FORMAT.format("Embeddings generation complete", str_ttime))

    configuration["embeddings_file"] = newf

    metrics.time_embeddings = duration.total_seconds()
    return configuration


def training_driver(configuration):
    """This function trains local embeddings according to the parameters
    specified in the configuration.
    The input dataset is transformed into a graph, then random walks are
    generated and the result is passed to the embeddings training algorithm.

    """
    edgelist_df = pd.read_csv(configuration["input_file"], dtype=str, index_col=False)
    edgelist_df = edgelist_df[edgelist_df.columns[:2]]
    edgelist_df.dropna(inplace=True)

    run_tag = configuration["output_file"]
    configuration["run-tag"] = run_tag
    # If task requires training, execute all the steps needed to generate the embeddings.
    if configuration["task"] in ["train", "train-test", "train-match"]:
        # Check if walks have been provided. If not, graph and walks will be generated.
        if configuration["walks_file"] is None:
            prefixes, edgelist = read_edgelist(configuration["input_file"])

            if configuration["compression"]:
                # Execute compression if required.
                edgelist_df, dictionary = dict_compression_edgelist(
                    edgelist_df, prefixes=prefixes
                )
                final_edgelist = edgelist_df.values.tolist()
            else:
                dictionary = None
                final_edgelist = edgelist
            # dictionary=None

            graph = graph_generation(
                configuration, final_edgelist, prefixes, dictionary
            )
            if configuration["n_sentences"] == "default":
                #  Compute the number of sentences according to the rule of thumb.
                configuration["n_sentences"] = graph.compute_n_sentences(
                    int(configuration["sentence_length"])
                )
            walks = random_walks_generation(configuration, graph)
            del graph  # Graph is not needed anymore, so it is deleted to reduce memory cost
        else:
            if configuration["compression"]:  # Execute compression if required.
                prefixes, edgelist = read_edgelist(configuration["input_file"])
                edgelist_df, dictionary = dict_compression_edgelist(
                    edgelist_df, prefixes=prefixes
                )
            else:
                dictionary = None
            configuration["write_walks"] = True
            walks = configuration["walks_file"]
        # return configuration
        configuration = embeddings_generation(walks, configuration, dictionary)
    return configuration


def testing_driver(configuration):
    """Simple caller function for the testing functions."""
    embeddings_file = configuration["embeddings_file"]
    # df = pd.read_csv(configuration['input_file'])
    test_driver(embeddings_file, configuration)


def matching_driver(configuration):
    """Function for generating and printing out the matches.

    Args:
        configuration (dict): Configuration of the run

    Returns:
        str: Path to the row match file.
    """
    embeddings_file = configuration["embeddings_file"]
    input_dataframe = pd.read_csv(configuration["input_file"])

    matches_tuples, matches_columns = match_driver(
        embeddings_file, input_dataframe, configuration
    )

    root_matches = "pipeline/generated-matches/"
    if "run-tag" in configuration:
        matches_file = root_matches + configuration["run-tag"]
    else:
        matches_file = root_matches + configuration["output_file"]
    file_col = matches_file + "_col" + ".matches"
    file_row = matches_file + "_tup" + ".matches"

    with open(file_col, "w") as fp_matches_columns:
        for match in matches_columns:
            output_str = "{} {}\n".format(*match)
            fp_matches_columns.write(output_str)

    with open(file_row, "w") as fp_matches_rows:
        for match in matches_tuples:
            output_str = "{} {}\n".format(*match)
            fp_matches_rows.write(output_str)

    return file_row


def read_configuration(config_file):
    """Read the configuration file and save it in the config dictionary.

    Args:
        config_file (str): Path to the configuration file.

    Returns:
        dict: Dictionary that contains the configuration for this run.
    """
    config = {}

    with open(config_file, "r") as fp_config_file:
        for idx, line in enumerate(fp_config_file):
            line = line.strip()
            if len(line) == 0 or line[0] == "#":
                continue
            split_line = line.split(":")
            if len(split_line) < 2:
                continue
            key, value = split_line
            value = value.strip()
            config[key] = value
    return config


def full_run(config_dir, config_file):
    # Parsing the configuration file.
    configuration = read_configuration(config_dir + "/" + config_file)
    # Checking the correctness of the configuration, setting default values for missing values.
    configuration = check_config_validity(configuration)

    # Running the task specified in the configuration file.
    params.par_dict = configuration

    if configuration["task"] == "train":
        configuration = training_driver(configuration)
    elif configuration["task"] == "test":
        testing_driver(configuration)
        log_params()
    elif configuration["task"] == "match":
        matching_driver(configuration)
    elif configuration["task"] == "train-test":
        configuration = training_driver(configuration)
        testing_driver(configuration)
        log_params()
    elif configuration["task"] == "train-match":
        configuration = training_driver(configuration)
        matching_driver(configuration)


def main(file_path=None, dir_path=None, args=None):
    # Building dir tree required to run the code.
    os.makedirs("pipeline/dump", exist_ok=True)
    os.makedirs("pipeline/walks", exist_ok=True)
    os.makedirs("pipeline/embeddings", exist_ok=True)
    os.makedirs("pipeline/generated-matches", exist_ok=True)
    os.makedirs("pipeline/logging", exist_ok=True)

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
            except RuntimeError as e:
                print(f"Run {file} has failed.")
                print(e)
            except ValueError as e:
                print(f"Run {file} has failed.")
                print(e)
            finally:
                print(f"Run {file} is over.")

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
