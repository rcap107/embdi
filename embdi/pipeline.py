"""
This script contains the higher level functions that are required to run the different
steps of the training-testing pipeline.

Author: Riccardo Cappuzzo
"""

import datetime
import tomllib
from pathlib import Path

import pandas as pd

from embdi.embeddings import learn_embeddings
from embdi.graph import graph_generation

# TODO: implement proper logging
from embdi.logging import *
from embdi.sentence_generation_strategies import random_walks_generation
from embdi.testing_functions import match_driver, test_driver
from embdi.utils import (
    OUTPUT_FORMAT,
    TIME_FORMAT,
    check_config_validity,
    clean_embeddings_file,
    dict_compression_edgelist,
    read_edgelist,
)


def embeddings_generation(walks, configuration, dictionary):
    """
    Take the generated walks and train embeddings using the walks as training corpus.
    :param walks:
    :param configuration:
    :param dictionary:
    :return:
    """
    t1 = datetime.datetime.now()
    output_file = configuration["run-tag"]

    print(OUTPUT_FORMAT.format("Training embeddings", t1.strftime(TIME_FORMAT)))
    t = "pipeline/embeddings/" + output_file + ".emb"
    print("# Writing embeddings in file: {}".format(t))
    learn_embeddings(
        t,
        walks,
        write_walks=configuration["write_walks"],
        dimensions=int(configuration["n_dimensions"]),
        window_size=int(configuration["window_size"]),
        training_algorithm=configuration["training_algorithm"],
        learning_method=configuration["learning_method"],
        sampling_factor=configuration["sampling_factor"],
    )
    if configuration["compression"]:
        newf = clean_embeddings_file(t, dictionary)
    else:
        newf = t
    t2 = datetime.datetime.now()
    dt = t2 - t1
    str_ttime = t2.strftime(TIME_FORMAT)
    print(OUTPUT_FORMAT.format("Embeddings generation complete", str_ttime))

    configuration["embeddings_file"] = newf

    metrics.time_embeddings = dt.total_seconds()
    return configuration


def training_driver(configuration):
    """This function trains local embeddings according to the parameters 
    specified in the configuration. The input dataset is transformed into a graph,
    then random walks are generated and the result is passed to the embeddings 
    training algorithm.

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
                el = edgelist_df.values.tolist()
            else:
                dictionary = None
                el = edgelist
            # dictionary=None

            graph = graph_generation(configuration, el, prefixes, dictionary)
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
    embeddings_file = configuration["embeddings_file"]
    df = pd.read_csv(configuration["input_file"])

    matches_tuples, matches_columns = match_driver(embeddings_file, df, configuration)

    root_matches = "pipeline/generated-matches/"
    if "run-tag" in configuration:
        matches_file = root_matches + configuration["run-tag"]
    else:
        matches_file = root_matches + configuration["output_file"]
    file_col = matches_file + "_col" + ".matches"
    file_row = matches_file + "_tup" + ".matches"

    with open(file_col, "w") as fp:
        for m in matches_columns:
            s = "{} {}\n".format(*m)
            fp.write(s)

    with open(file_row, "w") as fp:
        for m in matches_tuples:
            s = "{} {}\n".format(*m)
            fp.write(s)

    return file_row


def read_configuration(config_file):
    config = tomllib.load(open(config_file, "rb"))
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
        results = testing_driver(configuration)
        log_params()
    elif configuration["task"] == "match":
        matching_driver(configuration)
    elif configuration["task"] == "train-test":
        configuration = training_driver(configuration)
        results = testing_driver(configuration)
        log_params()
    elif configuration["task"] == "train-match":
        configuration = training_driver(configuration)
        matching_driver(configuration)
