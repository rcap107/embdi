import os
import string
import warnings

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

from EmbDI.logging import *

TIME_FORMAT = "%Y-%m-%d %H:%M:%S"
OUTPUT_FORMAT = "# {:.<60} {}"

POSSIBLE_TASKS = ["train", "test", "match", "train-test", "train-match", "debug"]

MLFLOW_NOT_FOUND = False

digs = string.digits + string.ascii_uppercase


def remove_prefixes(edgelist_file, model_file):
    """
    Utility function for preparing a new embeddings file in which all embeddings have their prefixes removed.
    The contents of embeddings file will be copied inside a new file.

    :param edgelist_file: File that contains the prefixes to remove in the header.
    :param model_file: Embeddings file to clean.
    :return: Path to the new, cleaned embeddings file.
    """
    newf, _ = os.path.splitext(model_file)
    newf += "_cleaned.emb"

    with open(edgelist_file) as fp:
        node_types = fp.readline().strip().split(",")
        prefixes = [_.split("__")[1] for _ in node_types]

    with open(model_file, "r") as fin, open(newf, "w") as fo:
        for idx, line in enumerate(fin):
            if idx > 0:
                split = line.split("__", maxsplit=1)
                if len(split) == 2:
                    pre, rest = split
                    if pre in prefixes:
                        fo.write(rest)
                    else:
                        fo.write(line)
                else:
                    fo.write(line)
            else:
                fo.write(line)


    return newf


def apply_PCA(embeddings_file, reduced_file, n_components):
    """
    Utility function for reducing the dimensionality of the embeddings. Given the embeddings file and a path to the
    output file, reduce the size of the input embeddings to n_components dimensions using PCA.

    :param embeddings_file: Path of the input embeddings file.
    :param reduced_file: Path of file  in which to save the reduced embeddings.
    :param n_components: Number of dimensions to reduce the embeddings to.
    :return:
    """
    keys = []

    with open(embeddings_file, "r") as fp:
        lines = fp.readlines()

        sizes = lines[0].split()
        sizes = [int(_) for _ in sizes]

        mat = np.zeros(shape=sizes)
        for n, line in enumerate(lines[1:]):
            ll = line.strip().split()
            mat[n, :] = np.array(ll[1:])
            keys.append(ll[0])

    if sizes[1] < n_components:
        raise ValueError(
            f"The number of input dimensions ({sizes[1]}) is smaller than "
            f"the number of output dimensions ({n_components})."
        )

    pca = PCA(n_components=n_components)

    mat_fit = pca.fit_transform(mat)

    with open(reduced_file, "w") as fp:
        fp.write("{} {}\n".format(*mat_fit.shape))
        for n, key in enumerate(keys):
            fp.write(
                "{} ".format(key) + " ".join([str(_) for _ in mat_fit[n, :]]) + "\n"
            )

    print("Written on file {}.".format(reduced_file))


def clean_dump():
    filelist = [f for f in os.listdir("pipeline/dump")]
    for f in filelist:
        os.remove(os.path.join("pipeline/dump", f))

    filelist = [f for f in os.listdir("pipeline/test")]
    for f in filelist:
        os.remove(os.path.join("pipeline/test", f))


def find_intersection_flatten(df, info_file):
    with open(info_file, "r") as fp:
        line = fp.readline()
        n_items = int(line.split(",")[1])
    df1 = df[:n_items]
    df2 = df[n_items:]
    #     Code to perform word-wise intersection
    # s1 = set([str(_) for word in df1.values.ravel().tolist() for _ in word.split('_')])
    # s2 = set([str(_) for word in df2.values.ravel().tolist() for _ in word.split('_')])
    s1 = set([str(_) for _ in df1.values.ravel().tolist()])
    s2 = set([str(_) for _ in df2.values.ravel().tolist()])

    intersection = s1.intersection(s2)

    return intersection


def compute_n_tokens(df_file):
    df = pd.read_csv(df_file, dtype=str)
    n_rows = len(df)
    #    n_values = len(set(df.values.ravel().tolist()))
    uniques = []
    n_col = len(df.columns)
    for col in df.columns:
        uniques += df[col].unique().tolist()
    n_values = len(set(uniques))
    return (n_rows + n_values + n_col) * 10


def int2base(x, base):
    """
    Convert x in base 10 to x in base 'base'
    :param x:
    :param base:
    :return: base(x)
    """
    if x < 0:
        sign = -1
    elif x == 0:
        return digs[0]
    else:
        sign = 1

    x *= sign
    digits = []

    while x:
        digits.append(digs[int(x % base)])
        x = int(x / base)

    if sign < 0:
        digits.append("-")

    digits.reverse()

    return "".join(digits)


def dict_compression_edgelist(edgelist, prefixes):
    uniques = sorted(list(set(edgelist.values.ravel().tolist())))

    # Expanding strings that contain '_'
    prefixes = [_[4:] for _ in prefixes]
    listed_uniques = {
        _ for l in uniques for idx, _ in enumerate(l.split("_")) if idx > 0 and _ != ""
    }
    uniques = sorted(list(listed_uniques))

    # Removing null values from the compression.
    if "" in uniques:
        uniques.remove("")
    if np.nan in uniques:
        uniques.remove(np.nan)

    # Generating the keys in base 36.
    keys = ["@{}".format(int2base(_, len(digs))) for _ in range(len(uniques))]
    dictionary = dict(zip(uniques, keys))

    # Replacing word by word according to the dictionary.
    def replace(line, dictionary, prefixes):
        s = []
        for idx, val in enumerate(line.split("_")):
            if val in prefixes:
                s.append(val + "_")
            elif val in dictionary:
                s.append(dictionary[val])
        return "_".join(s)

    for col in edgelist.columns:
        edgelist[col] = edgelist[col].apply(
            replace, dictionary=dictionary, prefixes=prefixes
        )
    return edgelist, {v: k for k, v in dictionary.items()}


def dict_decompression_flatten(df, dictionary):
    def replace(line, dictionary):
        s = []
        for val in line.split("_"):
            if val in dictionary:
                s.append(dictionary[val])
        return "_".join(s)

    d = dict(zip(dictionary.values(), dictionary.keys()))
    for col in df.columns:
        df[col] = df[col].apply(replace, dictionary=d)
    return df, d


def clean_embeddings_file(embeddings_file, dictionary):
    emb_path, ext = os.path.splitext(embeddings_file)
    with open(emb_path + ext, "r") as fp:
        newf = emb_path + ".embs"
        with open(newf, "w") as fp2:
            for i, line in enumerate(fp):
                if i > 0:
                    key, vector = line.strip().split(" ", maxsplit=1)
                    prefix, word = key.split("__", maxsplit=1)
                    if word.startswith("@"):
                        if word in dictionary:
                            match = dictionary[word]
                            s = "{} {}".format(prefix + "__" + match, vector) + "\n"
                            fp2.write(s)
                        else:
                            wlist = []
                            for w in word.split("@"):
                                if len(w) > 0:
                                    _t = "@" + w.strip("_")
                                    wlist.append(dictionary[_t])
                            k = prefix + "__" + "_".join(wlist)
                            s = "{} {}".format(k, vector) + "\n"
                            fp2.write(s)
                    else:
                        # fp2.write(line)
                        raise ValueError("{} does not work".format(word))
                else:
                    fp2.write(line)
    # os.remove(embeddings_file)
    return newf


def read_similarities(sim_file):
    sims = pd.read_csv(sim_file)
    if len(sims.columns) == 2:
        sims["distance"] = 1
    return sims.values.tolist()


def return_default_values(config):
    default_values = {
        "ntop": 10,
        "ncand": 1,
        "max_rank": 3,
        "follow_sub": False,
        "smoothing_method": "no",
        "backtrack": True,
        "training_algorithm": "word2vec",
        "write_walks": True,
        "flatten": "all",
        "indexing": "basic",
        "epsilon": 0.1,
        "num_trees": 250,
        "compression": False,
        "n_sentences": "default",
        "walks_strategy": "basic",
        "learning_method": "skipgram",
        "sentence_length": 60,
        "window_size": 5,
        "n_dimensions": 300,
        "numeric": "no",
        "experiment_type": "ER",
        "intersection": False,
        "walks_file": None,
        "mlflow": False,
        "repl_numbers": False,
        "repl_strings": False,
        "sampling_factor": 0.001,
    }

    for k in default_values:
        if k not in config:
            config[k] = default_values[k]
    return config


def _convert_to_bool(config, key):
    if config[key] in [True, False]:
        return config
    if config[key].lower() not in ["true", "false"]:
        raise ValueError(
            "Unknown {key} parameter {value}".format(key=key, value=config[key])
        )
    else:
        if config[key].lower() == "false":
            config[key] = False
        elif config[key].lower() == "true":
            config[key] = True
    return config


def read_edgelist(edgelist_path):
    with open(edgelist_path, "r") as fp:
        edgelist = []
        for idx, line in enumerate(fp):
            if idx == 0:
                node_types = line.strip().split(",")
            else:
                l = line.strip().split(",")
                l1 = l[:2]
                if len(l) > 2:
                    for _ in l[2:]:
                        w1 = float(_)
                        l1.append(w1)
                edgelist.append(l1)
    return node_types, edgelist


def check_config_validity(config):
    #### Set default values
    config = return_default_values(config)

    if config["task"] not in POSSIBLE_TASKS:
        raise ValueError("Task {} not supported.".format(config["task"]))

    if "test" in config["task"]:
        if "train" not in config["task"]:
            if config["embeddings_file"] == "" or (
                config["embeddings_file"] != ""
                and not os.path.exists(config["embeddings_file"])
            ):
                raise IOError(
                    "Embeddings file {} not found".format(config["embeddings_file"])
                )
        if config["experiment_type"] in ["ER", "SM"]:
            if not config["match_file"] or (
                config["match_file"] != "" and not os.path.exists(config["match_file"])
            ):
                raise IOError(
                    "Test file {} not found. "
                    "Tests require a valid Ground Truth file.".format(
                        config["match_file"]
                    )
                )
            if config["experiment_type"] == "SM":
                if "dataset_file" not in config or (
                    config["dataset_file"] != ""
                    and not os.path.exists(config["dataset_file"])
                ):
                    raise IOError(
                        "Dataset file {} not found. "
                        "SM tests require a valid dataset file.".format(
                            config["dataset_file"]
                        )
                    )

        elif config["experiment_type"] == "EQ":
            if not config["test_dir"] or (
                config["test_dir"] != "" and not os.path.exists(config["test_dir"])
            ):
                raise IOError("Test directory {} not found.".format(config["test_dir"]))
            if len(os.listdir(config["test_dir"])) == 0:
                raise IOError("Test directory {} is empty.".format(config["test_dir"]))
        else:
            raise ValueError(
                "Unknown experiment type {}".format(config["experiment_type"])
            )
    if "train" in config["task"]:
        try:
            config["sentence_length"] = int(config["sentence_length"])
        except ValueError:
            raise ValueError("Expected integer sentence_length value.")
        if not config["sentence_length"] > 0:
            raise ValueError("Sentence length must be > 0.")

        try:
            config["n_sentences"] = int(config["n_sentences"])
            if not config["n_sentences"] > 0:
                raise ValueError("Number of sentences must be > 0.")
        except ValueError:
            if config["n_sentences"] != "default":
                raise ValueError('Expected integer n_sentences value, or "default".')

        try:
            config["n_dimensions"] = int(config["n_dimensions"])
        except ValueError:
            raise ValueError("Expected integer n_dimensions value.")
        if not config["n_dimensions"] > 0:
            raise ValueError("Number of dimensions must be > 0.")

        try:
            config["window_size"] = int(config["window_size"])
        except ValueError:
            raise ValueError("Expected integer window_size value.")
        if not 0 < config["window_size"] <= config["sentence_length"]:
            raise ValueError("Window size must be between 0 and sentence_length")

    try:
        config["ntop"] = int(config["ntop"])
    except ValueError:
        raise ValueError("Expected integer ntop value.")
    if not config["ntop"] > 0:
        raise ValueError("Number of neighbors to be chosen must be > 0.")

    try:
        config["ncand"] = int(config["ncand"])
    except ValueError:
        raise ValueError("Expected integer ncand value.")
    if not 0 < config["ncand"] <= config["ntop"]:
        raise ValueError("Number of candidates must be between 0 and n_top.")

    try:
        config["sampling_factor"] = float(config["sampling_factor"])
    except ValueError:
        raise ValueError("Expected real sampling_factor value.")
    if not 1 > config["sampling_factor"] >= 0:
        raise ValueError("Sampling factor must be in [0,1).")

    if config["walks_strategy"] not in ["basic", "replacement"]:
        raise ValueError("Unknown walks strategy {}.".format(config["walks_strategy"]))
    if config["numeric"] not in ["no", "only", "all"]:
        raise ValueError("Unknown numeric strategy {}.".format(config["numeric"]))
    if config["training_algorithm"] not in ["word2vec", "fasttext"]:
        raise ValueError(
            "Unknown training algorithm {}.".format(config["training_algorithm"])
        )
    if config["learning_method"] not in ["skipgram", "CBOW"]:
        raise ValueError("Unknown learning method {}".format(config["learning_method"]))
    for key in [
        "backtrack",
        "write_walks",
        "compression",
        "intersection",
        "mlflow",
        "repl_strings",
        "repl_numbers",
    ]:
        config = _convert_to_bool(config, key)

    if "epsilon" in config:
        try:
            config["epsilon"] = float(config["epsilon"])
        except ValueError:
            print("Epsilon must be a float.")
            raise ValueError

    if config["intersection"]:
        if "dataset_file" not in config:
            raise ValueError(
                "A dataset file must be provided to perform intersection. "
            )
        if not os.path.exists(config["dataset_file"]):
            raise IOError("Dataset file {} not found.".format(config["dataset_file"]))

    if "flatten" in config:
        try:
            _convert_to_bool(config, "flatten")
        except ValueError:
            pass

    if config["mlflow"] and MLFLOW_NOT_FOUND:
        warnings.warn(
            "Package mlflow was not found. mlflow logging will not be available."
        )
        config["mlflow"] = False

    #### Path checks
    if "train" in config["task"] and not os.path.exists(config["input_file"]):
        raise IOError("Input file {} not found.".format(config["input_file"]))
    if not os.path.exists(config["dataset_info"]):
        raise IOError("Info file {} not found.".format(config["dataset_info"]))
    if config["walks_strategy"] == "replacement" and not os.path.exists(
        config["similarity_file"]
    ):
        raise IOError("Replacement strategy requires a similarity file.")
    if "walks_file" in config and config["walks_file"]:
        if not os.path.exists(config["walks_file"]):
            raise IOError("Walks file {} not found.".format(config["walks_file"]))
        if os.path.getsize(config["walks_file"]) == 0:
            raise IOError("Walks file is empty.")

    ###### WARNINGS
    if int(config["n_dimensions"]) != 300:
        warnings.warn(
            "Number of dimensions different from default (300): {}".format(
                config["n_dimensions"]
            )
        )
    if int(config["window_size"]) != 5:
        warnings.warn(
            "Window size different from default (5): {}".format(config["window_size"])
        )
    if config["walks_strategy"] == "basic" and config["numeric"] != "no":
        config["numeric"] = "no"
        warnings.warn("Basic random walks require no replacement strategy.")

    return config


def find_frequencies(configuration):
    with open(configuration["dataset_info"], "r") as fp:
        for i, line in enumerate(fp):
            path, length = line.strip().split(",")
            df = pd.read_csv(path)
            values, counts = np.unique(df.values.ravel(), return_counts=True)
