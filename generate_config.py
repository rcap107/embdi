import os

from EmbDI.edgelist import EdgeList

import pandas as pd

abbreviations = {
    "bb": "beer",
    "ia": "itunes_amazon",
    "im": "imdb_movielens",
    "wa": "walmart_amazon",
    "fz": "fodors_zagats",
    "ag": "amazon_google",
    "ds": "dblp_scholar",
    "da": "dblp_acm",
}


default_values = {
    "ntop": 10,
    "ncand": 1,
    "max_rank": 3,
    "follow_sub": "false",
    "smoothing_method": "no",
    "backtrack": "true",
    "training_algorithm": "word2vec",
    "write_walks": "true",
    "flatten": "all",
    "n_sentences": "default",
    "walks_strategy": "basic",
    "learning_method": "skipgram",
    "sentence_length": 60,
    "window_size": 3,
    "n_dimensions": 300,
    "experiment_type": "ER",
    "intersection": "false",
    "mlflow": "false",
    "repl_numbers": False,
    "repl_strings": False,
}

###### ER CASE

for ds in abbreviations:
    os.makedirs("pipeline/config_files/reproducibility/er", exist_ok=True)
    config = {
        "task": "train-test",
    }
    dataset = abbreviations[ds]
    print(f"ER - {dataset}")
    config["input_file"] = f"pipeline/edgelists/{dataset}-er-edgelist.txt"
    config["dataset_info"] = f"pipeline/info/info-{dataset}.txt"
    config["output_file"] = f"{dataset}-ER"
    config["flatten"] = "tt"
    config["experiment_type"] = "ER"
    config["match_file"] = f"pipeline/matches/er-matches/matches-{dataset}.txt"
    config["dataset_file"] = f"pipeline/datasets/{dataset}/{dataset}-master.csv"
    for k in default_values:
        if k not in config:
            config[k] = default_values[k]
    with open(f"pipeline/config_files/reproducibility/er/config-{dataset}-er", "w") as fp:
        for k, v in config.items():
            s = f"{k}:{v}\n"
            fp.write(s)
    pref = ["3#__tn", "3$__tt", "5$__idx", "1$__cid"]
    df = pd.read_csv(config["dataset_file"])
    # el = EdgeList(df, config['input_file'], pref,
    #               config['dataset_info'], flatten=False)


##### EQ CASE

for ds in abbreviations:
    os.makedirs("pipeline/config_files/reproducibility/eq", exist_ok=True)

    config = {
        "task": "train-test",
    }
    dataset = abbreviations[ds]
    print(f"EQ - {dataset}")
    config["input_file"] = f"pipeline/edgelists/{dataset}-eq-edgelist.txt"
    config["dataset_info"] = f"pipeline/info/info-{dataset}.txt"
    config["output_file"] = f"{dataset}-EQ"
    config["flatten"] = "false"
    config["experiment_type"] = "EQ"
    config["test_dir"] = f"pipeline/test_dir/{dataset}"
    config["intersection"] = "true"
    config["dataset_file"] = f"pipeline/datasets/{dataset}/{dataset}-master.csv"
    for k in default_values:
        if k not in config:
            config[k] = default_values[k]
    with open(f"pipeline/config_files/reproducibility/eq/config-{dataset}-eq", "w") as fp:
        for k, v in config.items():
            s = f"{k}:{v}\n"
            fp.write(s)
    pref = ["3#__tn", "3$__tt", "4$__idx", "1$__cid"]
    df = pd.read_csv(config["dataset_file"])
    # el = EdgeList(df, config['input_file'], pref,
    #               config['dataset_info'], flatten=False)

##### SM CASE

for ds in abbreviations:
    os.makedirs("pipeline/config_files/reproducibility/sm", exist_ok=True)

    config = {
        "task": "train-test",
    }
    dataset = abbreviations[ds]
    print(f"SM - {dataset}")
    config["input_file"] = f"pipeline/edgelists/{dataset}-sm-edgelist.txt"
    config["dataset_info"] = f"pipeline/info/info-{dataset}.txt"
    config["output_file"] = f"{dataset}-SM"
    config["flatten"] = "tt"
    config["experiment_type"] = "SM"
    config["test_dir"] = f"pipeline/test_dir/{dataset}"
    config["dataset_file"] = f"pipeline/datasets/{dataset}/{dataset}-master-sm.csv"
    config["match_file"] = f"pipeline/matches/sm-matches/sm-matches-{dataset}.txt"

    for k in default_values:
        if k not in config:
            config[k] = default_values[k]
    with open(f"pipeline/config_files/reproducibility/sm/config-{dataset}-sm", "w") as fp:
        for k, v in config.items():
            s = f"{k}:{v}\n"
            fp.write(s)
    pref = ["3#__tn", "3$__tt", "5$__idx", "1$__cid"]
    df = pd.read_csv(config["dataset_file"], low_memory=False)
    # el = EdgeList(df, config['input_file'], pref, flatten=False)
