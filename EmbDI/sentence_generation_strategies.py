import datetime
import random

from tqdm import tqdm

from EmbDI.graph import Node
from EmbDI.utils import *


class RandomWalk:
    def __init__(
        self,
        graph_nodes,
        starting_node_name,
        sentence_len,
        backtrack,
        uniform=True,
        repl_strings=True,
        repl_numbers=True,
        follow_replacement=False,
    ):
        starting_node = graph_nodes[starting_node_name]
        first_node_name = starting_node.get_random_start()
        if first_node_name != starting_node_name:
            self.walk = [first_node_name, starting_node_name]
        else:
            self.walk = [starting_node_name]
        current_node_name = starting_node_name
        current_node = graph_nodes[current_node_name]
        sentence_step = len(self.walk)
        while sentence_step < sentence_len:
            if uniform:
                current_node_name = current_node.get_random_neighbor()
            else:
                current_node_name = current_node.get_weighted_random_neighbor()

            if repl_numbers:
                current_node_name = self.replace_numeric_value(current_node_name, graph_nodes)
            if repl_strings:
                current_node_name, replaced_node = self.replace_string_value(graph_nodes[current_node_name])
            else:
                replaced_node = current_node_name
            if not backtrack and current_node_name == self.walk[-1]:
                continue
            if follow_replacement:
                current_node_name = replaced_node

            current_node = graph_nodes[current_node_name]
            if not current_node.node_class["isappear"]:
                continue
            else:
                if replaced_node != current_node_name:
                    self.walk.append(replaced_node)
                else:
                    self.walk.append(current_node_name)
            sentence_step += 1

    def get_walk(self):
        return self.walk

    def get_reversed_walk(self):
        return self.walk[::-1]

    def get_both_walks(self):
        return [self.get_walk(), self.get_reversed_walk()]

    def replace_numeric_value(self, value, nodes):
        if nodes[value].numeric:
            try:
                value = int(value)
            except ValueError:
                return value
            new_val = np.around(np.random.normal(loc=value, scale=1))
            cc = 0
            try:
                new_val = int(new_val)
            except OverflowError:
                return str(value)
            while (
                new_val not in nodes.keys() and str(new_val) not in nodes.keys() and float(new_val) not in nodes.keys()
            ):
                if cc > 1:
                    return str(value)
                new_val = np.around(np.random.normal(loc=value, scale=1))
                cc += 1
            return str(int(new_val))
        else:
            return value

    def replace_string_value(self, value: Node):
        if len(value.similar_tokens) > 1:
            return value.name, value.get_random_replacement()
        else:
            return value.name, value.name


def extract_numeric_rep(value, keys_array):
    new_val = np.around(np.random.normal(loc=value, scale=1))
    cc = 0

    try:
        tmp = int(new_val)
    except OverflowError:
        return value
    while (
        new_val not in keys_array
        and str(new_val) not in keys_array
        and str(tmp) not in keys_array
        and float(tmp) not in keys_array
    ):
        if cc > 1:
            return value
        new_val = np.around(np.random.normal(loc=value, scale=1))
        cc += 1
    return str(int(new_val))


def generate_walks(parameters, graph, intersection=None):
    sentences = []
    n_sentences = int(float(parameters["n_sentences"]))
    strategies = parameters["walks_strategy"]
    sentence_length = int(parameters["sentence_length"])
    backtrack = parameters["backtrack"]

    if intersection is None:
        intersection = set(graph.cell_list)
        n_cells = len(intersection)
    else:
        n_cells = len(intersection)
    random_walks_per_node = n_sentences // n_cells

    sentence_distribution = dict(zip([strat for strat in strategies], [0 for _ in range(len(strategies))]))

    # ########### Random walks ############
    # print('Generating random walks.')

    walks_file = "pipeline/walks/" + parameters["output_file"] + ".walks"

    if parameters["write_walks"]:
        fp_walks = open(walks_file, "w")
    t2 = datetime.datetime.now()
    str_start_time = t2.strftime(TIME_FORMAT)
    print(OUTPUT_FORMAT.format("Generating basic random walks.", str_start_time))
    sentence_counter = 0

    count_cells = 0

    if random_walks_per_node > 0:
        pbar = tqdm(desc="# Sentence generation progress: ", total=len(intersection) * random_walks_per_node)
        for cell in intersection:
            # if cell in intersection:
            r = []
            for _r in range(random_walks_per_node):
                w = RandomWalk(
                    graph.nodes,
                    cell,
                    sentence_length,
                    backtrack,
                    graph.uniform,
                    repl_numbers=parameters["repl_numbers"],
                    repl_strings=parameters["repl_strings"],
                )

                r.append(w.get_walk())

            if parameters["write_walks"]:
                if len(r) > 0:
                    ws = [" ".join(_) for _ in r]
                    s = "\n".join(ws) + "\n"
                    fp_walks.write(s)
                else:
                    pass
            else:
                sentences += r
            sentence_counter += random_walks_per_node
            count_cells += 1
            pbar.update(random_walks_per_node)
        pbar.close()

    needed = n_sentences - sentence_counter
    if needed > 0:
        t_comp = datetime.datetime.now()
        str_comp_time = t_comp.strftime(TIME_FORMAT)
        print(OUTPUT_FORMAT.format("Completing fraction of random walks.", str_comp_time))

        with tqdm(total=needed, desc="# Sentence generation progress: ") as pbar:
            l_int = list(intersection)
            for count_cells in range(needed):
                cell = random.choice(l_int)
                # if cell in intersection:
                w = RandomWalk(
                    graph.nodes,
                    cell,
                    sentence_length,
                    backtrack,
                    graph.uniform,
                    repl_numbers=parameters["repl_numbers"],
                    repl_strings=parameters["repl_strings"],
                )
                sen = [w.get_walk()]

                for s in sen:
                    if parameters["write_walks"]:
                        ws = " ".join(s)
                        s = ws + "\n"
                        fp_walks.write(s)
                    else:
                        sentences += s
                sentence_counter += len(sen)
                pbar.update(1)

    sentence_distribution["basic"] = sentence_counter
    start_time = datetime.datetime.now()
    str_start_time = start_time.strftime(TIME_FORMAT)
    print(OUTPUT_FORMAT.format("Generation of random walks completed", str_start_time))
    print()

    if parameters["write_walks"]:
        fp_walks.close()
        return walks_file
    else:
        return sentences


def split_remaining_sentences(freq_row, freq_col):
    if freq_row == freq_col == 0:
        return [0, 0]
    elif freq_row == 0 and freq_col > 0:
        return [0, 1]
    elif freq_row > 0 and freq_col == 0:
        return [1, 0]
    else:
        rescaling_factor = freq_row / freq_col
        fraction_row = rescaling_factor / (rescaling_factor + 1)
        fraction_column = 1 - fraction_row
        return fraction_row, fraction_column


def random_walks_generation(configuration, graph):
    """
    Traverse the graph using different random walks strategies.
    :param configuration: run parameters to be used during the generation
    :param df: input dataframe
    :param graph: graph generated starting from the input dataframe
    :return: the collection of random walks
    """
    t1 = datetime.datetime.now()
    # Find values in common between the datasets.
    if configuration["intersection"]:
        print("# Finding overlapping values. ")
        # Expansion works better when all tokens are considered, rather than only the overlapping ones.
        if configuration["flatten"]:
            warnings.warn("Executing intersection while flatten = True.")
        # Find the intersection
        df = pd.read_csv(configuration["dataset_file"])
        intersecting_nodes = find_intersection_flatten(df, configuration["dataset_info"])
        intersection = graph.produce_intersection(intersecting_nodes)
        if len(intersection) == 0:
            warnings.warn("Datasets have no tokens in common. Falling back to no-intersection.")
            intersection = None
        else:
            print("# Number of common values: {}".format(len(intersection)))
    else:
        print("# Skipping search of overlapping values. ")
        intersection = None
    # intersection = None

    # Generating walks.
    walks = generate_walks(configuration, graph, intersection=intersection)
    t2 = datetime.datetime.now()
    dt = t2 - t1

    metrics.time_walks = dt.total_seconds()
    metrics.generated_walks = len(walks)
    return walks
