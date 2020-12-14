import datetime

from EmbDI.utils import *
from EmbDI.graph import Node

from tqdm import tqdm

import random
# import mlflow

import multiprocessing as mp

import cProfile, pstats

global G

class RandomWalk:
    def __init__(self, graph_nodes, starting_node_name, sentence_len, backtrack, uniform, repl_strings=True, repl_numbers=True,
                 follow_replacement=False):
        # self.walk = []
        starting_node = graph_nodes[starting_node_name]
        first_node_name = starting_node.get_random_start()
        if first_node_name != starting_node_name:
            self.walk = [first_node_name, starting_node_name]
        else:
            self.walk = [starting_node_name]
        current_node_name = starting_node_name
        current_node = graph_nodes[current_node_name]
        sentence_step = 0
        while sentence_step < sentence_len - 1:
            # if False:
            if uniform:
                current_node_name = current_node.get_random_neighbor()
            else:
                current_node_name = current_node.get_weighted_random_neighbor()

            if repl_numbers:
                current_node_name = self.replace_numeric_value(current_node_name, G.nodes)
            if repl_strings:
                current_node_name, replaced_node= self.replace_string_value(G.nodes[current_node_name])
            else:
                replaced_node = current_node_name
            if not backtrack and current_node_name == self.walk[-1]:
                continue
            if follow_replacement:
                current_node_name = replaced_node

            current_node = G.nodes[current_node_name]
            if not current_node.node_class['isappear']:
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

    def get_sampled_walk(self, seq):

        def orderedSampleWithoutReplacement(seq, k):
            if not 0 <= k <= len(seq):
                raise ValueError('Required that 0 <= sample_size <= population_size')

            numbersPicked = 0
            for i, number in enumerate(seq):
                prob = (k - numbersPicked) / (len(seq) - i)
                if random.random() < prob:
                    yield number
                    numbersPicked += 1

        return orderedSampleWithoutReplacement(seq, 50)

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
            while new_val not in nodes.keys() and str(new_val) not in nodes.keys() and float(new_val) not in nodes.keys():
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

    # idx = (np.abs(keys_array - value)).argmin()
    try:
        tmp = int(new_val)
    except OverflowError:
        return value
    while new_val not in keys_array and str(new_val) not in keys_array and \
            str(tmp) not in keys_array and float(tmp) not in keys_array:
        if cc > 1:
            # print(new_val)
            # print(value)
            return value
        new_val = np.around(np.random.normal(loc=value, scale=1))
        cc += 1
    return str(int(new_val))


def generate_walks(parameters, graph, intersection=None):
    sentences = []
    n_sentences = int(float(parameters['n_sentences']))
    strategies = parameters['walks_strategy']
    sentence_length = int(parameters['sentence_length'])
    backtrack = parameters['backtrack']

    if intersection is None:
        intersection = graph.cell_list
        n_cells = len(intersection)
    else:
        n_cells = len(intersection)
    random_walks_per_node = n_sentences//n_cells

    sentence_distribution = dict(zip([strat for strat in strategies], [0 for _ in range(len(strategies))]))

    # ########### Random walks ############
    # print('Generating random walks.')
    # cells_list = list(cells)

    t2 = datetime.datetime.now()
    str_start_time = t2.strftime(TIME_FORMAT)

    walks_file = 'pipeline/walks/' + parameters['output_file'] + '.walks'

    if parameters['write_walks']: fp_walks = open(walks_file, 'w')

    print(OUTPUT_FORMAT.format('Generating basic random walks.', str_start_time))
    sentence_counter = 0

    count_cells = 0
    # pool_size = 2
    pool_size = mp.cpu_count()
    def cb(result):
        r.append(result.get_walk())

    profiler = cProfile.Profile()
    profiler.enable()
    if random_walks_per_node > 0:

        pbar = tqdm(desc='Sentence generation progress', total=len(graph.cell_list)*random_walks_per_node)
        # for cell in tqdm(graph.cell_list):
        # with mp.Pool(pool_size) as pool:
        # pool = mp.Pool(pool_size)
        for cell in graph.cell_list:
            if cell in intersection:
                r=[]
                for _r in range(random_walks_per_node):

                    # pool.apply_async(RandomWalk, (graph, cell, sentence_length, backtrack,),kwds=
                    #                 {'repl_numbers' : parameters['repl_numbers'],
                    #                 'repl_strings' : parameters['repl_strings']},
                    #                  callback=cb)

                    w = RandomWalk(graph, cell, sentence_length, backtrack,
                                   repl_numbers=parameters['repl_numbers'],
                                   repl_strings=parameters['repl_strings'])

                    r.append(w.get_walk())

                if parameters['write_walks']:
                    if len(r) > 0:
                        ws = [' '.join(_) for _ in r]
                        s = '\n'.join(ws) + '\n'
                        fp_walks.write(s)
                    else: pass
                else:
                    sentences += r
                sentence_counter += random_walks_per_node
                count_cells += 1
                pbar.update(random_walks_per_node)

        # pool.close()
        # pool.join()
        #
        pbar.close()

    needed = n_sentences - sentence_counter
    # cells = list(graph.cell_list)
    if needed > 0:
        with tqdm(total=needed, desc='Completing fraction of random walks') as pbar:
            for count_cells in range(needed):
        # while needed > count_cells:
                cell = random.choice(graph.cell_list)
                if cell in intersection:
                    w = RandomWalk(graph, cell, sentence_length, backtrack,
                                   repl_numbers=parameters['repl_numbers'],
                                   repl_strings=parameters['repl_strings'])
                    sen = [w.get_walk()]
                    # sen.append(w.get_reversed_walk())
                    # sen = []
                    # sen.append(list(w.get_sampled_walk(w.get_walk())))
                    # sen.append(list(w.get_sampled_walk(w.get_reversed_walk())))

                    for s in sen:
                        if parameters['write_walks']:
                            ws = ' '.join(s)
                            s = ws  + '\n'
                            fp_walks.write(s)
                        else:
                            sentences += s
                    sentence_counter += len(sen)
                    pbar.update(1)

    sentence_distribution['basic'] = sentence_counter
    start_time = datetime.datetime.now()
    str_start_time = start_time.strftime(TIME_FORMAT)
    print(OUTPUT_FORMAT.format('Generation of random walks completed', str_start_time))


    profiler.disable()
    stats = pstats.Stats(profiler).sort_stats('tottime')
    stats.print_stats(20)

    if parameters['write_walks']:
        fp_walks.close()
        return walks_file
    else:
        return sentences

    raise ValueError

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


def random_walks_generation(configuration, df, graph):
    """
    Traverse the graph using different random walks strategies.
    :param configuration: run parameters to be used during the generation
    :param df: input dataframe
    :param graph: graph generated starting from the input dataframe
    :return: the collection of random walks
    """
    t1 = datetime.datetime.now()
    # Find values in common between the datasets.
    if configuration['intersection']:
        print('# Finding overlapping values. ')
        # Expansion works better when all tokens are considered, rather than only the overlapping ones.
        if configuration['flatten']:
            warnings.warn('Executing intersection while flatten = True.')
        # Find the intersection
        intersection = find_intersection_flatten(df, configuration['dataset_info'])
        if len(intersection) == 0:
            warnings.warn('Datasets have no tokens in common. Falling back to no-intersection.')
            intersection = None
        else:
            print('# Number of common values: {}'.format(len(intersection)))
    else:
        print('# Skipping search of overlapping values. ')
        intersection = None
        # configuration['with_rid'] = WITH_RID_FIRST

    # Generating walks.
    walks = generate_walks(configuration, graph, intersection=intersection)
    t2 = datetime.datetime.now()
    dt = t2 - t1

    if configuration['mlflow']:
        with mlflow.start_run(run_id=configuration['run_id']):
            # Reporting the intersection flag.
            if intersection is None:
                mlflow.log_param('intersection', False)
            else:
                mlflow.log_param('intersection', True)
            mlflow.log_metric('generated_walks', len(walks))
            mlflow.log_metric('time_walks', dt.total_seconds())
    metrics.time_walks = dt.total_seconds()
    metrics.generated_walks = len(walks)
    return walks
