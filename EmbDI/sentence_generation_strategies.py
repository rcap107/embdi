import datetime
import random

import networkx as nx
import numpy as np

from EmbDI.utils import *
from EmbDI.graph import Node

from tqdm import tqdm

class RandomWalk:
    def __init__(self, G, starting_node_name, sentence_len, backtrack, repl_strings=True, repl_numbers=True,
                 follow_replacement=False):
        # self.walk = []
        starting_node = G.nodes[starting_node_name]
        first_node_name = starting_node.get_random_start()
        if first_node_name != starting_node_name:
            self.walk = [first_node_name, starting_node_name]
        else:
            self.walk = [starting_node_name]
        current_node_name = starting_node_name
        current_node = G.nodes[current_node_name]
        sentence_step = 0
        while sentence_step < sentence_len - 1:
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

def basic_random_walk(G, starting_node_name, sentence_len, n_sentences, backtrack):
    walks = []
    for _ in range(n_sentences):
        starting_node = G.nodes[starting_node_name]
        first_node_name = starting_node.get_random_start()
        if first_node_name != starting_node_name:
            walk = [first_node_name, starting_node_name]
        else:
            walk = [starting_node_name]
        current_node_name = starting_node_name
        current_node = G.nodes[current_node_name]
        sentence_step = 0
        full_steps = 0
        while sentence_step < sentence_len - 1:
            full_steps += 1
            current_node_name = current_node.get_weighted_random_neighbor()
            if not backtrack and current_node_name == walk[-1]:
                continue
            current_node = G.nodes[current_node_name]
            if not current_node.node_class['isappear']:
                continue
            else:
                walk.append(current_node_name)
            sentence_step += 1
        walks.append(walk)

    return walks


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


def random_walk_replacement(G, starting_node, sentence_len, n_sentences, cell_type, weights_by_cell, tokens,
                            follow_sub=False, with_rid=WITH_RID_FIRST, with_cid=WITH_CID_ALL,
                            numeric='no'):
    walks = []
    # keys_array = np.asarray(weights_by_cell.keys())
    for _ in range(n_sentences):
        first_node = random.choice(weights_by_cell[starting_node])
        walk = [first_node, starting_node]
        current = starting_node

        for _ in range(sentence_len - 1):
            c = random.choice(weights_by_cell[current])
            if numeric != 'no':
                try:
                    # raise ValueError
                    float_c = float(c)
                    sub = extract_numeric_rep(float_c, weights_by_cell)
                    # sub = float_c
                except ValueError:
                    if numeric == 'only':
                        sub = c
                    else:
                        if tokens[c].n_similar > 1:
                            tk = tokens[c].similar_tokens
                            ds = tokens[c].similar_distance
                            sub = random.choices(tk, weights=ds, k=1)[0]
                        else:
                            sub = c
            else:
                if tokens[c].n_similar > 1:
                    tk = tokens[c].similar_tokens
                    ds = tokens[c].similar_distance
                    sub = random.choices(tk, weights=ds, k=1)[0]
                else:
                    sub = c
            if sub in cell_type:
                ctype = cell_type[sub]
            else:
                ctype = cell_type[c]
            if ctype == 'rid':
                if with_rid != WITH_RID_ALL:
                    pass
                else:
                    walk.append(sub)
            elif ctype == 'attr':
                if with_cid != WITH_CID_ALL:
                    pass
                else:
                    walk.append(sub)
            else:
                walk.append(sub)

            if follow_sub:
                current = sub
            else:
                current = c

        walks.append(walk)

    return walks


def generate_walks(parameters, graph, intersection=None, backtrack=True):
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

    if 'basic' in strategies:
        print(OUTPUT_FORMAT.format('Generating basic random walks.', str_start_time))
        sentence_counter = 0

        count_cells = 0
        pbar = tqdm(desc='Sentence generation progress', total=len(graph.cell_list)*random_walks_per_node)
        # for cell in tqdm(graph.cell_list):
        for cell in graph.cell_list:
            if cell in intersection:
                r = []
                for _r in range(random_walks_per_node):
                    w = RandomWalk(graph, cell, sentence_length, backtrack,
                                   repl_numbers=parameters['repl_numbers'],
                                   repl_strings=parameters['repl_strings'])
                    r.append(w.get_walk())
                # r = basic_random_walk(graph, cell, sentence_length, random_walks_per_node, backtrack)
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
                        r = w.get_walk()
                        if parameters['write_walks']:
                            ws = ' '.join(r)
                            s = ws  + '\n'
                            fp_walks.write(s)
                        else:
                            sentences += r
                        sentence_counter += len(r)
                        pbar.update(1)

        sentence_distribution['basic'] = sentence_counter
        start_time = datetime.datetime.now()
        str_start_time = start_time.strftime(TIME_FORMAT)
        print(OUTPUT_FORMAT.format('Generation of random walks completed', str_start_time))

    if parameters['write_walks']:
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


def generate_walks_driver(parameters, graph, cells, tokens=None, follow_sub=True, with_rid='first', with_cid='all',
                          df=None, intersection=None, numeric='no'):
    w, _ = generate_walks_old(parameters, graph, cells, tokens, follow_sub, with_rid, with_cid, df, intersection, numeric)
    walks = []
    for walk in w:
        walks.append([str(_) for _ in walk])
    return walks


def generate_walks_driver_debug(parameters, graph, follow_sub=True, with_rid='first', with_cid='all', df=None,
                                intersection=None, numeric='no', backtrack=True):
    w, _ = generate_walks(parameters, graph, follow_sub, intersection=intersection, backtrack=backtrack)
    return w
