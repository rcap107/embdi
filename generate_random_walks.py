'''
This script is used to generate random walks starting from a given edgelist without the overhead required when running
the full algorithm. The parameters used here are the same as what is used in the main algorithm, so please refer to the
readme for more details.

@author: riccardo cappuzzo

'''

from EmbDI.utils import *
from EmbDI.graph import graph_generation
from EmbDI.sentence_generation_strategies import random_walks_generation
import pandas as pd

import argparse

def parse_args(args):
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input_file', required=True)
    parser.add_argument('-o','--output_file', required=True)
    parser.add_argument('')


# Default parameters
configuration = {
    'walks_strategy': 'basic',
    'flatten': 'all',
    'input_file': 'small_example.edgelist',
    'n_sentences': 'default',
    'sentence_length': 10,
    'write_walks': True,
    'intersection': False,
    'backtrack': True,
    'output_file': 'small_example',
    'repl_numbers': False,
    'repl_strings': False,
    'follow_replacement': False,
    'mlflow': False
}

df = pd.read_csv('pipeline/datasets/small_example.csv')

prefixes, edgelist = read_edgelist(configuration['input_file'])

graph = graph_generation(configuration, edgelist, prefixes, dictionary=None)
if configuration['n_sentences'] == 'default':
    #  Compute the number of sentences according to the rule of thumb.
    configuration['n_sentences'] = graph.compute_n_sentences(int(configuration['sentence_length']))
walks = random_walks_generation(configuration, graph)

