from EmbDI.utils import *
from EmbDI.graph import graph_generation
from EmbDI.sentence_generation_strategies import random_walks_generation
import pandas as pd


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
walks = random_walks_generation(configuration, df, graph)

