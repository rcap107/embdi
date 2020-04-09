#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec  4 10:04:06 2019

@author: spoutnik23
"""

import itertools
import os
import pandas as pd
import numpy as np


default_values = {
    'experiment_type':'EQ',
    'smoothing_method':'no',
    'smooth_k':'0.2',
    'inverse_k': '0.5',
    'smooth_t':'200',
    'log_base':'10',
    'window_size':'3',
    'n_dimensions':'300',
    'sentence_length':'60',
    'walks_strategy':'basic',
    'ntop':'10',
    'ncand':'1',
    'max_rank':'3',
    'learning_method':'skipgram',
    'training_algorithm':'word2vec',
    'follow_sub':'',
    'task': 'train-test',
    'with_cid': 'all',
    'with_rid': 'first',
    'numeric': 'no',
    'backtrack': True,
    'n_sentences': ''
}


def _get_match_files(basedir):
    match_files = {
        'EQ': 'pipeline/test_dir/{}'.format(basedir),
        'ER': 'pipeline/matches/matches-{}.txt'.format(basedir),
        'SM': 'pipeline/matches/sm-matches-{}.txt'.format(basedir),
    }
    return match_files


def _cartesian_product(*args):
    prod = []
    for _ in itertools.product(*args):
        prod.append(_)
    return prod


def _read_variables_file(var_file):
    variables = {}
    with open(var_file, 'r') as fp:
        for i, line in enumerate(fp):
            parameter, values = line.strip().split(':', maxsplit=1)
            variables[parameter] = values.split(',')
    for default_var in default_values:
        if default_var not in variables or variables[default_var][0] == '':
            variables[default_var] = [default_values[default_var]]
    return variables


def _create_name(configuration, variables):
    base = configuration['output_file']
    endname = base + '-' + configuration['experiment_type']
    for k in configuration:
        if k in variables and \
                len(variables[k]) > 1 and \
                k not in ['dataset', 'test']:
            endname = '{}-{}-{}'.format(endname, k, configuration[k])
    endname = endname.replace(',', '-')
    configuration['output_file'] = endname
    return configuration


def _write_config(configuration, match_files, configuration_dir):
    test = configuration['experiment_type']
    if test == 'EQ':
        # eq_graph = ''
        configuration['test_dir'] = match_files[test]
        configuration['match_file'] = ''
    elif test == 'ER':
        configuration['test_dir'] = ''
        configuration['match_file'] = match_files[test]
    else:
        configuration['test_dir'] = ''
        configuration['match_file'] = match_files[test]
        er_embfile = 'pipeline/embeddings/' + configuration['output_file'].replace('SM', 'ER') + '.emb'
        configuration['embeddings_file'] = er_embfile

    with open(configuration_dir.strip('/') + '/' + configuration['output_file'].replace('.', ''), 'w') as fp:
        for k in configuration:
            s = '{}:{}\n'.format(k, configuration[k])
            fp.write(s)


def _handle_smoothing_method(configuration):
    if configuration['smoothing_method'] == 'smooth':
        _smooth = (configuration['smooth_k'], configuration['smooth_t'])
        s = 'smooth,{},{}'.format(*_smooth)
        configuration['smoothing_method'] = s
    elif configuration['smoothing_method'] == 'log':
        s = 'log,{}'.format(configuration['log_base'])
        configuration['smoothing_method'] = s
    elif configuration['smoothing_method'] == 'inverse_smooth':
        s = 'inverse_smooth,{}'.format(configuration['inverse_k'])
        configuration['smoothing_method'] = s
    elif configuration['smoothing_method'] == 'no':
        pass
    else:
        raise ValueError('Unknown smoothing method {}'.format(configuration['smoothing_method']))
    return configuration


def _compute_n_sentences(df_file):
    df = pd.read_csv(df_file, dtype=str)
    n_rows = len(df)
#    n_values = len(set(df.values.ravel().tolist()))
    uniques = []
    n_col = len(df.columns)
    for col in df.columns:
        uniques+=df[col].unique().tolist()
    n_values = len(set(uniques))
    return (n_rows + n_values + n_col) * 10


def main_configuration(var_file=None, destination_dir=None):
    if not var_file:
        var_file = 'pipeline/config_files/var1'
    if not destination_dir:
        destination_dir = 'pipeline/config_files/'
    variables = _read_variables_file(var_file)
    variables_flat = [__ for __ in variables.values()]
    prod_tmp = _cartesian_product(*variables_flat)
    prod = []
    for ds in variables['dataset']:
        drop = False
        for val in prod_tmp:
            if val[0] == ds:
                if val[1] == 'no':
                    if not drop:
                        drop = True
                        prod.append(val)
                    else:
                        continue
                else:
                    prod.append(val)
    for _test in prod:
        configuration = dict(zip(variables.keys(), _test))
        current_file = '{}-master'.format(configuration['dataset'])

        basefile = configuration['dataset']
        basedir = basefile.split('-')[0]

        input_file = 'pipeline/datasets/' + current_file + '.csv'
        dataset_info = 'pipeline/info/info-' + basedir + '.txt'
        configuration['output_file'] = current_file
        configuration['input_file'] = input_file
        configuration['dataset_info'] = dataset_info
        configuration = _handle_smoothing_method(configuration)
        if configuration['n_sentences'] in ['','default']:
             configuration['n_sentences'] = _compute_n_sentences(input_file)
        configuration = _create_name(configuration, variables)
        _write_config(configuration, _get_match_files(basedir),
                      configuration_dir=destination_dir)


if __name__ == '__main__':
    main_configuration('pipeline/config_files/var_beer')
