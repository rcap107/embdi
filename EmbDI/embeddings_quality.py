import ast
import os

import gensim.models as models
import warnings
from itertools import chain
from EmbDI.utils import *

def _test_no_match_columns(model, list_files):
    correct = 0
    total = 0
    result_dict = {}

    for filename in list_files:
        basename, filnam = os.path.split(filename)
        print(filename)
        with open(filename, 'r') as fp:
            run_correct = 0
            run_total = 0
            for line in fp:
                try:
                    terms = ast.literal_eval(line)
                except ValueError:
                    # warnings.warn('Problem encountered while reading line {}'.format(line))
                    continue
                terms = [str(_) for _ in terms]
                expected = terms[-1]
                if len(terms) == 0:
                    continue
                try:
                    result = model.doesnt_match(terms)
                    if expected == result:
                        correct += 1
                        run_correct += 1
                    else:
                        _=1
                except ValueError:
                    pass
                total += 1
                run_total += 1
            try:
                perc_corr = run_correct / run_total * 100
            except ZeroDivisionError:
                warnings.warn('File {} contains no rows.'.format(filename))
            result_dict[filnam] = perc_corr
    try:
        result_dict['MA_avg'] = correct/total*100
    except ZeroDivisionError:
        result_dict['MA_avg'] = 0

    return result_dict


def _test_no_match_concept(model, list_files):
    correct = 0
    total = 0
    result_dict = {}
    for filename in list_files:
        basename, filnam = os.path.split(filename)

        with open(filename, 'r') as fp:
            run_correct = 0
            run_total = 0
            for line in fp:
                try:
                    terms = ast.literal_eval(line)
                except ValueError:
                    print(line)
                    quit()
                if len(terms) == 0:
                    warnings.warn('Problem encountered while reading line {}'.format(line))
                    continue
                terms = [str(_) for _ in terms]
                expected = terms[-1]
                try:
                    result = model.doesnt_match(terms)
                    if expected == result:
                        correct += 1
                        run_correct += 1
                except ValueError:
                    pass
                    # print(terms)

                total += 1
                run_total += 1
            try:
                perc_corr = run_correct / run_total * 100
            except ZeroDivisionError:
                warnings.warn('File {} contains no rows.'.format(filename))
            result_dict[filnam] = perc_corr
    try:
        result_dict['MC_avg'] = correct/total*100
    except ZeroDivisionError:
        result_dict['MC_avg'] = 0
    return result_dict


def _test_no_match_rows(model, list_files):
    correct = 0
    total = 0
    result_dict = {}
    for filename in list_files:
        basename, filnam = os.path.split(filename)
        with open(filename, 'r') as fp:
            run_correct = 0
            run_total = 0
            for line in fp:
                try:
                    terms = ast.literal_eval(line)
                except ValueError:
                    warnings.warn('Problem encountered while reading line {}'.format(line))
                    continue
                terms = [str(_) for _ in terms]
                expected = terms[-1]
                try:
                    result = model.doesnt_match(terms)
                    if expected == result:
                        correct += 1
                        run_correct += 1

                except ValueError:
                    pass
                total += 1
                run_total += 1
            try:
                perc_corr = run_correct / run_total * 100
            except ZeroDivisionError:
                warnings.warn('File {} contains no rows.'.format(filename))

            result_dict[filnam] = perc_corr
    try:
        result_dict['MR_avg'] = correct/total*100
    except ZeroDivisionError:
        result_dict['MR_avg'] = 0
    return result_dict


def embeddings_quality(embeddings_file, configuration):
    """Function used to test the quality of the embeddings provided in file 'embeddings_file'. The tests must already
    be provided in the directory 'test_dir' to be executed. Please refer to the readme for info about the test format.

    :param embeddings_file: path to the embeddings file to be tested
    :param test_dir: path to the directory that contains all the tests.
    """
    print('# Executing EQ tests.')
    test_dir = configuration['test_dir']
    test_dir = test_dir.strip('/') + '/'

    if configuration['training_algorithm'] == 'fasttext':
        wv = models.KeyedVectors.load(embeddings_file, mmap='r')
    else:
        wv = models.KeyedVectors.load_word2vec_format(embeddings_file, unicode_errors='ignore')
    sum_total = 0
    count_tests = 0
    result_col = {}
    result_con = {}
    result_row = {}

    nmr_tests = []
    nmcon_tests = []
    nmc_tests = []
    for fin in os.listdir(test_dir):
        if fin.startswith('nmr'):
            nmr_tests.append(test_dir + fin)
        elif fin.startswith('nmcon'):
            nmcon_tests.append(test_dir + fin)
        elif fin.startswith('nmc'):
            nmc_tests.append(test_dir + fin)
    if len(nmc_tests) == len(nmcon_tests) == len(nmr_tests) == 0:
        raise ValueError('No valid test files found. Exiting. ')

    if len(nmc_tests) > 0:
        print('# Testing columns.')
        result_col = _test_no_match_columns(wv, nmc_tests)
        sum_total += result_col['MA_avg']
        count_tests += 1
        print('# MA_avg: {:.2f}'.format(result_col['MA_avg']))
    else:
        warnings.warn('No valid nmc tests found. ')

    if len(nmr_tests) > 0:
        print('# Testing rows.')
        result_row = _test_no_match_rows(wv, nmr_tests)
        sum_total += result_row['MR_avg']
        count_tests += 1
        print('# MR_avg: {:.2f}'.format(result_row['MR_avg']))
    else:
        warnings.warn('No valid nmr tests found. ')
    if len(nmcon_tests) > 0:
        print('# Testing concepts.')
        result_con = _test_no_match_concept(wv, nmcon_tests)
        sum_total += result_con['MC_avg']
        count_tests += 1
        print('# MC_avg: {:.2f}'.format(result_con['MC_avg']))
    else:
        warnings.warn('No valid nmcon tests found. ')
    try:
        avg_results = sum_total / count_tests

    except ZeroDivisionError:
        print('No tests were executed.')
        avg_results = 0

    print('# EQ average: {:.2f}'.format(avg_results))

    result_dict = dict(chain.from_iterable(d.items() for d in (result_row, result_col, result_con)))
    _r = ['MA_avg', 'MR_avg', 'MC_avg', 'EQ_avg']
    result_dict['EQ_avg'] = avg_results
    print('\t'.join(_r))
    for k in _r:
        print(result_dict[k], end='\t')
    print()

    return result_dict

if __name__ == '__main__':
    emb_file = 'pipeline/embeddings/'
    test_dir = 'pipeline/test_dir/'
    embeddings_quality(emb_file, test_dir)
