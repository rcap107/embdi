import argparse
import warnings
from operator import itemgetter

import gensim.models as models
import pandas as pd


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input_file', action='store', required=True, type=str, help='Input embeddings file.')
    parser.add_argument('-d', '--dataset_file', action='store', required=True, type=str, help='Input dataset.')
    parser.add_argument('-m', '--match_file', action='store', required=True, type=str)

    return parser.parse_args()


def read_matches(match_file):
    with open(match_file, 'r', encoding='utf-8') as fp:
        md = {}
        for idx, line in enumerate(fp):
            t = line.strip().split(',')
            if t[0] not in md:
                md[t[0]] = {t[1]}
            else:
                md[t[0]].add(t[1])
                # matches.append(t)
    return md


def _clean_embeddings(emb_file, matches):
    gt = set()
    for k, v in matches.items():
        gt.add(k)
        for _ in v:
            gt.add(_)

    with open(emb_file, 'r') as fp:
        s = fp.readline()
        _, dimensions = s.strip().split(' ')
        viable_idx = []
        for idx, row in enumerate(fp):
            r = row.split(' ', maxsplit=1)[0]
            # r = 'cid__' + r
            if r.startswith('cid__'):
                r = r.strip('cid__')
            if r in gt:
                viable_idx.append(row.strip('cid__'))

    f = 'pipeline/dump/sm_dump.emb'
    with open(f, 'w', encoding='utf-8') as fp:
        fp.write('{} {}\n'.format(len(viable_idx), dimensions))
        for _ in viable_idx:
            fp.write(_)
    return f


def _infer_prefix(df):
    columns = df.columns
    prefixes = tuple([_.split('_') for _ in columns])
    if len(prefixes) > 2:
        return None
    else:
        return list(prefixes)


def _match(candidates, maxrank=3):
    to_be_matched = list(candidates.keys())
    misses = {k: 0 for k in candidates}

    mm = []

    while len(to_be_matched) > 0:
        tbm = to_be_matched.copy()
        for item in tbm:
            if item not in to_be_matched:
                continue
            else:
                if misses[item] > maxrank:
                    to_be_matched.remove(item)
                    continue
                else:
                    closest_list = candidates[item]
                    if len(closest_list) > 0:
                        for idx in range(len(closest_list)):
                            closest_to_item = closest_list[idx]
                            reciprocal_closest_list = candidates[closest_to_item]
                            reciprocal_closest = reciprocal_closest_list[0]
                            if closest_to_item in to_be_matched and reciprocal_closest == item:
                                to_be_matched.remove(item)
                                to_be_matched.remove(closest_to_item)
                                mm.append((item, closest_to_item))
                                for k in candidates:
                                    if item in candidates[k]:
                                        candidates[k].remove(item)
                                    if closest_to_item in candidates[k]:
                                        candidates[k].remove(closest_to_item)
                                break
                            else:
                                misses[item] += 1
                    else:
                        to_be_matched.remove(item)
    return mm


def _extract_candidates(wv, dataset):
    candidates = []
    for _1 in range(len(dataset.columns)):
        for _2 in range(0, len(dataset.columns)):
            if _1 == _2:
                continue
            c1 = f'{dataset.columns[_1]}'
            c2 = f'{dataset.columns[_2]}'
            # c1 = f'cid__{dataset.columns[_1]}'
            # c2 = f'cid__{dataset.columns[_2]}'
            try:
                rank = wv.distance(c1, c2)
                tup = (c1, c2, rank)
                candidates.append(tup)
            except KeyError:
                continue
    cleaned = []
    for k in candidates:
        prefix = k[0].split('_')[0]
        if not k[1].startswith(prefix):
            cleaned.append(k)

    cleaned_sorted = sorted(cleaned, key=itemgetter(0, 2), reverse=False)

    candidates = {}
    for value in cleaned_sorted:
        v1, v2, rank = value
        if v1 not in candidates:
            candidates[v1] = [v2]
        else:
            candidates[v1].append(v2)

    return candidates


def _produce_match_results(candidates):
    match_results = _match(candidates)

    match_results = [sorted(_) for _ in match_results]

    # refactored_match_results = [(int(_[0].split('_')[1]), int(_[1].split('_')[1])) for _ in match_results]
    refactored_match_results = match_results
    return refactored_match_results


def match_columns(dataset, embeddings_file):
    emb_file = _clean_embeddings(embeddings_file)
    if emb_file is None:
        return []
    wv = models.KeyedVectors.load_word2vec_format(emb_file, unicode_errors='ignore')
    # print('Model built from file {}'.format(embeddings_file))
    candidates = _extract_candidates(wv, dataset)

    match_results = _produce_match_results(candidates)

    return match_results


def schema_matching(embeddings_file, configuration):
    dataset = pd.read_csv(configuration['dataset_file'])
    print('# Executing SM tests.')
    match_file = configuration['match_file']
    ground_truth = read_matches(match_file)
    emb_file = _clean_embeddings(embeddings_file, ground_truth)

    wv = models.KeyedVectors.load_word2vec_format(emb_file, unicode_errors='ignore')
    # print('Model built from file {}'.format(embeddings_file))
    candidates = _extract_candidates(wv, dataset)

    match_results = _produce_match_results(candidates)

    count_hits = 0
    gt = 0
    for item in match_results:
        # gt += 1
        left = item[0]
        right = item[1]
        if left in ground_truth:
            gt+=1
            if right in ground_truth[left]:
                count_hits += 1
    if len(match_results) > 0:
        precision = count_hits/len(match_results)
    else:
        precision = 0

    if gt > 0:
        recall = count_hits/gt
    else:
        warnings.warn(f'No hits found. There may be a problem with the ground truth file {match_file},\n '
                      f'or with the input dataset {configuration["dataset_file"]}.')
        recall = 0
    try:
        f1_score = 2 * (precision * recall) / (precision + recall)
    except ZeroDivisionError:
        f1_score = 0

    result_dict = {
        'P': precision,
        'R': recall,
        'F': f1_score,
    }
    print('P\tR\tF')
    for _ in result_dict.values():
        print('{:.4f}\t'.format(_*100), end='')
    print('')

    return result_dict