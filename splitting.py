import networkx as nx
import gensim.models as models
import numpy as np
import scipy
import os
import pandas as pd


def isrid(val):
    stripped = val.strip('idx__')
    try:
        s = int(stripped)
        return s
    except ValueError:
        return None

def split(dsfile, info_file, embeddings_file):
    dfs = []
    emb_keys = []
    n_dimensions = 300
    with open(embeddings_file, 'r') as fp:
        for idx, line in enumerate(fp):
            if idx > 0:
                key, vector = line.split(' ', maxsplit=1)
                emb_keys.append(key)
    with open(info_file, 'r') as fp:
        _, n_lines = fp.readline().strip().split(',')
        # for idx, line in enumerate(fp):
        #     _, n_lines = line.strip().split(',')

    n_lines = int(n_lines)
    dataset_start = 0
    dataset_end = n_lines
    df = pd.read_csv(dsfile)

    df1 = df[:n_lines]
    df2 = df[n_lines:]

    common_values = set()
    for idx, _df in enumerate([df1, df2]):
        fp_emb = open(embeddings_file, 'r')
        for col in _df.columns:
            _df[col] = _df[col].astype('object')
        uniques = list(set(_df.values.ravel()))
        uniques = list(filter(lambda v: v==v, uniques))
        new = []
        for c in uniques:
            if c is np.nan or c == '':
                continue
            try:
                float_c = float(c)
                try:
                    new_c = 'tn__' + str(int(float_c))
                except OverflowError:
                    new_c = 'tn__' + str(c)
                new.append(new_c)
            except ValueError:
                try:
                    split = c.split('_')
                except AttributeError:
                    continue
                for s in split:
                    new_c = 'tt__' + str(s)
                    new.append(new_c)
                if len(split) > 1:
                    new_c = 'tt__' + str(c)
                    new.append(new_c)
        uniques = new

        temp_name = os.path.basename(output_file) + '_split{}'.format(idx + 1)
        fname = 'pipeline/embeddings/split/{}.emb'.format(temp_name)
        print('Writing on file {}'.format(fname))
        fp = open(fname, 'w')
        lines = []
        c1=0
        for i, line in enumerate(fp_emb):
            if i > 0:
                t = line.split()[0]
                if len(line.split()) != n_dimensions + 1:
                    print(line)
                    raise ValueError('Wrong number of values on line {}'.format(idx))
                    # print(line)
                    continue
                trid = isrid(t)
                if trid is not None:
                    if dataset_start <= trid < dataset_end:
                        lines.append(line)
                        c1 += 1
                else:
                    if t in uniques:
                        #                         TODO REMEMBER TO ADD COLUMNS BACK IN
                        lines.append(line)
                        c1 += 1
        dataset_start = n_lines
        dataset_end = len(df)
        fp.write('{} {}\n'.format(c1, n_dimensions))
        for _ in lines:
            fp.write(_)

        fp_emb.close()

if __name__ == '__main__':
    embeddings_file = 'pipeline/embeddings/amazon_google-edgelist-without-compression.emb'
    dsfile = 'pipeline/datasets/amazon_google/amazon_google-master.csv'
    output_file = 'pipeline/experiments/rotation/amazon_google-rotation'
    dataset_info = 'pipeline/info/info-amazon_google.txt'

    split(dsfile, dataset_info, embeddings_file)