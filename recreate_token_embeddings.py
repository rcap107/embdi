import gensim.models as models

import pandas as pd

import numpy as np


def average_word_embds(row, model):
    n = []
    for w in row:
        if (str(w) in model):
            n.append(model[str(w)])
    return np.average(n, axis=0)


embeddings_file = 'pipeline/embeddings/eqtest/amazon_google-ER-master-flatten.embs'

df = pd.read_csv('pipeline/datasets/amazon_google/amazon_google-master.csv')

unique_values = set(df.values.ravel().tolist())

wv = models.KeyedVectors.load_word2vec_format(embeddings_file, unicode_errors='ignore')

root, ext = embeddings_file.split('.')

c = 0
with open(root + '-recreated.' + ext, 'w') as fp:
    for idx, value in enumerate(unique_values):
        try:
            vec = wv.get_vector(value)
            s = '{} {}\n'.format(value, ' '.join([str(_) for _ in vec]))
        except KeyError:
            if isinstance(value, float):
                continue
            else:
                vec = average_word_embds(value, wv)
                s = '{} {}\n'.format(value, ' '.join([str(_) for _ in vec]))
        fp.write(s)
        c+=1
    fp.seek(0)
    fp.write('{} {}\n'.format(c, 300))
