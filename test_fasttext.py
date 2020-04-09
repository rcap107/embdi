import pandas as pd
import fasttext
from EmbDI.entity_resolution import entity_resolution
import csv
import os.path as osp
import datetime as dt
import numpy as np
from sklearn.decomposition import PCA
import sys


def concatenate_word_embs(row, model):
    emb = []
    for _ in row:
        item = str(_)
        emb += model.get_sentence_vector(item).tolist()
    return [str(_) for _ in np.array(emb)]


t_start = dt.datetime.now()
dataset_name = 'amazon_google'

print('Loading dataset {}'.format(dataset_name))
info_file = 'pipeline/info/info-{}.txt'.format(dataset_name)
generated_emb_file = 'pipeline/embeddings/fasttext/{}-full-fasttext.emb'.format(dataset_name)

fi = 'pipeline/datasets/{}/{}-master.csv'.format(dataset_name, dataset_name)
df = pd.read_csv(fi)

configuration = {
    'ntop': 10,
    'ncand': 1,
    'indexing': 'basic',
    'match_file': 'pipeline/matches/matches-{}.txt'.format(dataset_name),
    'epsilon': 0.1,
    'num_trees': 250,
}


# Build row embeddings using the fasttext pretrained model. 
if 1:
    print('Loading fasttext model...')
    model = fasttext.load_model('/home/spoutnik23/phd/datasets/cc.en.300.bin')
    print('Model loaded.')

    print('Generating new row embeddings...')
    new_emb = {}
    tot_rows = 0
    with open(generated_emb_file, 'w') as fp:
        for idx, row in df.iterrows():
            for w in row:
                item = str(w)
                vector = model.get_sentence_vector(item).tolist()
                s = '{} '.format(item) + ' '.join([str(_) for _ in vector]) + '\n'
                tot_rows+=1
                fp.write(s)
            # vector = concatenate_word_embs(row, model)
            # s = 'idx_{} '.format(idx) + ' '.join(vector) + '\n'
            # fp.write(s)
            # tot_rows += 1
        print(idx)
        fp.seek(0)
        fp.write('{} {}\n'.format(tot_rows, 300))

if 0:
    # Execute the entity resolution task
    result_dict = entity_resolution(generated_emb_file, configuration, df=df, info_file=info_file)

    t_end = dt.datetime.now()
    # Print the results.
    print('\t'.join(result_dict.keys()))
    print('\t'.join([str(x) for x in result_dict.values()]))
    d = t_end-t_start
    print('Time required: {}'.format(d.total_seconds()))
