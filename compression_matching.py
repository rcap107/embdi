import pandas as pd
import gensim.models as models
from EmbDI.utils import *

if __name__ == '__main__':
    uncompressed_file = 'pipeline/embeddings/amazon_google-edgelist-with-compression.emb'
    compressed_file = 'pipeline/embeddings/amazon_google-edgelist-without-compression.emb'

    edgelist_file = 'pipeline/experiments/amazon_google-edges-master.txt'

    df = pd.read_csv(edgelist_file, dtype=str, index_col=False)

    df, dictionary = dict_compression_edgelist(df, prefixes=['3#__tn', '3$__tt','5$__idx', '1$__cid'])
    undictionary = {v:k for k,v in dictionary.items()}
    el = df.values.tolist()

    uncomp = open(uncompressed_file, 'r')
    comp = open(compressed_file, 'r')

    l_uncomp, ndim = uncomp.readline().strip().split(' ')
    l_comp, ndim = comp.readline().strip().split(' ')

    words_uncomp = []

    ff = 'pipeline/experiments/ag_comp_concat.emb'
    with open(ff, 'w') as fp:
        fp.write('{} {}\n'.format(int(l_uncomp) + int(l_comp), ndim))
        for idx, ll in enumerate(uncomp):
            fp.write(ll)
        for idx, ll in enumerate(comp):
            fp.write(ll)
            words_uncomp.append(ll.split(' ', maxsplit=1)[0])

    conversion = {}

    for w in words_uncomp:
        split = w.split('_')
        pre = split[0]+'_'
        ss = split[2:]
        tmp = []
        for s in ss:
            tmp.append(undictionary[s])
        conversion[w] = '_'.join([pre] + tmp)

    uncomp.close()
    comp.close()

    # uncompressed_model = models.KeyedVectors.load_word2vec_format(uncompressed_file, unicode_errors='ignore')
    # compressed_model = models.KeyedVectors.load_word2vec_format(compressed_file, unicode_errors='ignore')
    model = models.KeyedVectors.load_word2vec_format(ff, unicode_errors='ignore')

    i = 0
    for val, match in conversion.items():
        ms1 = model.most_similar(val)
        m1 = [_[0] for _ in ms1]
        ms2 = model.most_similar(match)
        m2 = [_[0] for _ in ms2]

        print(val,match)
        print(m1, m2)
        for aa in m1:
            print(conversion[aa])
        if idx > 10:
            break
        idx+=1