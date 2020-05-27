import pandas as pd

df = pd.read_csv('pipeline/datasets/imdb_movielens/imdb_movielens-master.csv')

df_values = df.values.ravel().tolist()
val2 = [_1 for _2 in df_values for _1 in str(_2).strip().split(' ') ]
setval = set(val2)

first_idx = len(df)

with open('reviews-all.walks') as fp:
    with open('reviews-filter-edgelist.txt', 'w') as fo:
        for idx, line in enumerate(fp):
            rid = first_idx + idx
            for val in line.strip().split(' '):
                if val in setval:
                    n1 = 'idx__{}'.format(rid)
                    n2 = 'tt__' + str(val)
                    w1 = w2 = 1
                    edgerow = '{},{},{},{}\n'.format(n1, n2, w1, w2)
                    fo.write(edgerow)

        print(idx)