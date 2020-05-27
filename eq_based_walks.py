import pandas as pd
import random

dfname = 'imdb_movielens'

f = 'pipeline/datasets/{}/{}-master.csv'.format(dfname, dfname)
df = pd.read_csv(f)

min_len = 1
sentences = []
n_perm = 3


max_len = 5
for col1 in df.columns:
    group = df.groupby(col1)
    for k, g in group:
        len_g = len(g)
        if len_g >= min_len:
    #         print(k, len_g)
            for col2 in g.columns:
                if col1 == col2:
                    break
                s = [k]
                s += g[col2].dropna().tolist()
                s = list(set(s))
                if len(s) < min_len + 1:
                    continue
                else:
                    if len(s) > max_len:
                        for p in range(min([(n_perm), len(s)//n_perm])):
                            random.shuffle(s)
                            sus = []
                            for sp in range(1, len(s), max_len):
                                _ = [s[0]] + s[sp:sp+max_len]
                                random.shuffle(_)
                                sus.append(_)
                            sentences += sus
                    else:
                        sentences.append(s)
                        for p in range(n_perm*len(s)):
                            sentences.append(random.sample(s, k=len(s)))
print(len(sentences))

with open('{}-newwalks.walks'.format(dfname), 'w') as fp:
    for line in sentences:
        s = ' '.join([str(_) for _ in line]) + '\n'
        fp.write(s)

