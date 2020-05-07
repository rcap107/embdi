import pandas as pd
import random
import numpy as np
import os
import warnings

random.seed(1234)
np.random.seed(1234)


def analogies(combinations, df, test_dir, choices_per_tuple):
    choices = choices_per_tuple
    limit = 50  # maximum number of attempts when looking for distinct values

    unique, values = np.unique([str(_) for _ in df.values.ravel().tolist()], return_counts=True)
    threshold = len(unique)*0.05

    stop_words = list(unique[values > threshold])

    # df = df.replace(stop_words, '*')

    for comb in combinations:
        combination_list = []

        d = df[comb]
        category = (str(': ' + '-'.join(list(comb))))
        combination_list.append(category)
        for line in d.itertuples(index=False):
            found = 0
            break_counter = 0
            while found < choices:
                ix = random.choice(d.index)
                current_line = d.loc[ix]
                ll = list(line) + list(current_line)
                if len(np.unique(ll)) == 4:
                    skip = False
                    for _ in ll:
                        if len(_) == 0 and _ != '*':
                            skip = True
                    if np.nan not in ll and not skip:
                        ll = ' '.join(ll)
                        combination_list.append(ll)
                        found += 1
                # if len(ll[0]) == 0:
                #     print(ll)
                #     quit()

                if break_counter == limit:
                    break
                break_counter += 1

        file_name = 'q_' + comb[0] + '-' + comb[1] + '.test'
        with open(test_dir + file_name, 'w', encoding='utf-8') as fp:
            for combination in combination_list:
                fp.write(combination+'\n')


def gen_no_match_row(df, test_dir, test_number, target_columns):
    print('Generating row tests on target columns:')
    print(', '.join(target_columns))

    for column in target_columns:
        sentence_list = []
        uniq_col = list(set(df[column].values.tolist()))
        if len(uniq_col) < 10:
            continue
        if np.nan in uniq_col:
            uniq_col.remove(np.nan)
        if '' in uniq_col:
            uniq_col.remove('')

        # uniq_col = [_ for _ in uniq_col if (_ != np.nan) and (_ != '')]
        test_count = 0
        full_count = 0
        while full_count < test_number:
            # print(test_count)
        # for test_count in range(test_number):
            if full_count > 5*test_number:
                break
            full_count += 1
            target_right = np.random.choice(range(len(df)), 1)
            correct = df[target_columns].iloc[target_right]
            if correct.isnull().values.any():
                # Null values in the target line, retrying.
                test_count -= 1
                continue
            mistake = random.choice(uniq_col)
            while mistake == correct[column].values:
                if mistake in ['', np.nan]: raise ValueError('Wrong has illegal value \'{}\''.format(mistake))
                # "wrong" was already found, retrying
                mistake = random.choice(uniq_col)
            edited_correct = correct.copy()
            edited_correct[column] = mistake

            sentence = edited_correct.values.tolist()[0]
            try:
                sentence.remove(mistake)
                # sentence.append(mistake)
                sentence.append(mistake)
                sentence_list.append(list(sentence))
                test_count += 1
            except ValueError:
                test_count -= 1

        file_name = 'nmr_' + column + '.test'
        with open(test_dir + file_name, 'w') as fp:
            for l in sentence_list:
                fp.write(str(l)+'\n')
    print('Row tests were generated.')


def gen_no_match_col(combinations, df, test_dir, n_sentences, sentence_len):
    print('Generating column tests on target combinations.')
    print(combinations)
    # for comb in combinations:
    #     print(', '.join(comb), end='')

    for comb in combinations:
        print(comb)
        test_list = []
        d = df[comb]
        uniq_col1 = (set(d[comb[0]].values.tolist()))
        uniq_col2 = (set(d[comb[1]].values.tolist()))

        if np.nan in uniq_col1:
            uniq_col1.remove(np.nan)
        if np.nan in uniq_col1:
            uniq_col1.remove('')

        if np.nan in uniq_col2:
            uniq_col2.remove(np.nan)
        if np.nan in uniq_col2:
            uniq_col2.remove('')

        common_values = uniq_col1.intersection(uniq_col2)
        # common_values = [_ for _ in uniq_col1 if _ in uniq_col2]
        # common_values = [_ for _ in uniq_col1 if _ in uniq_col2]
        for _ in common_values:
            uniq_col1.remove(_)
            uniq_col2.remove(_)

        if len(uniq_col1) < sentence_len:
            warnings.warn('Not enough unique values in column {} for combination {}'.format(comb[0], comb))
            continue
            # raise ValueError()
        if len(uniq_col2) < sentence_len:
            warnings.warn('Not enough unique values in column {}'.format(comb[0]))
            continue

        uniq_col1 = list(uniq_col1)
        uniq_col2 = list(uniq_col2)

        for _ in range(n_sentences):
            sentence = np.random.choice(uniq_col1, sentence_len-1).tolist()
            wrong_term = np.random.choice(uniq_col2, 1).tolist()[0]
            while wrong_term in sentence:
                wrong_term = list(np.random.choice(uniq_col2, 1))[0]
            sentence.append(wrong_term)
            test_list.append(sentence)
            # print(_)
        file_name = 'nmc_' + comb[0] + '-' + comb[1] + '.test'
        with open(test_dir + file_name, 'w', encoding='utf-8') as fp:
            for l in test_list:
                fp.write(str(l)+'\n')


def gen_no_match_concept(combinations, df: pd.DataFrame, test_dir, n_sentences, sentence_len):
    print('Generating row tests on target columns:')
    # print(', '.join(combinations))

    for comb in combinations:
        test_list = []
        d = df[comb]
        target_uniques = d[comb[1]].unique().tolist()
        if '' in target_uniques:
            target_uniques.remove('')
        if np.nan in target_uniques:
            target_uniques.remove(np.nan)

        g = d.groupby(comb[0])
        eligible = {}
        if '' in g:
            del g['']

        for key, group in g:
            unique = group[comb[1]].unique().tolist()
            if '' in unique:
                unique.remove('')
            if np.nan in unique:
                unique.remove(np.nan)
            if len(unique) >= sentence_len:
                eligible[key] = group[comb[1]].unique().tolist()
        if len(eligible) == 0:
            raise ValueError('Combination {} did not contain any suitable test.'.format(comb))

        eligible.pop('', None)
        eligible.pop(np.nan, None)

        run_counter = 0
        choice = []
        while run_counter < n_sentences:
            chosen_value = random.choice(list(eligible.keys()))
            if chosen_value == '':
                raise ValueError('{} was not removed'.format('\'\''))
            run_limiter = 10
            found = False
            while not found:
                found = True
                choice = np.random.choice(eligible[chosen_value], sentence_len-1, replace=False)
                choice = choice.tolist()
                if np.nan in choice or '' in choice:
                    found = False
                if run_limiter == 0:
                    break
                run_limiter -= 1
            if run_limiter == 0:
                run_counter -= 1
                continue

            # target_key = np.random.choice(list(eligible.keys()), 1)[0]
            # target = np.random.choice(eligible[target_key], 1)[0]
            found = False
            run_limiter = 10
            while not found and run_limiter > 0:
                target_key = np.random.choice(list(eligible.keys()), 1)[0]
                target = np.random.choice(eligible[target_key], 1)[0]
                found = True
                if target == np.nan:
                    raise ValueError('nan not removed')
                elif target in choice:
                    found = False
                    run_limiter -= 1

            if found:
                choice.append(target)
                sen = [chosen_value] + choice
                for idx, v in enumerate(sen):
                    try:
                        sen[idx] = str(int(v))
                    except ValueError:
                        sen[idx] = str(v)

                test_list.append(sen)

                run_counter += 1

        file_name = 'nmcon_' + comb[0] + '-' + comb[1] + '.test'
        with open(test_dir + file_name, 'w') as fp:
            for l in test_list:
                fp.write(str(l)+'\n')


if __name__ == '__main__':
    # Read dataset
    f = 'fodors_zagats'
    print(f)
    df = pd.read_csv('pipeline/datasets/{}/{}-master.csv'.format(f, f), engine='python')
    # Drop all rows that contain null values
    # df = df.dropna()
    #
    # If necessary, sample the dataset.
    df1 = df.sample(frac=0.5)
    print(len(df1))
    flag = ''
    # Create the directory where all tests will be written into.
    test_dir = 'pipeline/test_dir/' + f + '{}'.format(flag) + '/'
    os.makedirs(test_dir, exist_ok=True)

    # For the nmr tests, choose the target attributes
    # nmr_attrs_to_change = ['title', 'director', 'original_language', 'year', 'production_countries']

    # For the nmc tests, choose the combination of columns

    nmc_col_combinations = []

    for _1 in df.columns:
        for _2 in df.columns:
            if _1 != _2:
                nmc_col_combinations.append([_1, _2])


    # dblp-acm

    # cols='author_1,author_2,author_3,author_4,title,venue,year'.split(',')
    # fd = [
    #     ['venue', 'title'],
    #     ['venue', 'author_1'],
    #     ['year', 'title']
    # ]

    # dblp-scholar

    # cols='''title,authors,venue,year'''.split(',')
    #
    # fd = [
    #     ['venue', 'title'],
    #     ['venue', 'authors'],
    #     ['year', 'title']
    # ]
    # beer
    # cols='''beer_name,brew_factory_name,style,abv'''.split(',')
    #
    # fd = [
    #     ['brew_factory_name', 'beer_name'],
    #     ['style', 'beer_name'],
    #     ['abv', 'beer_name']
    # ]
    # fodors zagats
    cols='''addr,city,class,name,phone,type'''.split(',')
    fd = [
        ['city', 'addr'],
        ['type','addr'],
    ]

    # walmart_amazon
    # cols='''title,category,brand,modelno,price'''.split(',')
    #
    # fd = [
    #     ['category', 'title'],
    #     ['brand', 'title'],
    #     ['brand', 'modelno']
    # ]

    #   itunes amazon
    # cols='''album_name,artist_name,copyright,genre,price,released,song_name,time'''.split(',')
    #
    # fd = [
    #     ['artist_name', 'album_name'],
    #     ['genre', 'artist_name'],
    #     ['genre', 'album_name'],
    #     ['album_name', 'song_name'],
    #     ['artist_name', 'song_name']
    # ]

    # amazon google
    # cols='''manufacturer,price,title'''.split(',')
    #
    # fd = [
    #     ['manufacturer', 'title']
    # ]

    # imdb movielens

    # cols = '''actor_1,actor_2,actor_3,director,original_language,production_countries,title,year'''.split(',')
    # fd = [
    #     # ['original_language', 'director'],
    #     # ['director', 'actor_1'],
    #     ['director', 'title'],
    #     ['actor_1', 'title'],
    #     ['original_language', 'title']
    # ]


    # msd-pre
    # '''title,release,artist_name,duration,year'''
    # fd = [
    #     ['artist_name', 'title'],
    #     ['artist_name', 'release'],
    #     ['year', 'title']
    # ]

    match_concept = fd
    # Length of each test sentence
    test_length = 5

    # Number of sentences to generate for each test
    n_sentences = 1000

    gen_no_match_col(fd, df1, test_dir, n_sentences, sentence_len=test_length)
    gen_no_match_row(df1, test_dir, n_sentences, cols)
    gen_no_match_concept(match_concept, df1, test_dir, n_sentences, sentence_len=5)
