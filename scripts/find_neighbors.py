import gensim.models as models
import argparse
import os.path as osp


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('model_file')
    parser.add_argument('-k', type=int, default=5)
    args = parser.parse_args()

    return args

def pprint(neighbors):
    for n in neighbors:
        node, dist = n
        print(f'{node:<60} -- {dist:.3f}')


if __name__ == '__main__':
    args = parse_args()

    model = models.KeyedVectors.load_word2vec_format(osp.join('..',args.model_file))
    node = input('Enter input node or "$Q" to quit: ')

    while node != '$Q':
        try:
            most_similar = model.most_similar(node, topn=args.k)
            pprint(most_similar)
        except KeyError:
            print(f'Word {node} not in vocabulary.')
        node = input('Enter input node or "$Q" to quit: ')
