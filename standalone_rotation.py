from EmbDI.utils import  *
import random

def generate_syn_file(src_emb, tgt_emb):
    common_values = set()

    with open(src_emb, 'r') as fp1:
        with open(tgt_emb, 'r') as fp2:
            for idx, line in enumerate(fp1):
                if idx > 0:
                    val, vector = line.split(' ', maxsplit=1)
                    common_values.add(val)
            for idx, line in enumerate(fp2):
                if idx > 0:
                    val, vector = line.split(' ', maxsplit=1)
                    common_values.add(val)
    return list(common_values)

def write_syn_file(common_values, dset_name, frac=0.7):
    random.shuffle(common_values)
    train_idx = int(len(common_values) * frac)
    train = common_values[:train_idx]
    test = common_values[train_idx:]

    with open('pipeline/replacements/' + dset_name + '-train.txt', 'w') as fp:
        for v in train:
            s = '{} {}\n'.format(v,v)
            fp.write(s)

    with open('pipeline/replacements/' + dset_name + '-test.txt', 'w') as fp:
        for v in test:
            s = '{} {}\n'.format(v,v)
            fp.write(s)



if __name__ == '__main__':
    dset_name = 'itunes_amazon'
    # src_emb = 'pipeline/embeddings/amazon-alone-edgelist.emb'.format(dset_name)
    # tgt_emb = 'pipeline/embeddings/google-alone-edgelist.emb'.format(dset_name)
    src_emb = 'pipeline/embeddings/split/{}-rotation_split1.emb'.format(dset_name)
    tgt_emb = 'pipeline/embeddings/split/{}-rotation_split2.emb'.format(dset_name)

    common_values = generate_syn_file(src_emb, tgt_emb)
    write_syn_file(common_values, dset_name)

    train = 'pipeline/replacements/{}-fakem.txt'.format(dset_name)
    # train = 'pipeline/replacements/{}-train.txt'.format(dset_name)
    test = 'pipeline/replacements/{}-test.txt'.format(dset_name)
    train = 'pipeline/replacements/{}-g.txt'.format(dset_name)
    ground_truth = 'pipeline/replacements/{}-gt.txt'.format(dset_name)
    apply_rotation(src_emb, tgt_emb, 300, train, ground_truth)
