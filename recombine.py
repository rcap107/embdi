f1 = open('pipeline/embeddings/concat-amazon_google-gt_on_id.emb', 'w')
df1 = open('pipeline/embeddings/split/gt/vectors-amazon_google-rotation_split1.txt', 'r')
df2 = open('pipeline/embeddings/split/gt/vectors-amazon_google-rotation_split2.txt', 'r')
n_lines = 1363

lines = []
for idx, line in enumerate(df1):
    if line.startswith('idx__'):
        lines.append(line)

for idx, line in enumerate(df2):
    if line.startswith('idx__'):
        val, vector = line.split(' ', maxsplit=1)
        pre, idd = val.split('__')
        idd = int(idd) + n_lines
        val = pre + '__' + str(idd)
        l = ' '.join([val, vector])
        lines.append(l)

f1.write(str(len(lines)) + ' ' + str(300) + '\n')
for line in lines:
    f1.write(line)

f1.close()
df1.close()
df2.close()
