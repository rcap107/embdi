import networkx as nx
import plotly.graph_objects as go
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import seaborn as sns


def apply_PCA(embeddings_file, reduced_file, n_components):
    keys = []

    with open(embeddings_file, "r") as fp:
        lines = fp.readlines()

        sizes = lines[0].split()
        sizes = [int(_) for _ in sizes]

        mat = np.zeros(shape=sizes)
        for n, line in enumerate(lines[1:]):
            ll = line.strip().split()
            mat[n, :] = np.array(ll[1:])
            keys.append(ll[0])

    pca = PCA(n_components=n_components)

    mat_fit = pca.fit_transform(mat)

    with open(reduced_file, "w") as fp:
        fp.write("{} {}\n".format(*mat_fit.shape))
        for n, key in enumerate(keys):
            fp.write("{} ".format(key) + " ".join([str(_) for _ in mat_fit[n, :]]) + "\n")

    print("Written on file {}.".format(reduced_file))


# Function used to prepare the node layout for the graph picture.
def tripartite(left, right, middle, aspect_ratio=4 / 3, scale=1):
    height = 1
    width = aspect_ratio * height
    nodes = list(left) + list(right) + list(middle)
    offset = (width / 2, height / 2)

    left_xs = np.repeat(0, len(left))
    right_xs = np.repeat(width, len(right))
    left_ys = np.linspace(0, height, len(left))
    right_ys = np.linspace(0, height, len(right))
    middle_xs = np.repeat(width / 2, len(middle))
    middle_ys = np.linspace(0, height, len(middle))

    top_pos = np.column_stack([left_xs, left_ys]) - offset
    bottom_pos = np.column_stack([right_xs, right_ys]) - offset
    middle_pos = np.column_stack([middle_xs, middle_ys]) - offset

    pos = np.concatenate([top_pos, bottom_pos, middle_pos])
    pos = nx.rescale_layout(pos, scale=scale)
    pos = dict(zip(nodes, pos))
    return pos


def plot_graph(graph, left, middle, right, image_path="", title="", width=800, height=600):
    # Preparing the traces to plot in the picture.

    # left = ['idx_1', 'idx_12', 'idx_22247']
    # right = list(df.columns)
    # middle = r1 + r2 + r3
    ll = tripartite(left, right, middle)

    edge_x = []
    edge_y = []

    for edge in graph.edges:
        x0, y0 = ll[edge[0]]
        x1, y1 = ll[edge[1]]
        edge_x.append(x0)
        edge_x.append(x1)
        edge_x.append(None)
        edge_y.append(y0)
        edge_y.append(y1)
        edge_y.append(None)

    edge_trace = go.Scatter(x=edge_x, y=edge_y, line=dict(width=0.5, color="#888"), hoverinfo="none", mode="lines")

    node_x = []
    node_y = []
    for node in graph.nodes():
        x, y = ll[node]
        node_x.append(x)
        node_y.append(y)

    node_trace = go.Scatter(
        x=node_x,
        y=node_y,
        mode="markers",
        hoverinfo="text",
        #     text=node_text,
        marker=dict(
            showscale=True,
            colorscale="Blackbody",
            reversescale=True,
            color=[],
            size=10,
            colorbar=dict(
                thickness=15,
                title="Node Type",
                xanchor="left",
                titleside="right",
                tickvals=[-1, 1, 0],
                ticktext=["RID", "CID", "Node"],
            ),
            line_width=2,
        ),
    )

    node_text = []
    ns = list(graph.nodes())

    node_colors = []
    for node, val in enumerate(ns):
        if val in left:
            node_colors.append(-1)
        elif val in right:
            node_colors.append(1)
        else:
            node_colors.append(0)
        node_text.append(ns[node])

    node_trace.marker.color = node_colors

    node_trace.text = node_text

    fig = go.Figure(
        data=[edge_trace, node_trace],
        layout=go.Layout(
            title=title,
            titlefont_size=16,
            showlegend=False,
            hovermode="closest",
            margin=dict(b=20, l=5, r=5, t=15),
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            width=width,
            height=height,
        ),
    )
    fig.show()
    if image_path:
        fig.write_html("{}".format(image_path))


def prepare_emb_matrix(embeddings_file):
    # Reading the reduced file
    keys = []
    with open(embeddings_file, "r") as fp:
        lines = fp.readlines()

        sizes = lines[0].split()
        sizes = [int(_) for _ in sizes]

        mat = np.zeros(shape=sizes)
        for n, line in enumerate(lines[1:]):
            ll = line.strip().split()
            mat[n, :] = np.array(ll[1:])
            keys.append(ll[0])
    return mat, keys


def produce_heatmap_plotly(keys, mat, values, path="", width=1200, height=800):
    k = []
    for _ in values:
        k.append(keys.index(_))

    hmap_trace = go.Heatmap(z=mat[k, :], x=np.arange(50), y=values, colorscale="RdBu", xgap=2, ygap=2)
    fig = go.Figure(data=[hmap_trace], layout=dict(width=width, height=height))
    fig.show()
    if path:
        # fig.write_html(path)
        fig.write_image(path)


def produce_heatmap(keys, values, mat, path="", rotation=0, labels=[], figsize=(6.5, 4)):
    k = []
    for _ in values:
        k.append(keys.index(_))
    if labels:
        refactored_values = labels
    else:
        refactored_values = values
    fig = plt.figure(figsize=figsize)
    g = sns.heatmap(data=mat[k, :], cmap="RdBu", cbar=False, linewidths=1)
    g.set_yticklabels(labels=refactored_values, rotation=rotation)
    if path:
        fig.savefig(path)


def print_most_similar(model, v_res, topn=20):
    r = model.similar_by_vector(v_res, topn=topn)

    r = [_ for _ in r if not _[0].startswith("idx_")]
    return r


def noidx(emb_file):
    with open(emb_file) as fi:
        count_lines = 0
        with open(emb_file[:-5] + "-noidx.embs", "w") as fo:
            for line in filter(lambda x: x[:5] != "idx__", fi):
                if len(line.split(" ")) == 2:
                    fo.write(line)
                    print(line)
                    _, dim = line.split(" ")
                else:
                    fo.write(line)
                    count_lines += 1
            fo.seek(0)
            fo.write("{} {}".format(count_lines, dim))


def tsne_plot_similar_words(title, labels, embedding_clusters, word_clusters, a, filename=None, idx=[]):
    plt.figure(figsize=(12, 12))
    colors = cm.rainbow(np.linspace(0, 1, len(labels)))
    for label, embeddings, words, color in zip(labels, embedding_clusters, word_clusters, colors):
        x = embeddings[:, 0]
        y = embeddings[:, 1]
        if label in idx:
            plt.scatter(x, y, c=[color], alpha=a, label=label.replace("_", " ").replace("|", " "), marker="+")
        else:
            plt.scatter(x, y, c=[color], alpha=a, label=label.replace("_", " ").replace("|", " "), marker="o")
        for i, word in enumerate(words):
            plt.annotate(
                word.replace("tt__", "").replace("_", " ").replace("|", " "),
                alpha=1,
                xy=(x[i], y[i]),
                xytext=(2, 2),
                textcoords="offset points",
                ha="center",
                va="bottom",
                size=20,
                annotation_clip=True,
            )
    plt.legend(loc="best", title="Neighbors of")
    plt.title(title)
    plt.tick_params(
        axis="both", which="both", bottom=False, top=False, labelbottom=False, left=False, right=False, labelleft=False
    )
    plt.grid(True)
    if filename:
        plt.savefig(filename, format="png", dpi=150, bbox_inches="tight")
    plt.show()


def produce_underdimensioned_clusters(embedding_clusters):
    embedding_clusters = np.array(embedding_clusters)
    n, m, k = embedding_clusters.shape
    tsne_model_en_2d = TSNE(perplexity=15, n_components=2, init="pca", n_iter=3500, random_state=32)
    embeddings_en_2d = np.array(tsne_model_en_2d.fit_transform(embedding_clusters.reshape(n * m, k))).reshape(n, m, 2)
    return embeddings_en_2d


def produce_clusters(model, keys):
    embedding_clusters = []
    word_clusters = []
    for word in keys:
        embeddings = []
        words = []
        for similar_word, _ in model.most_similar(word, topn=5):
            words.append(similar_word)
            embeddings.append(model[similar_word])
        embedding_clusters.append(embeddings)
        word_clusters.append(words)
    return embedding_clusters, word_clusters


def find_match(idx1, model, n_items, topn=100):
    def clean_candidates(target, candidates):
        tgt = int(target.split("_")[1])
        c = [_ for _ in candidates if _.startswith("idx_")]
        # print(c[:1])
        return c[:1]

    ms = model.most_similar(str(idx1), topn=topn)
    mm = [item[0] for item in ms]
    candidates = clean_candidates(idx1, mm)
    for c in candidates[:1]:
        ms = model.most_similar(c, topn=topn)
        mm = [item[0] for item in ms]
        cc = clean_candidates(c, mm)
        if cc[0] == idx1:
            return (idx1, c)


def pprint_sim(simlist, ntop=15):
    for tup in simlist[:ntop]:
        key, val = tup
        if key.startswith("tt__"):
            k = key[4:].replace("_", " ")
        else:
            k = key
        v = "{:.2f}".format(val)
        print("{:>70} {:>10}".format(k, v))
