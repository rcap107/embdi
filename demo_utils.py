import numpy as np
import networkx as nx
import plotly.graph_objects as go
from sklearn.decomposition import PCA


def apply_PCA(embeddings_file, reduced_file, n_components):
    keys = []

    with open(embeddings_file, 'r') as fp:
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

    with open(reduced_file, 'w') as fp:
        fp.write('{} {}\n'.format(*mat_fit.shape))
        for n, key in enumerate(keys):
            fp.write('{} '.format(key) + ' '.join([str(_) for _ in mat_fit[n, :]]) + '\n')

    print('Written on file {}.'.format(reduced_file))


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


def plot_graph(graph, left, middle, right, image_path='', width=800, height=600):
    # Preparing the traces to plot in the picture.

    # left = ['idx_1', 'idx_12', 'idx_22247']
    # right = list(df.columns)
    # middle = r1 + r2 + r3
    ll = tripartite(left,right,middle)

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

    edge_trace = go.Scatter(
        x=edge_x, y=edge_y,
        line=dict(width=0.5, color='#888'),
        hoverinfo='none',
        mode='lines')

    node_x = []
    node_y = []
    for node in graph.nodes():
        x, y = ll[node]
        node_x.append(x)
        node_y.append(y)

    node_trace = go.Scatter(
        x=node_x, y=node_y,
        mode='markers',
        hoverinfo='text',
        #     text=node_text,
        marker=dict(
            showscale=True,
            colorscale='Blackbody',
            reversescale=True,
            color=[],
            size=10,
            colorbar=dict(
                thickness=15,
                title='Node Type',
                xanchor='left',
                titleside='right',
                tickvals=[-1, 1, 0],
                ticktext=['RID', 'CID', 'Node']
            ),
            line_width=2))

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

    fig = go.Figure(data=[edge_trace, node_trace],
                    layout=go.Layout(
                        title='<br>Network graph on a snippet of imdb_movielens',
                        titlefont_size=16,
                        showlegend=False,
                        hovermode='closest',
                        margin=dict(b=20, l=5, r=5, t=15),
                        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                        width=width,
                        height=width
                    )
                    )
    fig.show()
    if image_path:
        # fig.write_html('../images/{}'.format(image_path))
        fig.write_image('../images/{}'.format(image_path))




def prepare_emb_matrix(embeddings_file):
    # Reading the reduced file
    keys = []
    with open(embeddings_file, 'r') as fp:
        lines = fp.readlines()

        sizes = lines[0].split()
        sizes = [int(_) for _ in sizes]

        mat = np.zeros(shape=sizes)
        for n, line in enumerate(lines[1:]):
            ll = line.strip().split()
            mat[n, :] = np.array(ll[1:])
            keys.append(ll[0])
    return mat, keys

def produce_heatmap(keys, mat, values, path=''):
    k = []
    for _ in values:
        k.append(keys.index(_))

    hmap_trace = go.Heatmap(z=mat[k, :], x=np.arange(50), y=values, colorscale='RdBu', xgap=2, ygap=2)
    fig = go.Figure(data=[hmap_trace], layout=dict(width=1200, height=800))
    fig.show()
    if path:
        fig.write_html(path)
