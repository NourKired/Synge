import pandas as pd

# Visualisation
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def combinliste(seq: list, k: int) -> list:
    """
    Doc: this function returns as output the different combination of k elem among seq
    useful to generate the different combination of dimensions for the plots
    """
    p = []
    i, imax = 0, 2 ** len(seq) - 1
    while i <= imax:
        s = []
        j, jmax = 0, len(seq) - 1
        while j <= jmax:
            if (i >> j) & 1 == 1:
                s.append(seq[j])
            j += 1
        if len(s) == k:
            p.append(s)
        i += 1
    return p


def visualize_(X, y):
    nb_feats = int(input("Number of features to plot "))
    nb_clusters = int(input("Number of clusters to plot "))
    listes = combinliste(np.arange(nb_feats), 3)
    Y = list(set(y.reshape(-1)))[
        :nb_clusters
    ]  ## à excuter qu'une seule fois ( pour une deuxieme exécution commenter cette ligne et la precedente)

    opacity: dict = {i: 1 for i in Y}
    j = 1
    data: list = []
    name: dict = {i: str("%d.%d" % (j, i)) for i, val in zip(Y, Y)}  # num des clusters

    data_plot: pd.DataFrame = pd.DataFrame(X)
    data_plot["prediction"] = y
    # data_plot["prediction"] = prediction.replace([-1,1], name)
    data: dict = {}
    # fig = make_subplots(rows=1, cols=1)
    for i1, i2, i3 in listes:
        dataa = []
        for i, val in zip(Y, Y):
            data_semi_plot: pd.DataFrame = data_plot[data_plot["prediction"] == val]
            dataa.append(
                go.Scatter3d(
                    x=data_semi_plot[i1],
                    y=data_semi_plot[i2],
                    z=data_semi_plot[i3],
                    name=name[val],
                    mode="markers",
                    marker=dict(size=6),
                    opacity=opacity[i],
                )
            )
        data[(i1, i2, i3)] = dataa

    fig = make_subplots(
        rows=len(listes),
        cols=1,
        specs=[[{"type": "scatter3d"}]] * len(listes),
        subplot_titles=[str("%d.%d.%d " % (i, j, k)) for (i, j, k) in listes],
    )

    count = 0
    for (i1, i2, i3), d in data.items():
        count += 1
        fig.add_traces(d, rows=[count] * len(d), cols=[1] * len(d))

    fig.update_layout(width=1000, height=4000, showlegend=True)

    return fig.show()
