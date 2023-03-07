from __future__ import division
import numpy as np
from six import string_types
import scipy
from synge.shape import Shape, hyperRectangle, hyperSphere
import json


def get_rotation_matrix(n_feats):
    rot_mat = 2 * (np.random.rand(n_feats, n_feats) - 0.5)
    ort = scipy.linalg.orth(rot_mat)
    if (
        ort.shape == rot_mat.shape
    ):  # check if `rot_mat` is full rank, so that `ort` keeps the same shape
        return ort
    else:
        return get_rotation_matrix(n_feats)


def read_data(val: str) -> dict:
    if val is not None:
        with open(r"" + str(val), "r", encoding="utf-8") as f:
            data: dict = json.load(f)
        return dict(
            zip([int(key) for key in data.keys()], [value for value in data.values()])
        )


# CELLULE IMPORTANTE POUR LES PLOTS
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


def matrice_cluster(X, y) -> list:
    """
    Doc : this function returns a list, where each element corresponds to the cluster data matrix
    """
    matrice_cluster = []
    y1 = y.reshape(-1)
    for i in set(y1):
        cluster_i = [X[j].tolist() for j in range(len(X)) if y[j] == i]
        matrice_cluster.append(cluster_i)
    return matrice_cluster


shape_list = {"hyper Rectangle": hyperRectangle, "hyper Sphere": hyperSphere}


def get_shape_function(s):
    """
    Transforms Shape name into respective function.
    Args:
        d (str or function): Input Shape str/function.
    Returns:
        function: Actual function to compute the intended Shape.
    """
    if isinstance(s, Shape):
        return s
    elif hasattr(s, "__call__"):
        return Shape(s)
    elif isinstance(s, string_types):
        try:
            return Shape(shape_list[s])
        except KeyError:
            raise ValueError(
                'Invalid Shape name "'
                + s
                + '". Available names are: '
                + ", ".join(shape_list.keys())
            )

    else:
        raise ValueError("Invalid Shape input!")
