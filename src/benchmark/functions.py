from __future__ import division
import numpy as np
from six import string_types
import scipy
from generator import Shape, hyperRectangle, hyperSphere


def get_rotation_matrix(n_feats):
    rot_mat = 2 * (np.random.rand(n_feats, n_feats) - 0.5)
    ort = scipy.linalg.orth(rot_mat)
    if (
        ort.shape == rot_mat.shape
    ):  # check if `rot_mat` is full rank, so that `ort` keeps the same shape
        return ort
    else:
        return get_rotation_matrix(n_feats)


# CELLULE IMPORTANTE POUR LES PLOTS
def combinliste(seq, k):
    """
    Doc: cette fonction renvoie en sortie les differents combinaison de k elem parmis seq
    utile pour generer les diffentes combinaision des dimensions pour les plots
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
    Doc : cette fonction renvoie les une liste, ou chaque elem correspond à la matrice de données des clusters
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
