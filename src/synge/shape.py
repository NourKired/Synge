import math
import numpy as np


def verify_shape(d_iter, shape, param, cluster, samples, clus_cfg, label):
    """documentation:
    cette fonction est utilisé pour conserver que les points dans les formes complexes
    parameters:
    """
    d_tot = []
    if shape is None:
        return d_iter
    if isinstance(shape, list):
        for i, shp_ in enumerate(shape):
            d = []
            if isinstance(shp_, list):
                for j, s in enumerate(shp_):
                    d += list(s(d_iter, param[i][j][0], param[i][j][1], param[i][j][2]))
            else:
                d += list(shp_(d_iter, param[i][0], param[i][1], param[i][2]))
            d = set(list(zip(*list(zip(*d)))))
            d_iter = set(list(zip(*list(zip(*d_iter)))))
            d_iter = list(d_iter.intersection(d))
        d_tot += d_iter
    else:
        points = cluster.generate_data(samples)
        if shape is not None:
            points = apply_cdt(
                points,
                clus_cfg.parametres_shapes[label][0],
                clus_cfg.parametres_shapes[label][1],
            )
            d_tot += shape(
                points,
                clus_cfg.parametres_shapes[label][0],
                clus_cfg.parametres_shapes[label][1],
                clus_cfg.parametres_shapes[label][2],
            )
        else:
            d_tot = points
    return d_tot


def apply_cdt(A, centr, R) -> list:
    """documentation:
    cette fonction est utilisé pour definir la zone du cluster " maximale"( apres la generation des données) avant application des formes
    parameters:
    """
    D: list = []
    [D.append(i) for i in list(A) if np.all(np.abs(np.array(i) - centr) <= R)]
    return D


def _validate_shape_intradistance(shape):
    if not (hasattr(shape, "__iter__") and len(shape) == 2):
        raise ValueError('Error! "shape" must be a tuple with size 2!')
    return True


def _aux_rms(mat):
    return np.sqrt((mat**2.0).sum(1) / mat.shape[1]).reshape((mat.shape[0], 1))


def _intradistance_aux(shape):
    assert _validate_shape_intradistance(shape)
    out = np.random.rand(*shape) - 0.5
    out = math.sqrt(shape[1]) * out / _aux_rms(out)
    return out


def hyperRectangle(A, centr, param, cdt, *args) -> list:
    """documentation:
    parameters:
    """
    D = []
    if cdt == "+":
        [
            D.append(i)
            for i in list(A)
            if np.all(np.abs(np.array(i) - centr) <= param / 2)
        ]
    else:
        [
            D.append(i)
            for i in list(A)
            if not np.all(np.abs(np.array(i) - centr) <= param / 2)
        ]
    return D


def hyperSphere(A, centr, param, cdt, *args) -> list:
    """documentation:
    parameters:
    liste des coordonnées des points generée
    """
    D = []
    if cdt == "+":
        [D.append(i) for i in A if (((np.array(i) - centr) ** 2).sum() <= (param) ** 2)]
    else:
        [
            D.append(i)
            for i in A
            if not (((np.array(i) - centr) ** 2).sum() <= (param) ** 2)
        ]
    return D


shape_list = {"hyper Rectangle": hyperRectangle, "hyper Sphere": hyperSphere}


class Shape(object):
    def __init__(self, s, *args):
        self.s = s
        self.args = args

    def __call__(self, A, centr, param, *args):
        return self.s(A, centr, param, *args)
