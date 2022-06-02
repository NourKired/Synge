from __future__ import division
import math
import numpy as np
from tqdm import tqdm
import logging

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(funcName)s - %(message)s ",
    level=logging.INFO,
)

def verify_shape(d_iter, shape, param, cluster, samples, clus_cfg, label):
    """documentation:
    cette fonction est utilisé pour conserver que les points dans les formes complexes
    parameters:
    """
    d_tot = []
    if shape is None:
        return d_iter
    if isinstance(shape, list):
        for (i, shp_) in enumerate(shape):
            d = []
            if isinstance(shp_, list):
                for (j, s) in enumerate(shp_):
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


def _validate_shape_intradistance(shape):

    if not (hasattr(shape, "__iter__") and len(shape) == 2):
        raise ValueError('Error! "shape" must be a tuple with size 2!')
    return True


def apply_cdt(A, centr, R) -> list[list]:
    """documentation:
    cette fonction est utilisé pour definir la zone du cluster " maximale"( apres la generation des données) avant application des formes
    parameters:
    """
    D : list = []
    [D.append(i) for i in list(A) if np.all(np.abs(np.array(i) - centr) <= R)]
    return D


def _aux_rms(mat):
    return np.sqrt((mat**2.0).sum(1) / mat.shape[1]).reshape((mat.shape[0], 1))


def _intradistance_aux(shape):
    assert _validate_shape_intradistance(shape)
    out = np.random.rand(*shape) - 0.5
    out = math.sqrt(shape[1]) * out / _aux_rms(out)
    return out


def compute_batch(clus_cfg, n_samples):
    """
    Generates one batch of data.
    Args:
        clus_cfg (clusters.DataConfig): Configuration.
        n_samples (int): Number of samples in the batch.
    Returns:
        np.array: Generated sample.
    """
    s = np.array(clus_cfg.clusters_label)
    logging.info("get probabilities of each class \n start generating labels")
    labels = np.random.choice(s, n_samples, p=clus_cfg.weight_cluster)
    data = np.zeros((n_samples, clus_cfg.n_feats))
    logging.info("setUp centroids")
    for lab in clus_cfg.centroids_.keys():
        clus_cfg._centroids[lab] = clus_cfg.centroids_[lab]
    logging.info("setUP overlapping")
    if clus_cfg.chevauchement is not None:
        indexes_overLap: dict = {}
        list_label_overlap = set(
            list(zip(*clus_cfg.chevauchement))[0]
            + list(zip(*clus_cfg.chevauchement))[1]
        )
    logging.info("generate samples for each cluster")
    for (label, param, shape) in tqdm(
        zip(clus_cfg.clusters_label, clus_cfg.parametres_shapes, clus_cfg._shapes)
    ):
        d_tot = []
        cluster = clus_cfg.clusters[label]
        indexes = labels == label
        # save indexes of overlaps clusters
        if clus_cfg.chevauchement is not None and label in list_label_overlap:
            indexes_overLap[label] = indexes
        samples = sum(indexes)  # nbr of samples in this cluster
        while len(d_tot) < samples:
            points = cluster.generate_data(samples * 5)
            d_iter = apply_cdt(
                points, clus_cfg._centroids[label], clus_cfg.par_shapes[label][1]
            )
            if clus_cfg.par_shapes[label][0] is not None:
                d_iter = clus_cfg.par_shapes[label][0](
                    d_iter,
                    clus_cfg._centroids[label],
                    clus_cfg.par_shapes[label][1],
                    "+",
                )

                if clus_cfg.shapes[label] is not None:
                    d_tot += verify_shape(
                        d_iter, shape, param, cluster, samples, clus_cfg, label
                    )
                else:
                    d_tot += d_iter
            else:
                d_tot += d_iter

        data[indexes] = d_tot[:samples]
        logging.info("apply rotation if true")
        if cluster.rotate:
            data[indexes] = data[indexes].dot(cluster.rotation_matrix)
        logging.info("apply overlapping if true")
        if clus_cfg.chevauchement is not None:
            apply_overlap(clus_cfg, indexes_overLap, data, cluster, samples, label)

    return data, labels


def apply_overlap(clus_cfg, indexes_overLap, data, cluster, samples, label):
    for (i, (l1, l2, percent)) in enumerate(clus_cfg.chevauchement):
        if l1 in indexes_overLap.keys() and l2 in indexes_overLap.keys():
            if np.any(clus_cfg._centroids[l1] < clus_cfg._centroids[l2]):
                xd = clus_cfg._centroids[l1] + clus_cfg.par_shapes[l1][1]
                d_ = (
                    data[indexes_overLap[l2]]
                    + xd
                    + clus_cfg.par_shapes[l2][1]
                    - clus_cfg._centroids[l2]
                )
                if clus_cfg.par_shapes[l1][0] is not None:
                    d_ = clus_cfg.par_shapes[l1][0](
                        d_,
                        clus_cfg._centroids[l1],
                        clus_cfg.par_shapes[l1][1],
                        "+",
                    )
                    if clus_cfg.shapes[l1] is not None:
                        d_ = verify_shape(
                            d_,
                            clus_cfg._shapes[l1],
                            clus_cfg.parametres_shapes[l1],
                            cluster,
                            samples,
                            clus_cfg,
                            label,
                        )
                result = [
                    point
                    for point in d_
                    if np.any(np.array(point) < np.amax(data[indexes_overLap[l1]]))
                    and np.any(np.array(point) < np.amax(data[indexes_overLap[l1]]))
                ]
                P = len(result) / samples
                xdmin = xd
                p_opt = P
                nb_opt = len(result)
                if P > percent:
                    while (
                        np.all(
                            np.array(xd)
                            <= np.array(
                                clus_cfg._centroids[l1] + clus_cfg.par_shapes[l1][1]
                            )
                        )
                        and np.all(
                            np.array(xd)
                            >= np.array(
                                clus_cfg._centroids[l1] - clus_cfg.par_shapes[l1][1]
                            )
                        )
                        and round(percent, 2) != round(P, 2)
                    ):
                        xd = xd + clus_cfg.par_shapes[l1][1] * 0.01
                        d_ = (
                            data[indexes_overLap[l2]]
                            + xd
                            + clus_cfg.par_shapes[l2][1]
                            - clus_cfg._centroids[l2]
                        )
                        if clus_cfg.par_shapes[l1][0] is not None:
                            d_ = clus_cfg.par_shapes[l1][0](
                                d_,
                                clus_cfg._centroids[l1],
                                clus_cfg.par_shapes[l1][1],
                                "+",
                            )
                            if clus_cfg.shapes[l1] is not None:
                                d_ = verify_shape(
                                    d_,
                                    clus_cfg._shapes[l1],
                                    clus_cfg.parametres_shapes[l1],
                                    cluster,
                                    samples,
                                    clus_cfg,
                                    label,
                                )
                        result = [
                            point
                            for point in d_
                            if np.any(
                                np.array(point) < np.amax(data[indexes_overLap[l1]])
                            )
                            and np.any(
                                np.array(point) < np.amax(data[indexes_overLap[l1]])
                            )
                        ]
                        P = len(result) / samples
                        if np.abs(P - percent) < np.abs(p_opt - percent):
                            xdmin = xd
                            p_opt = P
                            nb_opt = len(result)

                    d_f = (
                        data[indexes_overLap[l2]]
                        + xdmin
                        + clus_cfg.par_shapes[l2][1]
                        - clus_cfg._centroids[l2]
                    )
                    clus_cfg.chevauchement = clus_cfg.chevauchement[1:]
                    loggings_overlap(l1, l2, nb_opt, p_opt)
                    if p_opt < percent:
                        logging.warning(
                            "\n Warning : the percentage of overlap achieved is less than the desired percentage"
                        )
                else:
                    while (
                        (percent > P)
                        and np.all(
                            np.array(xd)
                            <= np.array(
                                clus_cfg._centroids[l1] + clus_cfg.par_shapes[l1][1]
                            )
                        )
                        and np.all(
                            np.array(xd)
                            >= np.array(
                                clus_cfg._centroids[l1] - clus_cfg.par_shapes[l1][1]
                            )
                        )
                        and round(percent, 2) != round(P, 2)
                    ):
                        xd = xd - clus_cfg.par_shapes[l1][1] * 0.01
                        d_ = (
                            data[indexes_overLap[l2]]
                            + xd
                            + clus_cfg.par_shapes[l2][1]
                            - clus_cfg._centroids[l2]
                        )
                        if clus_cfg.par_shapes[l1][0] is not None:
                            d_ = clus_cfg.par_shapes[l1][0](
                                d_,
                                clus_cfg._centroids[l1],
                                clus_cfg.par_shapes[l1][1],
                                "+",
                            )
                            if clus_cfg.shapes[l1] is not None:
                                d_ = verify_shape(
                                    d_,
                                    clus_cfg._shapes[l1],
                                    clus_cfg.parametres_shapes[l1],
                                    cluster,
                                    samples,
                                    clus_cfg,
                                    label,
                                )
                        result = [
                            point
                            for point in d_
                            if np.any(
                                np.array(point) < np.amax(data[indexes_overLap[l1]])
                            )
                            and np.any(
                                np.array(point) < np.amax(data[indexes_overLap[l1]])
                            )
                        ]
                        P = len(result) / samples
                        if np.abs(P - percent) < np.abs(p_opt - percent):
                            xdmin = xd
                            p_opt = P
                            nb_opt = len(result)
                    d_f = (
                        data[indexes_overLap[l2]]
                        + xdmin
                        + clus_cfg.par_shapes[l2][1]
                        - clus_cfg._centroids[l2]
                    )
                    clus_cfg.chevauchement = clus_cfg.chevauchement[1:]
                    loggings_overlap(l1, l2, nb_opt, p_opt)
                    if p_opt < percent:
                        logging.warning(
                            "\n Warning : the percentage of overlap achieved is less than the desired percentage"
                        )

            else:
                xd = clus_cfg._centroids[l1] - clus_cfg.par_shapes[l1][1]
                d_ = (
                    data[indexes_overLap[l2]]
                    + xd
                    - clus_cfg.par_shapes[l2][1]
                    - clus_cfg._centroids[l2]
                )
                if clus_cfg.par_shapes[l1][0] is not None:
                    d_ = clus_cfg.par_shapes[l1][0](
                        d_,
                        clus_cfg._centroids[l1],
                        clus_cfg.par_shapes[l1][1],
                        "+",
                    )
                    if clus_cfg.shapes[l1] is not None:
                        d_ = verify_shape(
                            d_,
                            clus_cfg._shapes[l1],
                            clus_cfg.parametres_shapes[l1],
                            cluster,
                            samples,
                            clus_cfg,
                            label,
                        )
                result = [
                    point
                    for point in d_
                    if np.any(np.array(point) < np.amax(data[indexes_overLap[l1]]))
                    and np.any(np.array(point) < np.amax(data[indexes_overLap[l1]]))
                ]
                P = len(result) / samples
                xdmin = xd
                p_opt = P
                nb_opt = len(result)
                if P > percent:
                    while (
                        np.all(
                            np.array(xd)
                            <= np.array(
                                clus_cfg._centroids[l1] + clus_cfg.par_shapes[l1][1]
                            )
                        )
                        and np.all(
                            np.array(xd)
                            >= np.array(
                                clus_cfg._centroids[l1] - clus_cfg.par_shapes[l1][1]
                            )
                        )
                        and round(percent, 2) != round(P, 2)
                    ):
                        xd = xd - clus_cfg.par_shapes[l1][1] * 0.01
                        d_ = (
                            data[indexes_overLap[l2]]
                            + xd
                            - clus_cfg.par_shapes[l2][1]
                            - clus_cfg._centroids[l2]
                        )
                        if clus_cfg.par_shapes[l1][0] is not None:
                            d_ = clus_cfg.par_shapes[l1][0](
                                d_,
                                clus_cfg._centroids[l1],
                                clus_cfg.par_shapes[l1][1],
                                "+",
                            )
                            if clus_cfg.shapes[l1] is not None:
                                d_ = verify_shape(
                                    d_,
                                    clus_cfg._shapes[l1],
                                    clus_cfg.parametres_shapes[l1],
                                    cluster,
                                    samples,
                                    clus_cfg,
                                    label,
                                )
                        result = [
                            point
                            for point in d_
                            if np.any(
                                np.array(point) < np.amax(data[indexes_overLap[l1]])
                            )
                            and np.any(
                                np.array(point) < np.amax(data[indexes_overLap[l1]])
                            )
                        ]
                        P = len(result) / samples
                        if np.abs(P - percent) < np.abs(p_opt - percent):
                            xdmin = xd
                            p_opt = P
                            nb_opt = len(result)

                    d_f = (
                        data[indexes_overLap[l2]]
                        + xdmin
                        - clus_cfg.par_shapes[l2][1]
                        - clus_cfg._centroids[l2]
                    )
                    clus_cfg.chevauchement = clus_cfg.chevauchement[1:]
                    loggings_overlap(l1, l2, nb_opt, p_opt)
                    if p_opt < percent:
                        logging.warning(
                            "\n Warning : the percentage of overlap achieved is less than the desired percentage"
                        )

                else:
                    while (
                        np.all(
                            np.array(xd)
                            <= np.array(
                                clus_cfg._centroids[l1] + clus_cfg.par_shapes[l1][1]
                            )
                        )
                        and np.all(
                            np.array(xd)
                            >= np.array(
                                clus_cfg._centroids[l1] - clus_cfg.par_shapes[l1][1]
                            )
                        )
                        and round(percent, 2) != round(P, 2)
                    ):
                        xd = xd + clus_cfg.par_shapes[l1][1] * 0.01
                        d_ = (
                            data[indexes_overLap[l2]]
                            + xd
                            - clus_cfg.par_shapes[l2][1]
                            - clus_cfg._centroids[l2]
                        )
                        if clus_cfg.par_shapes[l1][0] is not None:
                            d_ = clus_cfg.par_shapes[l1][0](
                                d_,
                                clus_cfg._centroids[l1],
                                clus_cfg.par_shapes[l1][1],
                                "+",
                            )
                            if clus_cfg.shapes[l1] is not None:
                                d_ = verify_shape(
                                    d_,
                                    clus_cfg._shapes[l1],
                                    clus_cfg.parametres_shapes[l1],
                                    cluster,
                                    samples,
                                    clus_cfg,
                                    label,
                                )
                        result = [
                            point
                            for point in d_
                            if np.any(
                                np.array(point) < np.amax(data[indexes_overLap[l1]])
                            )
                            and np.any(
                                np.array(point) < np.amax(data[indexes_overLap[l1]])
                            )
                        ]
                        P = len(result) / samples
                        if np.abs(P - percent) < np.abs(p_opt - percent):
                            nb_opt = len(result)
                            xdmin = xd
                            p_opt = P

                    d_f = (
                        data[indexes_overLap[l2]]
                        + xdmin
                        - clus_cfg.par_shapes[l2][1]
                        - clus_cfg._centroids[l2]
                    )
                    clus_cfg.chevauchement = clus_cfg.chevauchement[1:]
                    data[indexes_overLap[l2]] = d_f
                    loggings_overlap(l1, l2, nb_opt, p_opt)
                    if p_opt < percent:
                        logging.warning(
                            "\n Warning : the percentage of overlap achieved is less than the desired percentage"
                        )


def loggings_overlap(l1, l2, nb_opt, p_opt):
    logging.info(f"--------------- Overlapp between {l1} & {l2} ------")
    logging.info(
        f"Optimal percent = {p_opt} corresponding to {nb_opt} points on the overlap"
    )


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
