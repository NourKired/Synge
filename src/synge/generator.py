from __future__ import division
import numpy as np
from tqdm import tqdm
import logging
from synge.overlap_functions import apply_overlap
from synge.shape import apply_cdt, verify_shape

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(funcName)s - %(message)s ",
    level=logging.INFO,
)


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
    print("ok ok j'entre")
    print("je verifie avant dentrer",clus_cfg.parametres_shapes, clus_cfg._shapes)
    logging.info("generate samples for each cluster")
    for label, (param, shape) in tqdm(enumerate(zip(clus_cfg.parametres_shapes, clus_cfg._shapes))):
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
    print("data: ", data)
    return data, labels
