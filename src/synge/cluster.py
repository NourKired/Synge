from __future__ import division
import numpy as np
import six
from synge.validate_param import _validate_paramaters_, _initialize_parameters
from synge.validators import (
    _validate_mv,
    _validate_scale,
    _validate_corr,
    _validate_parametres_distributions,
    _validate_n_noise,
    _validate_rotate,
    _validate_parametres_shapes,
)
from synge.generator import compute_batch
from synge.functions import get_rotation_matrix, read_data


class ClusterGenerator(object):
    """
    Structure to handle the input and create clusters according to it.
    """

    def __init__(
        self,
        seed,
        clusters_label,
        centroids,
        par_shapes,
        weight_cluster,
        distributions,
        parametres_distributions,
        scale,
        rotate,
        shapes,
        chevauchement,
        parametres_shapes,
        n_cluster=3,
        n_samples=1000,
        n_feats=3,
        **kwargs
    ):
        self.seed = seed
        self.n_cluster = n_cluster
        self.n_samples = n_samples
        self.n_feats = n_feats
        self.min_samples = 0
        self.possible_distributions = [
            "uniform",
            "gaussian",
            # "logistic",
            # "triangular",
            # "gamma",
            # "gap",
            # "binaire",
        ]

        try:  # click entry
            self.clusters_label = (
                list(read_data(clusters_label))
                if clusters_label is not None
                else [i for i in range(n_cluster)]
            )
            self.centroids_ = (
                read_data(centroids) if centroids is not None else {}
            )
            self.par_shapes = read_data(par_shapes) if par_shapes is not None else {}
            self.weight_cluster = (
                read_data(weight_cluster)
                if weight_cluster is not None
                else [1 / n_cluster for _ in range(n_cluster)]
            )
            self.distributions = (
                read_data(distributions) if distributions is not None else {}
            )
            self.parametres_distributions = (
                read_data(parametres_distributions)
                if parametres_distributions is not None
                else {}
            )
            self.shapes = read_data(shapes) if shapes is not None else {}
            self.chevauchement = (
                read_data(chevauchement) if chevauchement is not None else chevauchement
            )
            self.parametres_shapes = (
                read_data(parametres_shapes) if parametres_shapes is not None else {}
            )
        except:
            self.clusters_label = (
                clusters_label
                if clusters_label is not None
                else [i for i in range(n_cluster)]
            )
            self.centroids_ = centroids if centroids is not None else {}
            self.par_shapes = par_shapes if par_shapes is not None else {}
            self.weight_cluster = (
                weight_cluster
                if weight_cluster is not None
                else [1 / n_cluster for _ in range(n_cluster)]
            )
            self.distributions = distributions if distributions is not None else {}
            self.parametres_distributions = (
                parametres_distributions if parametres_distributions is not None else {}
            )
            self.shapes = shapes if shapes is not None else {}
            self.chevauchement = (
                chevauchement if chevauchement is not None else chevauchement
            )
            self.parametres_shapes = (
                parametres_shapes if parametres_shapes is not None else {}
            )
        self.mv = True
        self.corr = 0
        self.alpha_n = 7
        self._cmax = None
        self.scale = scale
        self.outliers = 0
        self.rotate = rotate
        self.add_noise = 0
        self.n_noise = []
        self.ki_coeff = 0

        for key, val in kwargs.items():
            self.__dict__[key] = val

        self._distributions = None
        self._shapes = None
        _initialize_parameters(self)
        _validate_paramaters_(self)
        self.clusters = self.get_cluster_configs()
        self._centroids = None
        self._locis = None
        self._idx = None
        print(
            self.seed,
            self.n_samples,
            self.n_feats,
            self.n_cluster,
            self.clusters_label,
            "self.centroids_,",
            self.centroids_,
            self.par_shapes,
            self.weight_cluster,
            self.distributions,
            "self.distributions",
            self.parametres_distributions,
            self.scale,
            self.rotate,
            self.shapes,
            self.chevauchement,
            self.parametres_shapes,
        )

    def generate_data(self, batch_size=0):
        self._centroids, self._locis, self._idx = locate_centroids(self)
        batches = generate_clusters(self, batch_size)
        if (
            batch_size == 0
        ):  # if batch_size == 0, just return the data instead of the generator
            return next(batches)
        else:
            return batches

    def get_cluster_configs(self):
        return [Cluster(self, i) for i in range(self.n_cluster)]

        # check validity of self.parameter_shape ,self.parametres_distributions,self.distributions


class Cluster(object):
    """
    Contains the parameters of an individual cluster.
    """

    settables = [
        "distributions",
        "mv",
        "corr",
        "parametres_distributions",
        "scale",
        "rotate",
        "n_noise",
    ]

    """
    List of settable properties of Cluster. These are the parameters which can be set at a cluster level, and override
    the parameters of the cluster generator.
    """

    def __init__(self, cfg, idx, corr_matrix=None):
        """
        Args:
            cfg (ClusterGenerator): Configuration of the data.
            idx (int): Index of a cluster.
            corr_matrix (np.array): Valid correlation matrix to use in this cluster.
        """
        self.cfg = cfg
        self.idx = idx
        self.corr_matrix = corr_matrix

    def generate_data(self, samples):
        if hasattr(self.distributions, "__iter__"):
            out = np.zeros((samples, self.cfg.n_feats))
            for f in range(self.cfg.n_feats):
                o = self.distributions[f](
                    (samples, 1),
                    self.mv,
                    self.parametres_distributions[0],
                    self.parametres_distributions[1],
                )
                out[:, f] = o.reshape(-1)
            return out
        else:
            return self.distributions(
                (samples, self.cfg.n_feats),
                self.mv,
                self.parametres_distributions[0],
                self.parametres_distributions[1],
            )

    @property
    def n_feats(self):
        return self.cfg.n_feats

    @property
    def distributions(self):
        return self.cfg._distributions[self.idx]

    @property
    def shape(self):
        return self.cfg._shapes[self.idx]

    @distributions.setter
    def distributions(self, value):
        if isinstance(value, six.string_types):
            self.cfg._distributions[self.idx] = self.dist.get_dist_function(value)
        elif hasattr(value, "__iter__"):
            self.cfg._distributions[self.idx] = [
                self.dist.get_dist_function(d) for d in value
            ]
        else:
            self.cfg._distributions[self.idx] = self.dist.get_dist_function(value)

    @shape.setter
    def shape(self, value):
        if isinstance(value, six.string_types):
            self.cfg._shapes[self.idx] = self.dist.get_shape_function(value)
        elif hasattr(value, "__iter__"):
            self.cfg._shapes[self.idx] = [
                self.dist.get_shape_function(d) for d in value
            ]
        else:
            self.cfg._shapes[self.idx] = self.dist.get_shape_function(value)

    @property
    def mv(self):
        return self.cfg.mv[self.idx]

    @mv.setter
    def mv(self, value):
        assert _validate_mv(value)
        self.cfg.mv[self.idx] = value

    @property
    def corr(self):
        return self.cfg.corr[self.idx]

    @corr.setter
    def corr(self, value):
        assert _validate_corr(value)
        self.cfg.corr[self.idx] = value

    @property
    def parametres_distributions(self):
        return self.cfg.parametres_distributions[self.idx]

    @parametres_distributions.setter
    def parametres_distributions(self, value):
        assert _validate_parametres_distributions(value[1])
        self.cfg.parametres_distributions[self.idx] = value

    @property
    def parametres_shapes(self):
        return self.cfg.parametres_shape[self.idx]

    @parametres_shapes.setter
    def parametres_shapes(self, value):
        assert _validate_parametres_shapes(value[1])
        self.cfg.parametres_shapes[self.idx] = value

    @property
    def scale(self):
        return self.cfg.scale[self.idx]

    @scale.setter
    def scale(self, value):
        assert _validate_scale(value)
        self.cfg.scale[self.idx] = value

    @property
    def rotate(self):
        return self.cfg.rotate[self.idx]

    @rotate.setter
    def rotate(self, value):
        assert _validate_rotate(value)
        self.cfg.rotate[self.idx] = value

    @property
    def n_noise(self):
        return self.cfg.n_noise[self.idx]

    @n_noise.setter
    def n_noise(self, value):
        assert _validate_n_noise(value, self.cfg.n_feats)
        self.cfg.n_noise[self.idx] = value


class ScheduledClusterGenerator(ClusterGenerator):
    """
    This cluster generator takes a schedule and all the ClusterGenerator arguments, and activates only the specified
    clusters in the schedule, for each time step.
    A time step is defined as one get call to ``self.weigh_cluster``, which is done when generating each new batch.
    That is, one time step is one call to :func:`.generate.compute_batch`.
    """

    def __init__(self, schedule, *args, **kwargs):
        """
        Args:
            schedule (list): List in which each element contains the indexes of the clusters active in the respective
                time step.
            *args: args for :meth:`ClusterGenerator.__init__`.
            **kwargs: kwargs for :meth:`ClusterGenerator.__init__`.
        """
        super(ScheduledClusterGenerator, self).__init__(*args, **kwargs)
        self.cur_time = 0
        self.schedule = schedule


def generate_clusters(clus_cfg, batch_size=0):
    """
    Generate data.
    Args:
        clus_cfg (clusters.DataConfig): Configuration.
        batch_size (int): Number of samples for each batch.
    Yields:
        np.array: Generated samples.
        np.array: Labels for the samples.
    """
    # generate correlation and rotation matrices
    for cluster in clus_cfg.clusters:
        # generate random symmetric matrix with ones in the diagonal
        # uses the vine method described here
        # http://stats.stackexchange.com/questions/2746/how-to-efficiently-generate-random-positive-semidefinite-correlation-matrices
        # using the correlation input parameter to set a threshold on the values of the correlation matrix
        corr = np.eye(clus_cfg.n_feats)
        aux = np.zeros(corr.shape)

        beta_param = 4

        for k in range(clus_cfg.n_feats - 1):
            for i in range(k + 1, clus_cfg.n_feats):
                aux[k, i] = (
                    2 * cluster.corr * (np.random.beta(beta_param, beta_param) - 0.5)
                )
                p = aux[k, i]
                for ind in range(k - 1, -1, -1):
                    p = (
                        p * np.sqrt((1 - aux[ind, i] ** 2) * (1 - aux[ind, k] ** 2))
                        + aux[ind, i] * aux[ind, k]
                    )
                corr[k, i] = p
                corr[i, k] = p
        perm = np.random.permutation(clus_cfg.n_feats)
        corr = corr[perm, :][:, perm]
        cluster.corr_matrix = np.linalg.cholesky(corr)
        cluster.correlation_matrix = corr

        # rotation matrix
        if cluster.rotate:
            cluster.rotation_matrix = get_rotation_matrix(clus_cfg.n_feats)

    if batch_size == 0:
        batch_size = clus_cfg.n_samples
    for batch in range(((clus_cfg.n_samples - 1) // batch_size) + 1):
        n_samples = min(batch_size, clus_cfg.n_samples - batch * batch_size)
        data, labels = compute_batch(clus_cfg, n_samples)
        yield data, np.reshape(labels, (len(labels), 1))


def locate_centroids(clus_cfg):
    """
    Generate locations for the centroids of the clusters.
    Args:
        clus_cfg (clusters.DataConfig): Configuration.
    Returns:
        np.array: Matrix (n_clusters, n_feats) with positions of centroids.
    """
    centroids = np.zeros((clus_cfg.n_cluster, clus_cfg.n_feats))

    p = 1.0
    idx = 1
    for i, c in enumerate(clus_cfg._cmax):
        p *= c
        if p > 2 * clus_cfg.n_cluster + clus_cfg.outliers / clus_cfg.n_cluster:
            idx = i
            break
    idx += 1
    locis = np.arange(p)
    np.random.shuffle(locis)
    clin = locis[: clus_cfg.n_cluster]

    # voodoo magic for obtaining centroids
    res = clin
    parametr_dist =list(clus_cfg.parametres_distributions.values())
    parametr_dist=list(zip(*parametr_dist))[1]
    for j in range(idx):
        center = ((res % clus_cfg._cmax[j]) + 1) / (clus_cfg._cmax[j] + 1)
        noise = (np.random.rand(clus_cfg.n_cluster) - 0.5) * parametr_dist
        centroids[:, j] = center + noise
        res = np.floor(res / clus_cfg._cmax[j])
    for j in range(idx, clus_cfg.n_feats):
        center = np.floor(
            clus_cfg._cmax[j] * np.random.rand(clus_cfg.n_cluster) + 1
        ) / (clus_cfg._cmax[j] + 1)
        noise = (np.random.rand(clus_cfg.n_cluster) - 0.5) * parametr_dist

        centroids[:, j] = center + noise

    return centroids, locis, idx
