from __future__ import division
import math
import numpy as np
import random
import six
from validators import (
    _validate_mv,
    check_input_,
    _validate_scale,
    check_input_P,
    _validate_corr,
    _validate_alpha_n,
    check_input,
    _validate_parametres_distributions,
    _validate_n_noise,
    _validate_rotate,
    _validate_parametres_shapes,
)
from generator import compute_batch
from functions import get_rotation_matrix


class ClusterGenerator(object):
    """
    Structure to handle the input and create clusters according to it.
    """

    def __init__(
        self,
        seed=1,
        n_samples=2000,
        n_feats=3,
        k=5,
        clusters_label=None,
        centroids=None,
        par_shapes=None,
        weight_cluster=None,
        distributions=None,
        parametres_distributions=None,
        scale=True,
        rotate=True,
        shapes=None,
        chevauchement=None,
        parametres_shapes=None,
        **kwargs
    ):

        self.seed = seed
        self.n_samples = n_samples if n_samples is not None else 800
        self.n_feats = n_feats if n_feats is not None else 3
        self.k = k if k is not None else 1
        self.clusters_label = (
            clusters_label if clusters_label is not None else [i for i in range(k)]
        )
        self.centroids_ = centroids if centroids is not None else {}
        self.par_shapes = par_shapes if par_shapes is not None else {}
        self.n_clusters = len(k) if isinstance(k, list) else k
        self.weight_cluster = (
            weight_cluster if weight_cluster is not None else [1 / k for _ in range(k)]
        )
        self.min_samples = 0
        self.possible_distributions = ["gaussian", "uniform"]
        self.distributions = distributions
        self.parametres_distributions = (
            parametres_distributions if parametres_distributions is not None else {}
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
        self.shapes = shapes
        self.chevauchement = chevauchement
        self.parametres_shapes = (
            parametres_shapes if parametres_shapes is not None else {}
        )
        random.seed(self.seed)

        random.seed(self.seed)
        for key, val in kwargs.items():
            self.__dict__[key] = val

        self._distributions = None
        self._shapes = None
        self._validate_parameters()
        self.clusters = self.get_cluster_configs()

        self._centroids = None
        self._locis = None
        self._idx = None

    def generate_data(self, batch_size=0):
        np.random.seed(self.seed)
        self._centroids, self._locis, self._idx = locate_centroids(self)
        batches = generate_clusters(self, batch_size)
        if (
            batch_size == 0
        ):  # if batch_size == 0, just return the data instead of the generator
            return next(batches)
        else:
            return batches

    def get_cluster_configs(self):
        return [Cluster(self, i) for i in range(self.n_clusters)]

    def _validate_parameters(self):
        """
        Method to validate the parameters of the object.
        """
        if hasattr(self.k, "__iter__"):
            if len(self.k) == 1:  # if only one input, no point in being a list
                self.k = self.k[0]
                self.n_clusters = self.k
            elif len(self.k) < 1:
                raise ValueError('"k" parameter must have at least one value!')
            else:
                if sum(self.k) != self.n_samples:
                    raise ValueError(
                        "Total number of points must be the same as the sum of points in each cluster!"
                    )
        for i in range(self.k):
            if (
                self.parametres_distributions is None
                or i not in self.parametres_distributions.keys()
            ):

                self.parametres_distributions[i] = (0, 1 / self.k)
            if i not in self.par_shapes.keys():
                self.par_shapes[i] = (None, 1 / self.k)
            if i not in np.arange(len(self.parametres_shapes)):
                self.parametres_shapes[i] = None

        if isinstance(self.distributions, str):
            self.distributions = [self.distributions] * self.n_clusters
        else:
            if (
                isinstance(self.distributions, list)
                and len(self.distributions) != self.k
            ):
                length = len(self.distributions)
                self.distributions += ["uniform" for i in range(length, self.k)]

        for i, (par_shape, param_dist, dist) in enumerate(
            zip(
                self.par_shapes.values(),
                self.parametres_distributions.values(),
                self.distributions,
            )
        ):
            if dist is None:
                if param_dist is not None:
                    raise ValueError("shape parameter must be None when shape is None")
                self.distributions[i] = "uniform"

            if param_dist is not None and self.par_shapes[i][1] is not None:
                self.parametres_distributions[i] = (0, self.par_shapes[i][1])
            else:
                self.parametres_distributions[i] = (0, 1 / self.k)
            if self.par_shapes[i] is None:
                self.par_shapes[i] = (None, 1 / self.k)

            if self.par_shapes[i][0] is not None:
                if self.par_shapes[i][1] is None:
                    self.par_shapes[i][1] = 1 / self.n_clusters
            if self.parametres_distributions[i] is None:
                self.parametres_distributions[i] = (0, self.par_shapes[i][1])
            if self.parametres_distributions[i][0] is None:
                self.parametres_distributions[i][0] = 0
            if self.parametres_distributions[i][1] is None:
                self.parametres_distributions[i][1] = self.par_shapes[i][1]

        if self.distributions is not None:
            # check validity of self.distributions, and turning it into a (n_clusters, n_feats) matrix
            if hasattr(self.distributions, "__iter__") and not isinstance(
                self.distributions, str
            ):
                if len(self.distributions) != self.n_clusters:
                    raise ValueError(
                        "There must be exactly one distribution input for each cluster!"
                    )
                if isinstance(self.distributions[0], list):
                    if not all(
                        hasattr(elem, "__iter__") and len(elem) == self.n_feats
                        for elem in self.distributions
                    ):
                        raise ValueError(
                            "Invalid distributions input! Input must have dimensions (n_clusters, n_feats)."
                        )
            else:
                self.distributions = [self.distributions] * self.n_clusters
            self._distributions = check_input(self.distributions)
        else:
            self.distributions = [
                random.choice(self.possible_distributions)
                for _ in range(self.n_clusters)
            ]
            self._distributions = check_input(self.distributions)

        if self.shapes is not None:
            # check validity of self.distributions, and turning it into a (n_clusters, n_feats) matrix
            if isinstance(self.shapes, list) and not isinstance(self.shapes, str):
                self.shapes = [
                    self.shapes[i] if self.shapes[i] is not None else None
                    for i in range(len(self.shapes))
                ] + [None] * (self.n_clusters - len(self.shapes))
            else:
                self.shapes = [self.shapes] + [None] * (self.n_clusters - 1)
            self._shapes = check_input_(self.shapes)
        else:
            self.shapes = [None for _ in range(self.n_clusters)]
        self._shapes = check_input_(self.shapes)
        self.par_shapes = check_input_P(self.par_shapes)

        # check validity of self.mv, and turn it into a list with self.n_clusters elements
        if hasattr(self.mv, "__iter__"):
            if len(self.mv) != self.n_clusters:
                raise ValueError(
                    'There must be exactly one "mv" parameter for each cluster!'
                )
        else:
            if self.mv is None:
                self.mv = [random.choice([True, False]) for _ in range(self.n_clusters)]
            else:
                self.mv = [self.mv] * self.n_clusters
        assert all(_validate_mv(elem) for elem in self.mv)

        # check validity of self.scale, and turn it into a list with self.n_clusters elements
        if hasattr(self.scale, "__iter__"):
            if len(self.scale) != self.n_clusters:
                raise ValueError(
                    'There must be exactly one "scale" parameter for each cluster!'
                )
        else:
            self.scale = [self.scale] * self.n_clusters
        assert all(_validate_scale(elem) for elem in self.scale)

        # check validity of self.corr, and turn it into a list with self.n_clusters elements
        if hasattr(self.corr, "__iter__"):
            if len(self.corr) != self.n_clusters:
                raise ValueError(
                    'There must be exactly one correlation "corr" value for each cluster!'
                )
        else:
            self.corr = [self.corr] * self.n_clusters
        assert all(_validate_corr(elem) for elem in self.corr)

        # check validity of self.alpha_n, and turn it into a list with self.n_feats elements
        if hasattr(self.alpha_n, "__iter__"):
            if len(self.alpha_n) != self.n_feats:
                raise ValueError(
                    'There must be exactly one hyperplane parameter "alpha_n" value for each dimension!'
                )
        else:
            self.alpha_n = [self.alpha_n] * self.n_feats
        assert all(_validate_alpha_n(elem) for elem in self.alpha_n)

        # set self._cmax
        self._cmax = (
            [math.floor(1 + self.n_clusters / math.log(self.n_clusters))] * self.n_feats
            if self.n_clusters > 1
            else [1 + 2 * (self.outliers > 1)] * self.n_feats
        )
        self._cmax = [
            round(-a) if a < 0 else round(c * a)
            for a, c in zip(self.alpha_n, self._cmax)
        ]
        self._cmax = np.array(self._cmax)

        # check validity of self.parametres_distributions, and turn it into a list with self.n_clusters tuples

        if hasattr(self.parametres_distributions, "__iter__"):
            self.parametres_distributions = [
                (0, 1)
                if self.parametres_distributions[i] is None
                else self.parametres_distributions[i]
                for i in range(len(self.parametres_distributions))
            ] + [(0, 1)] * (self.n_clusters - len(self.parametres_distributions))
        else:
            self.parametres_distributions = [self.parametres_distributions] + [
                (0, 1)
            ] * (self.n_clusters - 1)
        assert all(
            _validate_parametres_distributions(elem)
            for elem in self.parametres_distributions
        )

        # check validity of self.parametres_shapes, and turn it into a list with self.n_clusters tuples
        if hasattr(self.parametres_shapes, "__iter__"):
            self.parametres_shapes = [
                self.parametres_shapes[i]
                if self.parametres_shapes[i] is not None
                else None
                for i in range(len(self.parametres_shapes))
            ] + [None] * (self.n_clusters - len(self.parametres_shapes))
        else:
            self.parametres_shapes = [self.parametres_shapes] + [None] * (
                self.n_clusters - 1
            )
        # assert all(_validate_parametres_shapes(elem) for elem in self.parametres_shapes)

        # check validity sizes of self.parametres_shapes, and shapes
        # assert (_validate_parametres_and_shapes(self.parametres_shapes,self.shapes))

        # check validity of self.rotate, and turn it into a list with self.n_clusters elements
        if hasattr(self.rotate, "__iter__"):
            if len(self.rotate) != self.n_clusters:
                raise ValueError(
                    "There must be exactly one rotate value for each cluster!"
                )
        else:
            self.rotate = [self.rotate] * self.n_clusters
        assert all(_validate_rotate(elem) for elem in self.rotate)
        # check validity of self.add_noise and self.n_noise
        if not isinstance(self.add_noise, six.integer_types):
            raise ValueError('Invalid input for "add_noise"! Input must be integer.')
        if isinstance(self.n_noise, list):
            if len(self.n_noise) == 0:
                self.n_noise = [[]] * self.n_clusters
            if hasattr(self.n_noise[0], "__iter__"):
                if len(self.n_noise) != self.n_clusters:
                    raise ValueError(
                        'Invalid input for "n_noise"! List length must be the number of clusters.'
                    )
            else:
                self.n_noise = [self.n_noise] * self.n_clusters
        else:
            raise ValueError('Invalid input for "n_noise"! Input must be a list.')
        assert all(_validate_n_noise(elem, self.n_feats) for elem in self.n_noise)

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
    centroids = np.zeros((clus_cfg.n_clusters, clus_cfg.n_feats))

    p = 1.0
    idx = 1
    for i, c in enumerate(clus_cfg._cmax):
        p *= c
        if p > 2 * clus_cfg.n_clusters + clus_cfg.outliers / clus_cfg.n_clusters:
            idx = i
            break
    idx += 1
    locis = np.arange(p)
    np.random.shuffle(locis)
    clin = locis[: clus_cfg.n_clusters]

    # voodoo magic for obtaining centroids
    res = clin
    parametr_dist = list(zip(*clus_cfg.parametres_distributions))[1]
    for j in range(idx):
        center = ((res % clus_cfg._cmax[j]) + 1) / (clus_cfg._cmax[j] + 1)
        noise = (np.random.rand(clus_cfg.n_clusters) - 0.5) * parametr_dist

        centroids[:, j] = center + noise
        res = np.floor(res / clus_cfg._cmax[j])
    for j in range(idx, clus_cfg.n_feats):
        center = np.floor(
            clus_cfg._cmax[j] * np.random.rand(clus_cfg.n_clusters) + 1
        ) / (clus_cfg._cmax[j] + 1)
        noise = (np.random.rand(clus_cfg.n_clusters) - 0.5) * parametr_dist

        centroids[:, j] = center + noise

    return centroids, locis, idx
