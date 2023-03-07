from __future__ import division
import math
import logging
import numpy as np
import random
import six
from synge.validators import (
    _validate_mv,
    _initialize_secondary_shape,
    _validate_scale,
    _initialize_prinicipal_shape,
    _validate_corr,
    _validate_alpha_n,
    _initialize_distribution,
    _validate_parametres_distributions,
    _validate_n_noise,
    _validate_rotate,
)


logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(funcName)s - %(message)s ",
    level=logging.INFO,
)


def _initialize_parameters(self):
    """
    Method to initialize the parameters of the object.
    """

    ## seed ##
    self.seed = int(self.seed) if self.seed is not None else random.random()

    ## n_cluster ##
    ##  already initialized

    ## n_samples ##
    ##  already initialized

    ## n_feats ##
    ##  already initialized

    ## clusters_label ##
    ##  already initialized

    ## centroids  ##
    ## the initialization will be managed in the following processes

    ## par_shapes #
    for i_cluster in range(self.n_cluster):
        if i_cluster not in self.par_shapes.keys():
            self.par_shapes[i_cluster] = (None, 1 / self.n_cluster)
    self.par_shapes = _initialize_prinicipal_shape(self.par_shapes)
    ## weight_cluster ##
    ##  already initialized

    ## distributions ##
    for i_cluster in range(self.n_cluster):
        if i_cluster not in self.distributions.keys():
            self.distributions[i_cluster] = random.choice(self.possible_distributions)

    if type(self.distributions) == str:
        dist = self.distributions
        self.distributions = {i: dist for i in range(self.n_cluster)}



    self._distributions = _initialize_distribution(self.distributions)
    ## parametres_distributions ##
    for i_cluster in range(self.n_cluster):
        if i_cluster not in self.parametres_distributions.keys():
            self.parametres_distributions[i_cluster] = (0, self.par_shapes[i_cluster][1])

    ## chv ##
    ## no need to initialize

    ## shapes ##
    for i_cluster in range(self.n_cluster):
        if i_cluster not in self.shapes.keys():
            self.shapes[i_cluster] = None

    if type(self.shapes) == str:
        shp = self.shapes
        self.shapes = {i: shp for i in range(self.n_cluster)}

    print("self.shapes",list(self.shapes.values()))
    self._shapes = _initialize_secondary_shape(self.shapes.values())

    ## prt_shapes ##
    for i_cluster in range(self.n_cluster):
        if i_cluster not in self.parametres_shapes.keys():
            self.parametres_shapes[i_cluster] = None

    if type(self.parametres_shapes) == str:
        shp = self.parametres_shapes
        self.parametres_shapes = {i: shp for i in range(self.n_cluster)}

    print("self.parametres_shapes",list(self.parametres_shapes.values()))

    ## scale  ##
    ##  already initialized

    ## rotate ##
    ##  already initialized

    ## visualize ##
    ##  already initialized

    # check validity of self.mv, and turn it into a list with self.n_cluster elements
    if hasattr(self.mv, "__iter__"):
        if len(self.mv) != self.n_cluster:
            raise ValueError(
                'There must be exactly one "mv" parameter for each cluster!'
            )
    else:
        if self.mv is None:
            self.mv = [random.choice([True, False]) for _ in range(self.n_cluster)]
        else:
            self.mv = [self.mv] * self.n_cluster
    assert all(_validate_mv(elem) for elem in self.mv)

    # check validity of self.scale, and turn it into a list with self.n_cluster elements
    if hasattr(self.scale, "__iter__"):
        if len(self.scale) != self.n_cluster:
            raise ValueError(
                'There must be exactly one "scale" parameter for each cluster!'
            )
    else:
        self.scale = [self.scale] * self.n_cluster
    assert all(_validate_scale(elem) for elem in self.scale)

    # check validity of self.corr, and turn it into a list with self.n_cluster elements
    if hasattr(self.corr, "__iter__"):
        if len(self.corr) != self.n_cluster:
            raise ValueError(
                'There must be exactly one correlation "corr" value for each cluster!'
            )
    else:
        self.corr = [self.corr] * self.n_cluster
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
        [math.floor(1 + self.n_cluster / math.log(self.n_cluster))] * self.n_feats
        if self.n_cluster > 1
        else [1 + 2 * (self.outliers > 1)] * self.n_feats
    )
    self._cmax = [
        round(-a) if a < 0 else round(c * a) for a, c in zip(self.alpha_n, self._cmax)
    ]
    self._cmax = np.array(self._cmax)


    # check validity of self.rotate, and turn it into a list with self.n_cluster elements
    if hasattr(self.rotate, "__iter__"):
        if len(self.rotate) != self.n_cluster:
            raise ValueError(
                "There must be exactly one rotate value for each cluster!"
            )
    else:
        self.rotate = [self.rotate] * self.n_cluster
    assert all(_validate_rotate(elem) for elem in self.rotate)
    # check validity of self.add_noise and self.n_noise
    if not isinstance(self.add_noise, six.integer_types):
        raise ValueError('Invalid input for "add_noise"! Input must be integer.')
    if isinstance(self.n_noise, list):
        if len(self.n_noise) == 0:
            self.n_noise = [[]] * self.n_cluster
        if hasattr(self.n_noise[0], "__iter__"):
            if len(self.n_noise) != self.n_cluster:
                raise ValueError(
                    'Invalid input for "n_noise"! List length must be the number of clusters.'
                )
        else:
            self.n_noise = [self.n_noise] * self.n_cluster
    else:
        raise ValueError('Invalid input for "n_noise"! Input must be a list.')
    assert all(_validate_n_noise(elem, self.n_feats) for elem in self.n_noise)
    return self


def _validate_paramaters_(self):
    ## seed ##
    ### already verified

    ## n_cluster ##
    ### already verified

    ## n_samples ##

    ## n_feats ##
    ##  already verified

    ## clusters_label ##
    if type(self.clusters_label) != list or len(self.clusters_label) != self.n_cluster:
        raise ValueError(
            "clusters_label must be  a list of length of the number of clusters."
        )

    ## centroids  ##
    if self.centroids_ != {} and self.centroids_ != [] and self.centroids_ is not None:
        for key, item in self.centroids_.items():
            if key > self.n_cluster:
                raise ValueError("centroids keys must be in range of n_cluster")
            if len(item) != self.n_feats:
                raise ValueError(
                    "centroids values must be a list (center coordinates) of length n_feats"
                )

    ## par_shapes ##
    if self.par_shapes != {}:
        for key, item in self.par_shapes.items():
            if key > self.n_cluster:
                raise ValueError("par_shapes keys must be in range of n_cluster")
            if len(item) != 2:
                raise ValueError(
                    "par_shapes values must be a tuple (main shape, radius|length)"
                )

    ## weight_cluster ##
    ##  already verified

    ## distributions ##
    for i in range(len(self.distributions)):
        if (
            type(self.distributions[i]) == list
            and len(self.distributions[i]) != self.n_feats
        ):
            raise ValueError(
                "Invalid distributions input! Input must have dimensions (n_cluster, n_feats)."
            )

    ## parametres_distributions ##
    if self.parametres_distributions != {}:
        for key, item in self.parametres_distributions.items():
            if key > self.n_cluster:
                raise ValueError(
                    "parametres_distributions keys must be in range of n_cluster"
                )
            if len(item) != 2:
                raise ValueError(
                    "parametres_distributions values must be a tuple (distribution center, length distibution)"
                )

    ## chv ##
    if self.chevauchement is not None:
        for i1, i2, perc in self.chevauchement:
            assert "overlap parameter must be (index st cluster, index 2nd cluster, percent of overlap)", (i1 < self.n_cluster and i2 < self.n_cluster and perc <= 1 and perc >= 0)

    ## shapes ##
    if self.shapes != {}:
        for key, item in self.shapes.items():
            if key > self.n_cluster:
                raise ValueError("shapes keys must be in range of n_cluster")

    ## prt_shapes ##
    if self.parametres_shapes != {}:
        for (key_par, item_par), (key_shp, item_shp) in zip(
            self.parametres_shapes.items(), self.shapes.items()
        ):
            if key_par > self.n_cluster:
                raise ValueError("par_shapes keys must be in range of n_cluster")
        if item_par is not None:
            if len(item_par) != len(item_shp):
                raise ValueError("they must have the same length")
            if len(item_par[0]) != self.n_feats:
                raise ValueError(
                    "the coordinate of the secondary centroids must correspond to number of dimensions clusters"
                )
            if len(item_par[2]) not in ["-", "+"]:
                raise ValueError(
                    "the sign must be i (-,+) so we can remove or add the shape into the cluster"
                )

    ## scale  ##
    ##  already verified

    ## rotate ##
    ##  already verified

    ## visualize ##
    ##  already verified

    return self
