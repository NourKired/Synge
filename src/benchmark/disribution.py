import numpy as np
from six import string_types, integer_types
from generator import _intradistance_aux, _aux_rms


def _validate_shape(shape):
    if not (
        hasattr(shape, "__iter__") and (len(shape) == 2 or len(shape) == 1)
    ) and not isinstance(shape, integer_types):
        raise ValueError('Error! "shape" must be an integer or a tuple with size 2!')
    return True


def _validate_shape_intradistance(shape):

    if not (hasattr(shape, "__iter__") and len(shape) == 2):
        raise ValueError('Error! "shape" must be a tuple with size 2!')
    return True


distributions_list = {
    "uniform": lambda shape, param1, param2: np.random.uniform(
        param1 - param2, param1 + param2, shape
    ),
    "gaussian": lambda shape, param1, param2: np.random.normal(param1, param2, shape),
    # 'logistic': lambda shape, param: np.random.logistic(0, param, shape),
    # 'triangular': lambda shape, param: np.random.triangular(-param, 0, param, shape),
    # 'gamma': lambda shape, param: np.random.gamma(2 + 8 * np.random.rand(), param / 5, shape),
    # 'gap': lambda shape, param: gap(shape, param),
    # 'binaire': lambda shape,param : np.random.choice([0,1],shape)
}
"""List of distributions for which you can just provide a string as input."""

# Aliases for distributions should be put here.
distributions_list["normal"] = distributions_list["gaussian"]

valid_distributions = distributions_list.keys()
"""List of valid strings for distributions."""


class Distribution(object):
    def __init__(self, f, **kwargs):
        self.f = f
        self.kwargs = kwargs

    def __call__(self, shape, intra_distance, *args, **kwargs):
        new_kwargs = self.kwargs.copy()
        new_kwargs.update(kwargs)  # add keyword arguments given in __init__
        if intra_distance:
            assert _validate_shape_intradistance(shape)
            out = _intradistance_aux(shape)
            return out * self.f((shape[0], 1), *args, **new_kwargs)
        else:
            assert _validate_shape(shape)
            return self.f(shape, *args, **new_kwargs)


def gap(shape, param):
    out = np.zeros(shape)
    try:
        for j in range(shape[1]):
            new_shape = (2 * shape[0], 1)
            aux = np.random.normal(0, param, new_shape)
            med_aux = _aux_rms(aux)
            median = np.median(med_aux)
            out[:, j] = aux[med_aux > median][: shape[0]]
    except BaseException:
        j = 0
        new_shape = (2 * shape[0], 1)
        aux = np.random.normal(0, param, new_shape)
        med_aux = _aux_rms(aux)
        median = np.median(med_aux)
        out[:, j] = aux[med_aux > median][: shape[0]]
    return out


def get_dist_function(d):
    """
    Transforms distribution name into respective function.
    Args:
        d (str or function): Input distribution str/function.
    Returns:
        function: Actual function to compute the intended distribution.
    """
    if isinstance(d, Distribution):
        return d
    elif hasattr(d, "__call__"):
        return Distribution(d)
    elif isinstance(d, string_types):
        try:
            return Distribution(distributions_list[d])
        except KeyError:
            raise ValueError(
                'Invalid distribution name "'
                + d
                + '". Available names are: '
                + ", ".join(distributions_list.keys())
            )
    else:
        raise ValueError("Invalid distribution input!")
