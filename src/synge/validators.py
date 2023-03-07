from __future__ import division
from six import string_types
from numbers import Number
import six
from synge.distribution import get_dist_function
from synge.functions import get_shape_function


def _validate_corr(corr):
    """
    Checks validity of input for `corr`.
    Args:
        corr (float): Input to check validity.
    Returns:
        bool: True if valid. Raises exception if not.
    """
    if not isinstance(corr, Number):
        raise ValueError('Invalid input value for "corr"! Values must be numeric')
    if not 0 <= corr <= 1:
        raise ValueError(
            'Invalid input value for "corr"! Values must be between 0 and 1.'
        )
    return True


def _validate_parametres_distributions(parametres_distributions):
    """
    Checks validity of input for `parametres_distributions`.

    Args:
        parametres_distributions (float): Input to check validity.

    Returns:
        bool: True if valid. Raises exception if not.
    """
    if len(parametres_distributions) != 2:
        raise ValueError(
            'Invalid input value for "parametres_distributions"! Values must be numeric'
        )

    # TODO 0 <= parametres_distributions <= 1 ?

    return True


def _validate_parametres_shapes(parametres_shapes):
    """
    Checks validity of input for `parametres_shapes`.

    Args:
        parametres_shapes (float): Input to check validity.

    Returns:
        bool: True if valid. Raises exception if not.
    """
    if parametres_shapes is not None and not isinstance(parametres_shapes, list):
        raise ValueError(
            'Invalid input value for "parametres_shapes"! Values must be tuple (cote,centr,cdt) where cdt in  [+ and -]'
        )
    # TODO 0 <= parametres_distributions <= 1 ?
    if parametres_shapes is not None and len(parametres_shapes) != 3:
        raise ValueError(
            "Invalid input value for parametres_shapes ! values must be tuple(cote,cntr,cdt) where cdy in [+ and - ]"
        )
    return True


def _validate_parametres_and_shapes(parametres_shapes, shapes):
    """
    Checks validity of input for `parametres_shapes` and 'shapes.

    Args:
        parametres_shapes :
        shapes:

    Returns:
        bool: True if valid. Raises exception if not.
    #"""
    # if len(parametres_shapes)!=len(shapes):
    #     raise ValueError('Invalid input value for "parametres_distributions"! Values must be tuple (cote,centr,cdt) where cdt in [+, -]')
    # else:
    #   if np.any(type(parametres_shapes[i])!=type(shapes[i]) for i in range(len(shapes))):
    #     raise ValueError('Invalid input value for "parametres_distributions" and "shapes"! Values must be of the same sizes')
    #   else:
    #      for i in range(len(shapes)):
    #        if type(shapes[i])==list:
    #          if len(shapes[i])!= len(parametres_shapes[i]):
    #            raise ValueError('Invalid input value for "parametres_distributions" and "shapes"! Values must be of the same sizes')
    return True


def _validate_alpha_n(alpha_n):
    """
    Checks validity of input for `alpha_n`.
    Args:
        alpha_n (float): Input to check validity.
    Returns:
        bool: True if valid. Raises exception if not.
    """
    if not isinstance(alpha_n, Number):
        raise ValueError('Invalid input for "alpha_n"! Values must be numeric.')
    if alpha_n == 0:
        raise ValueError(
            'Invalid input for "alpha_n"! Values must be different from 0.'
        )
    return True


def _validate_scale(scale):
    """
    Checks validity of input for `scale`.
    Args:
        scale (bool): Input to check validity.
    Returns:
        bool: True if valid. Raises exception if not.
    """
    if scale not in [True, None, False]:
        raise ValueError(
            'Invalid input value for "scale"! Input must be boolean (or None).'
        )
    return True


def _validate_rotate(rotate):
    """
    Checks validity of input for `rotate`.
    Args:
        rotate (bool): Input to check validity.
    Returns:
        bool: True if valid. Raises exception if not.
    """
    if rotate not in [True, False]:
        raise ValueError('Invalid input for "rotate"! Input must be boolean.')
    return True


def _validate_n_noise(n_noise, n_feats):
    """
    Checks validity of input for `n_noise`.
    Args:
        n_noise (list of int): Input to check validity.
        n_feats (int): Number of dimensions/features.
    Returns:
    """
    if not hasattr(n_noise, "__iter__"):
        raise ValueError('Invalid input for "n_noise"! Input must be a list.')
    if len(n_noise) > n_feats:
        raise ValueError(
            'Invalid input for "n_noise"! Input has more dimensions than total number of dimensions.'
        )
    if not all(isinstance(n, six.integer_types) for n in n_noise):
        raise ValueError(
            'Invalid input for "n_noise"! Input dimensions must be integers.'
        )
    if not all(0 <= n < n_feats for n in n_noise):
        raise ValueError(
            'Invalid input for "n_noise"! Input dimensions must be in the interval [0, "n_feats"[.'
        )
    return True


def _validate_mv(mv):
    """
    Checks validity of input for `mv`.
    Args:
        mv (bool): Input to check validity
    Returns:
        bool: True if valid. Raises exception if not.
    """
    if mv not in [True, None, False]:
        raise ValueError('Invalid input value for "mv"!')
    return True


def _initialize_distribution(distributions):
    """
    Checks if the input distributions are valid. That is, check if they are either strings or functions. If they are
    strings, also check if they are contained in `distributions_list`.
    Args:
        distributions (list of list of (str or function)): Distributions given as input.
    Returns:
        (list of list of function): Functions for the distributions given as input.
    """
    return [
        [get_dist_function(d) for d in lst]
        if hasattr(lst, "__iter__") and not isinstance(lst, string_types)
        else get_dist_function(lst)
        for lst in distributions.values()
    ]


def _initialize_secondary_shape(Shapes):
    """
    Checks if the input Shape are valid. That is, check if they are either strings or functions. If they are
    strings, also check if they are contained in `Shape`.
    Args:
        Shape (list of list of (str or function)): Shapes given as input.
    Returns:
        (list of list of function): Functions for the Shape given as input.
    """
    out = [
        [["" for i in j] if isinstance(j, list) else "" for j in k]
        if isinstance(k, list)
        else ""
        for k in Shapes
    ]
    for i, shape in enumerate(Shapes):
        if shape is not None:
            if isinstance(shape, list):
                for j, s in enumerate(shape):
                    if isinstance(s, list):
                        for k, l in enumerate(s):
                            if not isinstance(l, string_types):
                                raise ValueError("Invalid Shape input!")
                            else:
                                out[i][j][k] = get_shape_function(l)
                    else:
                        out[i][j] = get_shape_function(s)
            elif isinstance(shape, string_types):
                out[i] = get_shape_function(shape)
        else:
            out[i] = None

    return out


def _initialize_prinicipal_shape(Shapes_):
    """
    Checks if the input Shape are valid. That is, check if they are either strings or functions. If they are
    strings, also check if they are contained in `Shape`.
    Args:
        Shape (list of list of (str or function)): Shapes given as input.
    Returns:
        (list of list of function): Functions for the Shape given as input.
    """
    out = {}
    for i, (shape, j) in Shapes_.items():
        if shape is not None:
            if isinstance(shape, string_types):
                out[i] = (get_shape_function(shape), j)
        else:
            out[i] = Shapes_[i]

    return out
