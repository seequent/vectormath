"""matutils.py defines several utilities for tensor/OUV/vector manipulation"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import numpy as np

from . import vector


def mkvc(vec):
    """Flatten vector with Fortran (column-major) ordering"""
    return vec.flatten(order='F')


def ndgrid(*args, **kwargs):
    """Form tensorial grid for 1, 2, or 3 dimensions.

    Returns as column vectors by default.
    To return as matrix input::

        ndgrid(..., vector=False)

    The inputs can be a list or separate arguments.

    .. code::

        a = np.array([1, 2, 3])
        b = np.array([1, 2])
        XY = ndgrid(a, b)

        > [[1 1]
           [2 1]
           [3 1]
           [1 2]
           [2 2]
           [3 2]]

        X, Y = ndgrid(a, b, vector=False)

        > X = [[1 1]
               [2 2]
               [3 3]]
        > Y = [[1 2]
               [1 2]
               [1 2]]
    """

    # Read the keyword arguments, and only accept a vector=True/False
    vec = kwargs.pop('vector', True)
    assert isinstance(vec, bool), '\'vector\' keyword must be a bool'
    assert len(kwargs) == 0, 'Only \'vector\' keyword accepted'

    # you can either pass a list [x1, x2, x3] or each separately
    if isinstance(args[0], list):
        xin = args[0]
    else:
        xin = args

    # Each vector needs to be a numpy array
    assert np.all([isinstance(x, np.ndarray) for x in xin]), (
        'All vectors must be numpy arrays'
    )

    if len(xin) == 1:
        return xin[0]
    elif len(xin) == 2:
        xy_arr = np.broadcast_arrays(
            mkvc(xin[1]),
            mkvc(xin[0])[:, np.newaxis]
        )
        if vec:
            x2_arr, x1_arr = [mkvc(x) for x in xy_arr]
            return np.c_[x1_arr, x2_arr]
        else:
            return xy_arr[1], xy_arr[0]
    elif len(xin) == 3:
        xyz_arr = np.broadcast_arrays(
            mkvc(xin[2]),
            mkvc(xin[1])[:, np.newaxis],
            mkvc(xin[0])[:, np.newaxis, np.newaxis]
        )
        if vec:
            x3_arr, x2_arr, x1_arr = [mkvc(x) for x in xyz_arr]
            return np.c_[x1_arr, x2_arr, x3_arr]
        else:
            return xyz_arr[2], xyz_arr[1], xyz_arr[0]


def ouv_to_vec(O, U, V, n):                                                    #pylint: disable=invalid-name
    """A grid in the parallelogram defined by the OUV Vector3s and n cells

    If len(n) == 2, n is (nx, ny)
    If n is an integer, n is the number of cells in the longer dimension
    (and cell x-width = y-width)
    """
    aspect = U.length / V.length
    if isinstance(n, (list, tuple)) and len(n) == 2:
        numx, numy = int(n[0]), int(n[1])
    elif np.isscalar(n):
        numx = int(n) if aspect >= 1 else int(n*aspect)
        numy = int(n / aspect) if aspect >= 1 else int(n)
    else:
        raise ValueError('n must be a scalar or a list/tuple of 2 scalars')
    square = ndgrid(np.linspace(0, 1, numx), np.linspace(1, 0, numy))
    X = O.x + U.x * square[:, 0] + V.x * square[:, 1]
    Y = O.y + U.y * square[:, 0] + V.y * square[:, 1]
    Z = O.z + U.z * square[:, 0] + V.z * square[:, 1]
    vec = vector.Vector3Array(X, Y, Z)
    return vec, (numx, numy)


def switch_ouvz(ouvz1, ouvz2, vec):
    """Switches a vector from one OUVZ space to another."""
    O1, U1, V1, Z1 = ouvz1                                                     #pylint: disable=invalid-name
    O2, U2, V2, Z2 = ouvz2                                                     #pylint: disable=invalid-name
    opu = (vec - O1).dot(vector.Vector3Array(U1).normalize()) / U1.length
    opv = (vec - O1).dot(vector.Vector3Array(V1).normalize()) / V1.length
    opz = (vec - O1).dot(vector.Vector3Array(Z1).normalize()) / Z1.length
    return O2 + U2.as_percent(opu) + V2.as_percent(opv) + Z2.as_percent(opz)


def transform_ouv(ouvz1, ouvz2, ouv):
    """Transforms a OUV reference from one OUVZ space to another."""
    o_in, u_in, v_in = ouv
    o_out = switch_ouvz(ouvz1, ouvz2, o_in)
    u_out = switch_ouvz(ouvz1, ouvz2, o_in + u_in) - o_out
    v_out = switch_ouvz(ouvz1, ouvz2, o_in + v_in) - o_out
    return (o_out, u_out, v_out)

def get_sd_from_normal(normal):                                                #pylint: disable=unused-argument
    """Get Strike and Dip from normal"""
    raise NotImplementedError('Get Strike/Dip from normal is not implemented')
