from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from . import vector
import numpy as np


def mkvc(v):
    return v.flatten(order='F')


def ndgrid(*args, **kwargs):
    """
    Form tensorial grid for 1, 2, or 3 dimensions.
    Returns as column vectors by default.
    To return as matrix input:
        ndgrid(..., vector=False)
    The inputs can be a list or separate arguments.
    e.g.::
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
    vector = kwargs.pop('vector', True)
    assert type(vector) == bool, "'vector' keyword must be a bool"
    assert len(kwargs) == 0, "Only 'vector' keyword accepted"

    # you can either pass a list [x1, x2, x3] or each seperately
    if type(args[0]) == list:
        xin = args[0]
    else:
        xin = args

    # Each vector needs to be a numpy array
    assert np.all([isinstance(x, np.ndarray) for x in xin]), (
        "All vectors must be numpy arrays."
    )

    if len(xin) == 1:
        return xin[0]
    elif len(xin) == 2:
        XY = np.broadcast_arrays(
            mkvc(xin[1]),
            mkvc(xin[0])[:, np.newaxis]
        )
        if vector:
            X2, X1 = [mkvc(x) for x in XY]
            return np.c_[X1, X2]
        else:
            return XY[1], XY[0]
    elif len(xin) == 3:
        XYZ = np.broadcast_arrays(
            mkvc(xin[2]),
            mkvc(xin[1])[:, np.newaxis],
            mkvc(xin[0])[:, np.newaxis, np.newaxis]
        )
        if vector:
            X3, X2, X1 = [mkvc(x) for x in XYZ]
            return np.c_[X1, X2, X3]
        else:
            return XYZ[2], XYZ[1], XYZ[0]


def ouv2vec(O, U, V, n):
    aspect = U.length / V.length

    if type(n) in [list, tuple]:
        nx, ny = map(int, n)
    else:
        nx = int(n) if aspect >= 1 else int(n*aspect)
        ny = int(n / aspect) if aspect >= 1 else int(n)

    square = ndgrid(np.linspace(0, 1, nx), np.linspace(1, 0, ny))
    X = O.x + U.x * square[:, 0] + V.x * square[:, 1]
    Y = O.y + U.y * square[:, 0] + V.y * square[:, 1]
    Z = O.z + U.z * square[:, 0] + V.z * square[:, 1]
    vec = vector.Vector3(X, Y, Z)
    return vec, (nx, ny)


def switchOUVZ(OUVZ1, OUVZ2, v):
    """Switches a vector from one OUVZ space to another."""
    O1, U1, V1, Z1 = OUVZ1
    O2, U2, V2, Z2 = OUVZ2
    opu = (v - O1).dot(Vector3(U1).normalize()) / U1.length
    opv = (v - O1).dot(Vector3(V1).normalize()) / V1.length
    opz = (v - O1).dot(Vector3(Z1).normalize()) / Z1.length
    return O2 + U2.asPercent(opu) + V2.asPercent(opv) + Z2.asPercent(opz)


def transformOUV(OUVZ1, OUVZ2, OUV):
    """Transforms a OUV reference from one OUVZ space to another."""
    o, u, v = OUV
    O = switchOUVZ(OUVZ1, OUVZ2, o)
    U = switchOUVZ(OUVZ1, OUVZ2, o + u) - O
    V = switchOUVZ(OUVZ1, OUVZ2, o + v) - O
    return O, U, V
