"""matrix.py defines the Matrix class"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import numpy as np

from .vector import Vector3, Vector3Array


class Matrix3(np.ndarray):                                                     #pylint: disable=too-few-public-methods
    """Primitive 3x3 matrix

    New Matrix3 can be created with:
        - 3x3 array or matrix
        - angle and axis of rotation.
        - nothing (returns identity matrix)

    Examples:

    .. code:: python

        mat_ident = Matrix3(np.identity(3))

        strike = 135
        mat_from_angle = Matrix3(strike, 'Z')

    .. note::

        If an angle is provided it is assumed to be in **degrees**.

    """

    def __new__(cls, mat=None, axis=None):
        if mat is None and axis is None:
            mat = np.identity(3)
        elif isinstance(mat, list) and axis is None:
            mat = np.array(mat)
        elif mat is not None and axis is None:
            pass
        elif np.isscalar(mat):
            theta = np.deg2rad(float(mat))                                     #pylint: disable=no-member
            c = np.cos(theta)                                                  #pylint: disable=invalid-name
            s = np.sin(theta)                                                  #pylint: disable=invalid-name

            if isinstance(axis, Vector3):
                t = 1 - c                                                      #pylint: disable=invalid-name
                x = axis.x
                y = axis.y
                z = axis.z
                tx = t * x                                                     #pylint: disable=invalid-name
                ty = t * y                                                     #pylint: disable=invalid-name
                mat = np.array([
                    [tx * x + c, tx * y - s * z, tx * z + s * y],
                    [tx * y + s * z, ty * y + c, ty * z - s * x],
                    [tx * z - s * y, ty * z + s * x, t * z * z + c]
                ])
            else:
                if axis not in ('X', 'Y', 'Z'):
                    raise TypeError('Axis must be either a Vector3 or \'X\', '
                                    '\'Y\', or \'Z\'')

                if axis == 'X':
                    mat = np.array([
                        [1, 0, 0],
                        [0, c, -s],
                        [0, s, c]])
                elif axis == 'Y':
                    mat = np.array([
                        [c, 0, s],
                        [0, 1, 0],
                        [-s, 0, c]])
                elif axis == 'Z':
                    mat = np.array([
                        [c, -s, 0],
                        [s, c, 0],
                        [0, 0, 1]])

        if not isinstance(mat, np.ndarray) or mat.shape != (3, 3):
            raise ValueError('Must be a 3x3 matrix.')
        return np.asarray(mat).view(cls)

    def __array_finalize__(self, obj):                                         #pylint: disable=no-self-use
        """This is called at the end of array creation

        obj depends on the context. Currently, this does not need to do
        anything regardless of context. See `subclassing docs
        <https://docs.scipy.org/numpy/user/basics.subclassing.html>`_.
        """
        if obj is None:
            return

    def __mul__(self, multiplier):
        """Multiply Matrix by Vector, another Matrix, or scalar"""
        if isinstance(multiplier, (Vector3, Vector3Array)):
            return multiplier.__class__(self.dot(multiplier.T).T)
        elif isinstance(multiplier, Matrix3):
            return Matrix3(self.dot(multiplier))
        return Matrix3(self.view(np.ndarray) * multiplier)
