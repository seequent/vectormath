from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import numpy as np

from .vector import Vector3


class Matrix3(np.ndarray):
    """
    3x3 matrix::

        Matrix3(np.identity(3))

    or make rotation::

        Matrix3(strike, 'Z')
        Matrix3(angle, axis of rotation)


    .. note::

        If an angle is provided it is assumed to be in **degrees**.

    """

    def __new__(cls, M=None, axis=None):
        if M is None and axis is None:
            M = np.identity(3)
        elif isinstance(M, list) and axis is None:
            M = np.array(M)
        elif M is not None and axis is None:
            pass
        elif np.isscalar(M):
            theta = np.deg2rad(float(M))
            c = np.cos(theta)
            s = np.sin(theta)

            if isinstance(axis, Vector3):
                t = 1 - c
                x = axis.x
                y = axis.y
                z = axis.z
                tx = t * x
                ty = t * y
                M = np.array([
                    [tx * x + c, tx * y - s * z, tx * z + s * y],
                    [tx * y + s * z, ty * y + c, ty * z - s * x],
                    [tx * z - s * y, ty * z + s * x, t * z * z + c]])
            else:
                if axis not in ['X', 'Y', 'Z']:
                    raise TypeError(
                        "Axis must be either a Vector3 or 'X', 'Y', or 'Z'"
                    )

                if axis == 'X':
                    M = np.array([
                        [1, 0, 0],
                        [0, c, -s],
                        [0, s, c]])
                elif axis == 'Y':
                    M = np.array([
                        [c, 0, s],
                        [0, 1, 0],
                        [-s, 0, c]])
                elif axis == 'Z':
                    M = np.array([
                        [c, -s, 0],
                        [s, c, 0],
                        [0, 0, 1]])

        if not isinstance(M, np.ndarray) or M.shape != (3, 3):
            raise ValueError('Must be a 3x3 matrix.')
        return np.asarray(M).view(cls)

    def __array_finalize__(self, obj):
        if obj is None:
            return

    def __mul__(self, v):
        if isinstance(v, Vector3):
            return Vector3(self.dot(v.T).T)
        elif isinstance(v, Matrix3):
            return Matrix3(self.dot(v))
        return Matrix3(self.view(np.ndarray) * v)
