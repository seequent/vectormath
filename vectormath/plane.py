"""plane.py defines Plane class"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import numpy as np

from .matrix import Matrix3
from .matutils import get_sd_from_normal
from .vector import Vector3


class Plane(object):
    """Construct a geometric plane

    Planes can be constructed with:
        - Normal Vector
        - Normal Vector and Intersection Point
        - Strike and Dip Vectors
        - Strike and Dip Vectors and Intersection Point
        - Three Points
    """

    def __init__(self, *args):
        if len(args) == 1:
            norm, cent = self._from_normal_and_point(args[0], Vector3())
        elif len(args) == 2 and np.isscalar(args[0]):
            norm, cent = self._from_strike_and_dip(args[0], args[1], Vector3())
        elif len(args) == 2:
            norm, cent = self._from_normal_and_point(args[0], args[1])
        elif len(args) == 3 and np.isscalar(args[0]):
            norm, cent = self._from_strike_and_dip(*args)
        elif len(args) == 3:
            norm, cent = self._from_three_points(*args)
        else:
            raise ValueError(
                'Plane() takes 1, 2, or 3 arguments ({} given)'.format(
                    len(args)
                )
            )
        self.normal = norm.normalize()
        self.centroid = cent

    @classmethod
    def _from_three_points(cls, a, b, c):                                      #pylint: disable=invalid-name
        a, b, c = Vector3(a), Vector3(b), Vector3(c)
        if (
                np.array_equal(a, b) or
                np.array_equal(a, c) or
                np.array_equal(b, c)
        ):
            raise ValueError('Must provide three unique points')
        if np.array_equal((b-a).normalize(), (c-a).normalize()):
            raise ValueError('Points must not be co-linear')
        return (b-a).cross(c-a), (a + b + c)/3.0

    @classmethod
    def _from_strike_and_dip(cls, strike, dip, point):
        if not -180 <= strike <= 360:
            raise ValueError('Strike must be value between 0 and 360: '
                             '{:4.2f}'.format(strike))
        if not 0 <= dip <= 90:
            raise ValueError('Dip must be value between 0 and 90: '
                             '{:4.2f}'.format(strike))
        normal = Matrix3(-strike, 'Z')*(Matrix3(dip, 'Y')*Vector3(0, 0, 1))
        return normal, point

    @classmethod
    def _from_normal_and_point(cls, normal, point):
        normal = Vector3(normal)
        point = Vector3(point)
        if normal.length == 0:
            raise ValueError('Must be a non-zero normal')
        return normal, point

    def __str__(self):
        return 'Plane. A={:4.4f}, B={:4.4f}, C={:4.4f}, D={:4.4f}'.format(
            self.A, self.B, self.C, self.D
        )

    @property
    def strike(self):
        """Strike of plane calculated from normal"""
        return get_sd_from_normal(self.normal)[0]

    @property
    def dip(self):
        """Dip of plane calculated from normal"""
        return get_sd_from_normal(self.normal)[1]

    @property
    def A(self):                                                               #pylint: disable=invalid-name
        """x-component of normal"""
        return self.normal.x

    @property
    def B(self):                                                               #pylint: disable=invalid-name
        """y-component of normal"""
        return self.normal.y

    @property
    def C(self):                                                               #pylint: disable=invalid-name
        """z-component of normal"""
        return self.normal.z

    @property
    def D(self):                                                               #pylint: disable=invalid-name
        """Dot product of centroid and normal"""
        return self.centroid.dot(self.normal)

    def dist(self, point, tol=1e-6):
        """Distance of point from plane within a tolerance (default: 1e-6)"""
        dist = self.normal.dot(point) - self.D
        if isinstance(point.x, float) and np.abs(dist) < tol:
            dist = 0
        elif isinstance(point.x, np.ndarray):
            dist[np.abs(dist) < tol] = 0
        return dist
