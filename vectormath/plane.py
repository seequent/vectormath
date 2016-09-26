from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import numpy as np
from .vector import Vector3
from .matrix3 import Matrix3


class Plane(object):

    def __init__(self, *args):
        """

            Signature:

            Normal

            Strike, Dip

            Normal, Point

            Strike, Dip, Point

            Point, Point, Point

        """
        if len(args) == 1:
            self._fromNormalAndPoint(args[0], Vector3())
        elif len(args) == 2 and np.isscalar(args[0]):
            self._fromStrikeAndDip(args[0], args[1], Vector3())
        elif len(args) == 2:
            self._fromNormalAndPoint(args[0], args[1])
        elif len(args) == 3 and np.isscalar(args[0]):
            self._fromStrikeAndDip(*args)
        elif len(args) == 3:
            self._from3Points(*args)
        else:
            raise ValueError(
                'Plane() takes 1, 2, or 3 arguments ({} given)'.format(
                    len(args)
                )
            )

    def __str__(self):
        return 'Plane. A={:4.4f}, B={:4.4f}, C={:4.4f}, D={:4.4f}'.format(
            self.A, self.B, self.C, self.D
        )

    @property
    def strike(self):
        if getattr(self, '_strike', None) is None:
            self._strike, self._dip = getStrikeAndDipFromNormal(self.normal)
        return self._strike

    @property
    def dip(self):
        if getattr(self, '_dip', None) is None:
            self._strike, self._dip = getStrikeAndDipFromNormal(self.normal)
        return self._dip

    @property
    def A(self):
        return self.normal.x

    @property
    def B(self):
        return self.normal.y

    @property
    def C(self):
        return self.normal.z

    @property
    def D(self):
        return self.centroid.dot(self.normal)

    def _from3Points(self, a, b, c):
        a, b, c = Vector3(a), Vector3(b), Vector3(c)
        if a.nV > 1 or b.nV > 1 or c.nV > 1:
            raise ValueError('Arguments must be single points')
        if (
                np.array_equal(a, b) or
                np.array_equal(a, c) or
                np.array_equal(b, c)
           ):
            raise ValueError('Must provide three unique points')
        if np.array_equal((b-a).normalize(), (c-a).normalize()):
            raise ValueError('Points must not be co-linear')
        self.planepts = [a, b, c]
        self.centroid = (a + b + c)/3.0
        self.normal = (b-a).cross(c-a).normalize()

    def _fromStrikeAndDip(self, s, d, pt):
        if not (-180 <= s <= 360):
            raise ValueError(
                'Strike must be value between 0 and 360: {:4.2f}'.format(s)
            )
        if not (0 <= d <= 90):
            raise ValueError(
                'Dip must be value between 0 and 90: {:4.2f}'.format(s)
            )
        N = Matrix3(-s, 'Z')*(Matrix3(d, 'Y')*Vector3(0, 0, 1))
        self._fromNormalAndPoint(N, pt)

    def _fromNormalAndPoint(self, N, pt):
        N = Vector3(N)
        pt = Vector3(pt)
        if N.nV > 1 or pt.nV > 1:
            raise ValueError(
                'Must only provide one vector for normal and point'
            )
        if N.length == 0:
            raise ValueError('Must be a non-zero normal')
        self.normal = N.normalize()
        self.centroid = pt

    def dist(self, pt, tol=1e-6):
        dist = self.normal.dot(pt) - self.D
        if isinstance(pt.x, float):
            if(np.abs(dist) < tol):
                dist = 0
        elif isinstance(pt.x, np.ndarray):
            dist[np.abs(dist) < tol] = 0
        return dist
