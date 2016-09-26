from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from builtins import super
from builtins import map
import numpy as np

from .plane import Plane


class Parallelogram(Plane):

    def __init__(self, *args):
        """

            signature:

            O, U, V

            O, u, v

            u, v

            OUV


            You can always do this (uses python iteration)::

                O, U, V = OUV()

        """

        if not args:
            O, U, V = Vector(), Vector(1, 0, 0), Vector(0, 1, 0)
        elif len(args) == 2:
            u, v = list(map(float, args))
            O, U, V = Vector(), Vector(u, 0, 0), Vector(0, v, 0)
        elif len(args) == 3 and np.all(list(map(isvector, args))):
            O, U, V = list(map(Vector, args))
        elif len(args) == 3 and isinstance(args[0], Vector):
            O = Vector(args[0])
            u, v = list(map(float, args[1:]))
            U, V = Vector(u, 0, 0), Vector(0, v, 0)
        else:
            raise Exception('Unknown inputs. See help!')

        self.O, self.U, self.V = O, U, V

        super().__init__(self.O, self.O+self.U, self.O+self.V)

    def __iter__(self):
        self._current = 0
        return self

    # Assuming the pt lies on the plane..., see if it lies in the ouv
    def pt_in_OUV(self, pt):

        a = self.O
        b = self.O + self.U
        c = self.O + self.V
        d = b + self.V

        v1 = (a-pt).cross(b-pt)
        v2 = (b-pt).cross(c-pt)
        v3 = (c-pt).cross(a-pt)

        # is in Triangle abc?
        if (
                np.all(v1.dot(v2) >= 0) and
                np.all(v2.dot(v3) >= 0) and
                np.all(v3.dot(v1) >= 0)
           ):
            return True
        # else, is in Triangle dbc?
        else:
            v1 = (b-pt).cross(c-pt)
            v2 = (c-pt).cross(d-pt)
            v3 = (d-pt).cross(b-pt)

            return (
                np.all(v1.dot(v2) >= 0) and
                np.all(v2.dot(v3) >= 0) and
                np.all(v3.dot(v1) >= 0)
            )

    def intersect_face(self, a, b, c):

        from utils3pt.Mesh import RayCaster
        rc = RayCaster(self.O, self.U)

        ret = rc.intersectTriangle(a, b, c)

        if ret is None:
            rc.d = self.V
            ret = rc.intersectTriangle(a, b, c)
        if ret is None:
            rc.ro = self.O+self.U
            rc.d = self.V
            ret = rc.intersectTriangle(a, b, c)
        if ret is None:
            rc.ro = self.O+self.V
            rc.d = self.U
            ret = rc.intersectTriangle(a, b, c)

        return ret

    def __next__(self):
        if self._current > 2:
            raise StopIteration
        else:
            self._current += 1
            return [self.O, self.U, self.V][self._current-1]
