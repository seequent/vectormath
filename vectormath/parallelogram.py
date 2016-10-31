"""parallelogram.py defines Parallelogram class"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from future.utils import implements_iterator
import numpy as np

from .plane import Plane
from .vector import Vector3


@implements_iterator
class Parallelogram(Plane):
    """Construct a Parallelogram

    Parallelograms can be constructed with:
        - O, U, V vectors
        - O vector and u/v axes lengths
        - u/v axes lengths
        - OUV
        - nothing (for O = [0, 0, 0], U = [1, 0, 0], V = [0, 1, 0])
    """

    def __init__(self, *args):

        if not args:
            O, U, V = Vector3(), Vector3(1, 0, 0), Vector3(0, 1, 0)            #pylint: disable=invalid-name
        elif len(args) == 2:
            O, U, V = (Vector3(),                                              #pylint: disable=invalid-name
                       Vector3(float(args[0]), 0, 0),
                       Vector3(0, float(args[1]), 0))
        elif len(args) == 3 and np.all([self.isvector(v) for v in args]):
            O, U, V = (Vector3(v) for v in args)                               #pylint: disable=invalid-name
        elif len(args) == 3 and isinstance(args[0], Vector3):
            O, U, V = (Vector3(args[0]),                                       #pylint: disable=invalid-name
                       Vector3(float(args[1]), 0, 0),
                       Vector3(0, float(args[2]), 0))
        else:
            raise ValueError('Unknown inputs. See help!')

        self.O, self.U, self.V = O, U, V                                       #pylint: disable=invalid-name

        super(Parallelogram, self).__init__(
            self.O, self.O+self.U, self.O+self.V
        )

    def point_in_ouv(self, point):
        """Assuming the pt lies on the plane, see if it lies in the ouv"""

        a = self.O                                                             #pylint: disable=invalid-name
        b = self.O + self.U                                                    #pylint: disable=invalid-name
        c = self.O + self.V                                                    #pylint: disable=invalid-name
        d = b + self.V                                                         #pylint: disable=invalid-name

        pairs = ((a, b), (b, c), (c, a))

        v1, v2, v3 = ((pt1-point).cross(pt2-point) for pt1, pt2 in pairs)      #pylint: disable=invalid-name

        # is in Triangle abc?
        if (
                np.all(v1.dot(v2) >= 0) and
                np.all(v2.dot(v3) >= 0) and
                np.all(v3.dot(v1) >= 0)
        ):
            return True

        # else, is in Triangle dbc?
        pairs = ((b, c), (c, d), (d, b))

        v1, v2, v3 = ((pt1-point).cross(pt2-point) for pt1, pt2 in pairs)      #pylint: disable=invalid-name

        return (
            np.all(v1.dot(v2) >= 0) and
            np.all(v2.dot(v3) >= 0) and
            np.all(v3.dot(v1) >= 0)
        )

    def __iter__(self):                                                        #pylint: disable=non-iterator-returned
        self._current = 0                                                      #pylint: disable=attribute-defined-outside-init
        return self

    def __next__(self):
        if self._current > 2:
            raise StopIteration
        else:
            self._current += 1
            return [self.O, self.U, self.V][self._current-1]

    @staticmethod
    def isvector(vec):
        """Test if input is a Vector3"""
        try:
            Vector3(vec)
        except Exception:                                                      #pylint: disable=broad-except
            return False
        return True
