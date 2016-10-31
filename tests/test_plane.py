from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import unittest
from vectormath import Vector3, Vector3Array, Plane
import numpy as np
import six


class TestVMathPlane(unittest.TestCase):

    def test_init_exceptions(self):
        self.assertRaises(ValueError, Plane)
        v1 = Vector3()
        self.assertRaises(ValueError, Plane, v1, v1, v1, v1)
        self.assertRaises(ValueError, Plane, v1)
        v2 = Vector3([1, 0, 0])
        self.assertRaises(ValueError, Plane, v1, v2, v2)
        self.assertRaises(ValueError, Plane, v2, 2*v2, 5*v2)
        v3 = Vector3Array([[1, 0, 0], [0, 1, 0]])
        self.assertRaises(ValueError, Plane, v3)
        v4 = Vector3([0, 0, 1])
        self.assertRaises(ValueError, Plane, v3, v4, v1)
        self.assertRaises(ValueError, Plane, v4, v3)
        self.assertRaises(Exception, Plane, 3)
        if six.PY3:
            self.assertRaises(TypeError, Plane, 3, 'point')
        else:
            self.assertRaises(ValueError, Plane, 3, 'point')
        self.assertRaises(Exception, Plane, 3, 'point', 'Science')
        self.assertRaises(ValueError, Plane, -181, 5)
        self.assertRaises(ValueError, Plane, 361, 5)
        self.assertRaises(ValueError, Plane, 5, -1)
        self.assertRaises(ValueError, Plane, 5, 91)

    def test_init_fromNormalAndPoint(self):
        v1 = Vector3([1, 0, 0])
        P1 = Plane(v1)
        self.assertTrue(np.array_equal(P1.centroid, Vector3()))
        self.assertTrue(np.array_equal(P1.normal, v1))
        v2 = [1, 2, 3]
        P2 = Plane(v2)
        self.assertTrue(np.array_equal(P2.centroid, Vector3()))
        self.assertTrue(np.array_equal(P2.normal, Vector3(v2).as_unit()))
        P3 = Plane(v1, v2)
        self.assertTrue(np.array_equal(P3.centroid, Vector3(v2)))
        self.assertTrue(np.array_equal(P3.normal, v1))
        P4 = Plane(v2, v1)
        self.assertTrue(np.array_equal(P4.centroid, v1))
        self.assertTrue(np.array_equal(P4.normal, Vector3(v2).as_unit()))

    def test_init_from3Points(self):
        v1 = [0, 0, 0]
        v2 = [6, 0, 0]
        v3 = [0, 3, 0]
        P1 = Plane(v1, v2, v3)
        self.assertTrue(np.array_equal(P1.normal, Vector3([0, 0, 1])))
        self.assertTrue(np.array_equal(P1.centroid, Vector3([2, 1, 0])))
        v4 = [3, 0, 0]
        v5 = [0, 0, 3]
        P2 = Plane(v3, v5, v4)
        self.assertTrue(np.allclose(P2.normal,
                                    Vector3([1, 1, 1]).as_unit()))
        self.assertTrue(np.array_equal(P2.centroid, Vector3([1, 1, 1])))
        P3 = Plane(v3, v4, v5)
        self.assertTrue(np.allclose(P3.normal,
                                    -Vector3([1, 1, 1]).as_unit()))
        self.assertTrue(np.array_equal(P3.centroid, Vector3([1, 1, 1])))

    def test_init_fromStrikeAndDip(self):
        P1 = Plane(0, 0)
        self.assertTrue(np.array_equal(P1.normal, Vector3(0, 0, 1)))
        self.assertTrue(np.array_equal(P1.centroid, Vector3()))
        v1 = [1, 2, 3]
        P2 = Plane(0, 0, v1)
        self.assertTrue(np.array_equal(P2.normal, Vector3(0, 0, 1)))
        self.assertTrue(np.array_equal(P2.centroid, Vector3(v1)))
        P3 = Plane(90, 0)
        self.assertTrue(np.allclose(P3.normal, Vector3(0, 0, 1)))
        self.assertTrue(np.array_equal(P3.centroid, Vector3()))
        P4 = Plane(0, 90)
        self.assertTrue(np.allclose(P4.normal, Vector3(1, 0, 0)))
        P5 = Plane(90, 90)
        self.assertTrue(np.allclose(P5.normal, Vector3(0, -1, 0)))
        P6 = Plane(0, 45)
        self.assertTrue(np.allclose(P6.normal,
                                    Vector3(1, 0, 1).as_unit()))
        P7 = Plane(12.44, 90)
        P8 = Plane(192.44, 90)
        self.assertTrue(np.allclose(P7.normal, -P8.normal))

    def test_get_ABCD(self):
        P1 = Plane([1, 0, 0])
        self.assertTrue(P1.A == 1)
        self.assertTrue(P1.B == 0)
        self.assertTrue(P1.C == 0)
        self.assertTrue(P1.D == 0)


if __name__ == '__main__':
    unittest.main()
