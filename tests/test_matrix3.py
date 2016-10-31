from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import unittest
from vectormath import Vector3, Vector3Array, Matrix3
import numpy as np


class TestVMathMatrix(unittest.TestCase):

    def test_init(self):
        self.assertRaises(ValueError, Matrix3, np.zeros((4, 4)))
        self.assertRaises(ValueError, Matrix3, 3)
        self.assertRaises(ValueError, Matrix3, 'Hello Mr. Anderson')
        self.assertRaises(TypeError, Matrix3, +90, 'Up')
        self.assertRaises(ValueError, Matrix3, ([1, 1, 1],
                                                [1, 1, 0],
                                                [0, 0, 0]))
        M1 = Matrix3()
        self.assertTrue(np.array_equal(M1, np.identity(3)))
        M2 = Matrix3(np.array([[1, 1, 1],
                               [1, 1, 0],
                               [0, 0, 0]]))
        self.assertTrue(np.array_equal(M2, np.array([[1, 1, 1],
                                                     [1, 1, 0],
                                                     [0, 0, 0]])))
        self.assertTrue(isinstance(M2, Matrix3))
        M3 = Matrix3(M2)
        self.assertTrue(M2 is not M3)
        self.assertTrue(np.array_equal(M2, M3))
        v1 = Vector3(1, 0, 0)
        M4 = Matrix3(45, 'X')
        M5 = Matrix3(45, v1)
        self.assertTrue(np.array_equal(M4, M5))
        v2 = Vector3(0, 1, 0)
        M6 = Matrix3(30, 'Y')
        M7 = Matrix3(30, v2)
        self.assertTrue(np.array_equal(M6, M7))
        v3 = Vector3(0, 0, 1)
        M8 = Matrix3(90, 'Z')
        M9 = Matrix3(90, v3)
        self.assertTrue(np.array_equal(M8, M9))
        M10 = Matrix3([[1, 1, 1],
                      [1, 1, 0],
                      [0, 0, 0]])
        self.assertTrue(np.array_equal(M2, M10))
        self.assertTrue(isinstance(M10, Matrix3))

    def test_mult(self):
        M1 = Matrix3()
        M2 = M1 * 3
        M3 = Matrix3(np.identity(3)*3)
        self.assertTrue(isinstance(M2, Matrix3))
        self.assertTrue(np.array_equal(M2, M3))
        M4 = 3 * M1
        self.assertTrue(np.array_equal(M4, M3))
        v1 = Vector3Array()
        v2 = M1 * v1
        self.assertTrue(np.array_equal(v1, v2))
        v2 = Vector3Array([[1, 2, 3], [-10, -20, 30]])
        M5 = Matrix3([[.5, 1, 1.5],
                      [-.5, 0, 0],
                      [1, 10, 100]])
        v3 = Vector3Array([[7, -.5, 321], [20, 5, 2790]])
        self.assertTrue(np.array_equal(M5*v2, v3))
        self.assertTrue(np.array_equal(M1, M1*M1))
        M6 = Matrix3([[0, 2, 4],
                      [1, 3, 5],
                      [-2, -4, -6]])
        M7 = Matrix3([[-2, -2, -2],
                      [0, -1, -2],
                      [-190, -368, -546]])
        self.assertTrue(np.array_equal(M7, M5*M6))

    def test_rotation(self):
        v = Vector3(1, 0, 0)
        M1 = Matrix3(+90, 'Z')
        M2 = Matrix3(-90, 'Z')
        self.assertAlmostEqual((M1 * v).x, 0)
        self.assertAlmostEqual((M1 * v).y, 1)
        self.assertAlmostEqual((M1 * v).z, 0)
        self.assertAlmostEqual((M2 * M1 * v).x, 1)
        self.assertAlmostEqual((M2 * M1 * v).y, 0)
        self.assertAlmostEqual((M2 * M1 * v).z, 0)
        M1 = Matrix3(+90, 'Y')
        M2 = Matrix3(-90, 'Y')
        self.assertAlmostEqual((M1 * v).x, 0)
        self.assertAlmostEqual((M1 * v).y, 0)
        self.assertAlmostEqual((M1 * v).z, -1)
        self.assertAlmostEqual((M2 * M1 * v).x, 1)
        self.assertAlmostEqual((M2 * M1 * v).y, 0)
        self.assertAlmostEqual((M2 * M1 * v).z, 0)
        M1 = Matrix3(+90, 'X')
        M2 = Matrix3(-90, 'X')
        self.assertAlmostEqual((M1 * v).x, 1)
        self.assertAlmostEqual((M1 * v).y, 0)
        self.assertAlmostEqual((M1 * v).z, 0)
        self.assertAlmostEqual((M2 * M1 * v).x, 1)
        self.assertAlmostEqual((M2 * M1 * v).y, 0)
        self.assertAlmostEqual((M2 * M1 * v).z, 0)


if __name__ == '__main__':
    unittest.main()
