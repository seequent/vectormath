from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import unittest
import numpy as np

from vectormath import Vector2, Vector2Array, Vector3, Vector3Array


class TestVMathVector3(unittest.TestCase):

    def test_init_exceptions(self):
        self.assertRaises(TypeError, Vector3Array, np.r_[1], np.r_[1], 3)
        self.assertRaises(ValueError,
                          Vector3Array, np.r_[1, 2], np.r_[1], np.r_[1])
        self.assertRaises(ValueError, Vector3Array, np.array([0, 0]))
        self.assertRaises(ValueError,
                          Vector3Array, 'Make', ' me a ', 'vector!')
        self.assertRaises(ValueError, Vector3Array, ([0, 0], [0, 0], [0, 0]))

    def test_init(self):
        v1 = Vector3Array()
        v2 = Vector3Array(0, 0, 0)
        self.assertTrue(np.array_equal(v1, v2))
        v3 = Vector3Array(v1)
        self.assertTrue(np.array_equal(v1, v3))
        self.assertTrue(v1 is not v3)
        v4 = Vector3Array(np.r_[0, 0, 0])
        self.assertTrue(np.array_equal(v1, v4))
        v5 = Vector3Array(np.c_[np.r_[1, 0, 0],
                                np.r_[0, 1, 0],
                                np.r_[0, 0, 1]])
        self.assertTrue(np.array_equal(v5.length, np.r_[1, 1, 1]))
        v6 = Vector3Array(np.r_[1, 0, 0], np.r_[0, 1, 0], np.r_[0, 0, 1])
        self.assertTrue(np.array_equal(v6.length, np.r_[1, 1, 1]))
        v7 = Vector3Array([0, 0, 0])
        self.assertTrue(np.array_equal(v1, v7))
        v8 = Vector3Array(x=0, y=0, z=0)
        self.assertTrue(np.array_equal(v1, v8))
        v9 = Vector3Array(
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0]
        )
        v10 = Vector3Array([
            [0, 0, 0],
            [0, 0, 0],
            [0, 0, 0],
            [0, 0, 0],
            [0, 0, 0]
        ])
        self.assertTrue(np.array_equal(v9, v10))
        v11 = Vector3Array([[[[[0]], [[0]], [[0]]]]])
        self.assertTrue(np.array_equal(v1, v11))
        v12 = Vector3Array([0]*5, [0]*5, [0]*5)
        self.assertTrue(np.array_equal(v10, v12))
        v13 = Vector3Array((0, 0, 0))
        self.assertTrue(np.array_equal(v1, v13))
        v14 = Vector3Array(([0, 0, 0], [0, 0, 0]))
        self.assertTrue(np.array_equal(v14, Vector3Array([0]*2, [0]*2, [0]*2)))

    def test_indexing(self):
        v2 = Vector3Array(1, 2, 3)
        self.assertTrue(v2[0, 0] == 1)
        self.assertTrue(v2[0, 1] == 2)
        self.assertTrue(v2[0, 2] == 3)
        self.assertTrue(len(v2[0]) == 3)

        self.assertRaises(IndexError, lambda: v2[3])
        self.assertRaises(IndexError, lambda: v2[0, 3])
        l = []
        for x in v2[0]:
            l.append(x)
        self.assertTrue(np.array_equal(np.array(l), np.r_[1, 2, 3]))
        self.assertTrue(np.array_equal(v2, Vector3Array(l)))
        l = []
        v3 = Vector3Array([[1, 2, 3],
                           [2, 3, 4]])
        for v in v3:
            l.append(v)
        self.assertTrue(np.array_equal(
            np.array(l),
            np.array([[1, 2, 3], [2, 3, 4]]))
        )
        self.assertTrue(np.array_equal(Vector3Array(l), v3))
        v4 = Vector3Array()
        v4[0, 0] = 1
        v4[0, 1] = 2
        v4[0, 2] = 3
        self.assertTrue(np.array_equal(v2, v4))

    def test_copy(self):
        vOrig = Vector3Array()
        vCopy = vOrig.copy()
        self.assertTrue(np.array_equal(vOrig, vCopy))
        self.assertTrue(vOrig is not vCopy)

    def test_size(self):
        v1 = Vector3Array()
        self.assertTrue(v1.nV == 1)
        v2 = Vector3Array(np.c_[np.r_[1, 0, 0],
                                np.r_[0, 1, 0],
                                np.r_[0, 0, 1]])
        self.assertTrue(v2.nV == 3)
        v3 = Vector3Array(
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0]
        )
        self.assertTrue(v3.nV == 5)
        v4 = Vector3Array(0, 0, 0)
        self.assertTrue(v4.nV == 1)

    def test_setget(self):
        v1 = Vector3Array(1, 1, 1)
        self.assertTrue(v1.x == 1)
        v1.x = 2
        self.assertTrue(v1.x == 2)
        self.assertTrue(v1.y == 1)
        v1.y = 2
        self.assertTrue(v1.y == 2)
        self.assertTrue(v1.z == 1)
        v1.z = 2
        self.assertTrue(v1.z == 2)
        v2 = Vector3Array([[0, 1, 2],
                           [1, 2, 3]])
        self.assertTrue(np.array_equal(v2.x, [0, 1]))
        v2.x = [0, -1]
        self.assertTrue(np.array_equal(v2.x, [0, -1]))
        self.assertTrue(np.array_equal(v2.y, [1, 2]))
        v2.y = [-1, -2]
        self.assertTrue(np.array_equal(v2.y, [-1, -2]))
        self.assertTrue(np.array_equal(v2.z, [2, 3]))
        v2.z = [0, 0]
        self.assertTrue(np.array_equal(v2.z, [0, 0]))

    def test_length(self):
        v1 = Vector3Array(1, 1, 1)
        self.assertTrue(v1.length == np.sqrt(3))
        v2 = Vector3Array(np.r_[1, 2], np.r_[1, 2], np.r_[1, 2])
        self.assertTrue(np.array_equal(v2.length, np.sqrt(np.r_[3, 12])))
        v3 = Vector3Array(1, 0, 0)
        v3.length = 5
        self.assertTrue(v3.length == 5)
        v4 = Vector3Array(np.r_[1, 1], np.r_[0, 0], np.r_[1, 2])

        self.assertRaises(ValueError, lambda: setattr(v4, 'length', 5))
        v5 = Vector3Array(np.r_[1, 0], np.r_[0, 0], np.r_[0, 1])
        self.assertTrue(np.array_equal(v5.length, [1, 1]))
        v5.length = [-1, 3]
        self.assertTrue(np.array_equal(v5, [[-1., -0., -0.], [0., 0., 3.]]))
        self.assertTrue(np.array_equal(v5.length, [1, 3]))
        v6 = Vector3Array()
        self.assertTrue(v6.length == 0)

        self.assertRaises(ZeroDivisionError, lambda: setattr(v6, 'length', 5))
        v6.length = 0
        self.assertTrue(v6.length == 0)
        v7 = Vector3Array(
            [0, 0, 1, 0, 0],
            [0, 1, 0, 0, 0],
            [1, 0, 0, 0, 0]
        )
        length = [5, 5, 5, 5, 5]

        self.assertRaises(ZeroDivisionError,
                          lambda: setattr(v7, 'length', length))
        length = [5, 5, 5, 0, 0]
        v7.length = length
        self.assertTrue(np.array_equal(length, v7.length))

    def test_ops(self):
        v1 = Vector3Array(1, 1, 1)
        v2 = Vector3Array(2, 2, 2)
        self.assertTrue(np.array_equal(v2-v1, v1))
        self.assertTrue(np.array_equal(v1-v2, -v1))
        self.assertTrue(np.array_equal(v1+v1, v2))
        self.assertTrue(np.array_equal(v1*v2, v2))
        self.assertTrue(np.array_equal(v2/v1, v2))
        self.assertTrue(np.array_equal(2*v1, v2))
        self.assertTrue(np.array_equal(v2/2, v1))
        self.assertTrue(np.array_equal(v1+1, v2))
        self.assertTrue(np.array_equal(v2-1, v1))
        v1 = Vector3Array(np.r_[1, 1.], np.r_[1, 1.], np.r_[1, 1.])
        v2 = Vector3Array(np.r_[2, 2.], np.r_[2, 2.], np.r_[2, 2.])
        self.assertTrue(np.array_equal(v2-v1, v1))
        self.assertTrue(np.array_equal(v1-v2, -v1))
        self.assertTrue(np.array_equal(v1+v1, v2))
        self.assertTrue(np.array_equal(v1*v2, v2))
        self.assertTrue(np.array_equal(v2/v1, v2))
        self.assertTrue(np.array_equal(2*v1, v2))
        self.assertTrue(np.array_equal(v2/2, v1))
        self.assertTrue(np.array_equal(v1+1, v2))
        self.assertTrue(np.array_equal(v2-1, v1))

    def test_dot(self):
        v1 = Vector3Array(1, 1, 1)
        v2 = Vector3Array(2, 2, 2)
        self.assertTrue(v1.dot(v2) == 6)
        v1l = Vector3Array(np.r_[1, 1.], np.r_[1, 1.], np.r_[1, 1.])
        v2l = Vector3Array(np.r_[2, 2.], np.r_[2, 2.], np.r_[2, 2.])
        self.assertTrue(np.array_equal(v1l.dot(v2l), np.r_[6, 6]))
        self.assertTrue(np.array_equal(v1.dot(v2l), np.r_[6, 6]))
        self.assertTrue(np.array_equal(v1l.dot(v2), np.r_[6, 6]))
        v3 = Vector3Array([3]*4, [3]*4, [3]*4)

        self.assertRaises(ValueError, lambda: v3.dot(v2l))
        self.assertRaises(TypeError, lambda: v3.dot(5))

    def test_cross(self):
        v1 = Vector3Array(1, 0, 0)
        v2 = Vector3Array(0, 1, 0)
        vC = Vector3Array(0, 0, 1)
        self.assertTrue(np.array_equal(v1.cross(v2), vC))
        v1 = Vector3Array(np.r_[1, 1], np.r_[0, 0], np.r_[0, 0])
        v2 = Vector3Array(np.r_[0, 0], np.r_[1, 1], np.r_[0, 0])
        vC = Vector3Array(np.r_[0, 0], np.r_[0, 0], np.r_[1, 1])
        self.assertTrue(np.array_equal(v1.cross(v2), vC))
        v3 = Vector3Array([3]*4, [3]*4, [3]*4)

        def f(): v3.cross(v2)
        self.assertRaises(ValueError, f)

        def f(): v3.cross(5)
        self.assertRaises(TypeError, f)

    def test_as_percent(self):
        v1 = Vector3Array(10, 0, 0)
        v2 = Vector3Array(20, 0, 0)
        self.assertTrue(np.array_equal(v1.as_percent(2), v2))
        self.assertTrue(np.array_equal(v1, Vector3Array(10, 0, 0)))# not copied
        v3 = Vector3Array(
            [0, 0, 2, 0, 0],
            [0, 2, 0, 0, 0],
            [2, 0, 0, 0, 0]
        )
        v4 = v3 * .5
        self.assertTrue(np.array_equal(v3.as_percent(.5), v4))
        v5 = Vector3Array()
        self.assertTrue(np.array_equal(v5.as_percent(100), v5))
        v6 = Vector3Array(5, 5, 5)
        self.assertTrue(np.array_equal(v6.as_percent(0), v5))

        self.assertRaises(TypeError,
                          lambda: v6.as_percent('One Hundred Percent'))

    def test_normalize(self):
        v1 = Vector3Array(5, 0, 0)
        self.assertTrue(v1.length == 5)
        self.assertTrue(v1.normalize() is v1)
        self.assertTrue(v1.length == 1)
        v2 = Vector3Array()

        self.assertRaises(ZeroDivisionError, lambda: v2.normalize())
        v3 = Vector3Array(
            [0, 0, 2],
            [0, 2, 0],
            [2, 0, 0]
        )
        self.assertTrue(np.array_equal(v3.length, [2, 2, 2]))
        self.assertTrue(v3.normalize() is v3)
        self.assertTrue(np.array_equal(v3.length, [1, 1, 1]))

    def test_as_length(self):
        v1 = Vector3Array(1, 1, 1)
        v2 = v1.as_length(1)
        self.assertTrue(v1 is not v2)
        self.assertTrue(v1.length == np.sqrt(3))
        self.assertTrue(v2.length == 1)

        v3 = Vector3Array(np.r_[1, 2], np.r_[1, 2], np.r_[1, 2])
        v4 = v3.as_length([1, 2])
        self.assertTrue(np.array_equal(v4.length, [1, 2]))

        self.assertRaises(ValueError, lambda: v3.as_length(5))
        v5 = Vector3Array(np.r_[1, 0], np.r_[0, 0], np.r_[0, 1])
        self.assertTrue(np.array_equal(v5.length, [1, 1]))
        v6 = v5.as_length([-1, 3])
        self.assertTrue(np.array_equal(v6, [[-1., -0., -0.], [0., 0., 3.]]))
        self.assertTrue(np.array_equal(v6.length, [1, 3]))
        v7 = Vector3Array()

        self.assertRaises(ZeroDivisionError, lambda: v7.as_length(5))
        v8 = v7.as_length(0)
        self.assertTrue(v8.length == 0)
        v9 = Vector3Array(
            [0, 0, 1, 0, 0],
            [0, 1, 0, 0, 0],
            [1, 0, 0, 0, 0]
        )
        length = [5, 5, 5, 5, 5]

        self.assertRaises(ZeroDivisionError, lambda: v9.as_length(length))
        length = [5, 5, 5, 0, 0]
        v10 = v9.as_length(length)
        self.assertTrue(np.array_equal(length, v10.length))

    def test_as_unit(self):
        v1 = Vector3Array(1, 0, 0)
        v2 = v1.as_unit()
        self.assertTrue(v1 is not v2)
        self.assertTrue(np.array_equal(v1, v2))
        self.assertTrue(v2.length == 1)
        v3 = Vector3Array(np.r_[1, 2], np.r_[1, 2], np.r_[1, 2])
        v4 = v3.as_unit()
        self.assertTrue(np.array_equal(v4.length, [1, 1]))
        v5 = Vector3Array(1, 1, 1)
        v6 = v5.as_unit()
        self.assertTrue(v6.length == 1)
        self.assertTrue(v6.x == v6.y)
        self.assertTrue(v6.z == v6.y)
        v7 = Vector3Array()

        self.assertRaises(ZeroDivisionError, v7.as_unit)
        v9 = Vector3Array(
            [0, 0, 1, 0, 0],
            [0, 1, 0, 0, 0],
            [1, 0, 0, 0, 0]
        )

        self.assertRaises(ZeroDivisionError, v9.as_unit)

    def test_view_types(self):
        v1 = Vector3Array(np.random.rand(100, 3))
        self.assertTrue(isinstance(v1, Vector3Array))
        self.assertTrue(isinstance(v1[1:2], Vector3Array))
        self.assertTrue(isinstance(v1[1:50:2], Vector3Array))
        self.assertTrue(isinstance(v1[4], Vector3))
        self.assertTrue(isinstance(v1[4, :], np.ndarray))
        self.assertTrue(isinstance(v1.x, np.ndarray))
        self.assertTrue(isinstance(v1[1:30, :], np.ndarray))

        a1 = np.array([1., 2.])
        with self.assertRaises(ValueError):
            a1.view(Vector3)
        with self.assertRaises(ValueError):
            a1.view(Vector3Array)
        a1 = np.array([1., 2., 3.])
        self.assertTrue(isinstance(a1.view(Vector3), Vector3))
        with self.assertRaises(ValueError):
            a1.view(Vector3Array)
        a1 = np.array([[1., 2., 3.]])
        with self.assertRaises(ValueError):
            a1.view(Vector3)
        self.assertTrue(isinstance(a1.view(Vector3Array), Vector3Array))

        self.assertTrue(isinstance(v1.view(Vector3Array), Vector3Array))
        with self.assertRaises(ValueError):
            v1.view(Vector2Array)
        with self.assertRaises(ValueError):
            v1.view(Vector3)
        with self.assertRaises(ValueError):
            v1.view(Vector2)
        v1 = Vector3([1., 2., 3.])
        with self.assertRaises(ValueError):
            v1.view(Vector3Array)
        with self.assertRaises(ValueError):
            v1.view(Vector2Array)
        with self.assertRaises(ValueError):
            v1.view(Vector2)
        self.assertTrue(isinstance(v1.view(Vector3), Vector3))

        v1 = np.kron(Vector3([1., 0., 0.]), np.atleast_2d(np.ones(10)).T)
        self.assertFalse(isinstance(v1, Vector3))

    def test_angle(self):

        # test a unit vector along each coordinate
        v1 = Vector3(1, 0, 0)  # x-axis, use this as datum
        v = [Vector3(1, 0, 0), Vector3(0, 1, 0), Vector3(0, 0, 1),
             Vector3(-1, 0, 0), Vector3(0, -1, 0), Vector3(0, 0, -1)]
        angles_deg = [0, 90, 90, 180, 90, 90]
        angles_rad = [0, np.pi / 2, np.pi / 2, np.pi, np.pi / 2, np.pi / 2]
        for k in range(6):
            a_deg = v1.angle(v[k], unit='deg')
            a_rad0 = v1.angle(v[k], unit='rad')
            a_rad1 = v1.angle(v[k])
            self.assertEqual(a_deg, angles_deg[k])
            self.assertEqual(a_rad0, angles_rad[k])
            self.assertEqual(a_rad1, angles_rad[k])

            # verify the associative property
            self.assertEqual(v1.angle(v[k]), v[k].angle(v1))

        with self.assertRaises(TypeError):
            angleResult = v1.angle('anything but Vector3')
        with self.assertRaises(ValueError):
            angleResult = v1.angle(v[0], unit='invalid entry')
        with self.assertRaises(ZeroDivisionError):
            angleResult = v1.angle(Vector3(0, 0, 0))

    # def test_mult_warning(self):
    #     with warnings.catch_warnings(record=True) as w:
    #         v1 = Vector3Array()
    #         v2 = v1 * 3
    #         self.assertTrue(len(w) == 0)
    #         M = Matrix3()
    #         v3 = v2 * M
    #         self.assertTrue(len(w) == 1)


if __name__ == '__main__':
    unittest.main()
