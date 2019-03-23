from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import unittest
import numpy as np

from vectormath import Vector2, Vector2Array, Vector3, Vector3Array


class TestVMathVector2(unittest.TestCase):

    def test_init_exceptions(self):
        self.assertRaises(TypeError, Vector2Array, np.r_[1], 1.0)
        self.assertRaises(ValueError, Vector2Array, np.r_[1, 2], np.r_[1])
        self.assertRaises(ValueError, Vector2Array, np.array([0, 0, 0]))
        self.assertRaises(ValueError, Vector2Array, 'Make', ' me a ')
        self.assertRaises(ValueError, Vector2Array, ([0, 0, 0], [0, 0, 0]))

    def test_init(self):
        v1 = Vector2Array()
        v2 = Vector2Array(0, 0)
        self.assertTrue(np.array_equal(v1, v2))
        v3 = Vector2Array(v1)
        self.assertTrue(np.array_equal(v1, v3))
        self.assertTrue(v1 is not v3)
        v4 = Vector2Array(np.r_[0, 0])
        self.assertTrue(np.array_equal(v1, v4))
        v5 = Vector2Array(np.c_[np.r_[1, 0], np.r_[0, 1]])
        self.assertTrue(np.array_equal(v5.length, np.r_[1, 1]))
        v6 = Vector2Array(np.r_[1, 0], np.r_[0, 1])
        self.assertTrue(np.array_equal(v6.length, np.r_[1, 1]))
        v7 = Vector2Array([0, 0])
        self.assertTrue(np.array_equal(v1, v7))
        v8 = Vector2Array(x=0, y=0)
        self.assertTrue(np.array_equal(v1, v8))
        v9 = Vector2Array(
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0]
        )
        v10 = Vector2Array([
            [0, 0],
            [0, 0],
            [0, 0],
            [0, 0],
            [0, 0]
        ])
        self.assertTrue(np.array_equal(v9, v10))
        v11 = Vector2Array([[[[[0]], [[0]], ]]])
        self.assertTrue(np.array_equal(v1, v11))
        v12 = Vector2Array([0]*5, [0]*5)
        self.assertTrue(np.array_equal(v10, v12))
        v13 = Vector2Array((0, 0))
        self.assertTrue(np.array_equal(v1, v13))
        v14 = Vector2Array(([0, 0], [0, 0]))
        self.assertTrue(np.array_equal(v14, Vector2Array([0]*2, [0]*2)))

    def test_indexing(self):
        v2 = Vector2Array(1, 2)
        self.assertTrue(v2[0, 0] == 1)
        self.assertTrue(v2[0, 1] == 2)
        self.assertTrue(len(v2[0]) == 2)

        def f(): v2[3]
        self.assertRaises(IndexError, f)

        def f(): v2[0, 3]
        self.assertRaises(IndexError, f)
        l = []
        for x in v2[0]:
            l.append(x)
        self.assertTrue(np.array_equal(np.array(l), np.r_[1, 2]))
        self.assertTrue(np.array_equal(v2, Vector2Array(l)))
        l = []
        v3 = Vector2Array([[1, 2],
                     [2, 3]])
        for v in v3:
            l.append(v)
        self.assertTrue(np.array_equal(
            np.array(l),
            np.array([[1, 2], [2, 3]]))
        )
        self.assertTrue(np.array_equal(Vector2Array(l), v3))
        v4 = Vector2Array()
        v4[0, 0] = 1
        v4[0, 1] = 2
        self.assertTrue(np.array_equal(v2, v4))

    def test_copy(self):
        vOrig = Vector2Array()
        vCopy = vOrig.copy()
        self.assertTrue(np.array_equal(vOrig, vCopy))
        self.assertTrue(vOrig is not vCopy)

    def test_size(self):
        v1 = Vector2Array()
        self.assertTrue(v1.nV == 1)
        v2 = Vector2Array(np.c_[np.r_[1, 0, 0], np.r_[0, 1, 0]])
        self.assertTrue(v2.nV == 3)
        v3 = Vector2Array(
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0]
        )
        self.assertTrue(v3.nV == 5)
        v4 = Vector2Array(0, 0)
        self.assertTrue(v4.nV == 1)

    def test_setget(self):
        v1 = Vector2Array(1, 1)
        self.assertTrue(v1.x == 1)
        v1.x = 2
        self.assertTrue(v1.x == 2)
        self.assertTrue(v1.y == 1)
        v1.y = 2
        self.assertTrue(v1.y == 2)

        v2 = Vector2Array([[0, 1],
                     [1, 2]])
        self.assertTrue(np.array_equal(v2.x, [0, 1]))
        v2.x = [0, -1]
        self.assertTrue(np.array_equal(v2.x, [0, -1]))
        self.assertTrue(np.array_equal(v2.y, [1, 2]))
        v2.y = [-1, -2]
        self.assertTrue(np.array_equal(v2.y, [-1, -2]))

    def test_length(self):
        v1 = Vector2Array(1, 1)
        self.assertTrue(v1.length == np.sqrt(2))
        v2 = Vector2Array(np.r_[1, 2], np.r_[1, 2])
        self.assertTrue(np.array_equal(v2.length, np.sqrt(np.r_[2, 8])))
        v3 = Vector2Array(1, 0)
        v3.length = 5
        assert v3.x == 5
        self.assertTrue(v3.length == 5)
        v4 = Vector2Array(np.r_[1, 1], np.r_[0, 0])

        def f(): v4.length = 5
        self.assertRaises(ValueError, f)
        v5 = Vector2Array(np.r_[1, 0], np.r_[0, 1])
        self.assertTrue(np.array_equal(v5.length, [1, 1]))
        v5.length = [-1, 3]
        self.assertTrue(np.array_equal(v5, [[-1., -0.], [0., 3.]]))
        self.assertTrue(np.array_equal(v5.length, [1, 3]))
        v6 = Vector2Array()
        self.assertTrue(v6.length == 0)

        def f(): v6.length = 5
        self.assertRaises(ZeroDivisionError, f)
        v6.length = 0
        self.assertTrue(v6.length == 0)
        v7 = Vector2Array(
            [0, 0, 1, 0, 0],
            [1, 1, 0, 0, 0]
        )
        length = [5, 5, 5, 5, 5]

        def f(): v7.length = length
        self.assertRaises(ZeroDivisionError, f)
        length = [5, 5, 5, 0, 0]
        v7.length = length
        self.assertTrue(np.array_equal(length, v7.length))

    def test_ops(self):
        v1 = Vector2Array(1, 1)
        v2 = Vector2Array(2, 2)
        self.assertTrue(np.array_equal(v2-v1, v1))
        self.assertTrue(np.array_equal(v1-v2, -v1))
        self.assertTrue(np.array_equal(v1+v1, v2))
        self.assertTrue(np.array_equal(v1*v2, v2))
        self.assertTrue(np.array_equal(v2/v1, v2))
        self.assertTrue(np.array_equal(2*v1, v2))
        self.assertTrue(np.array_equal(v2/2, v1))
        self.assertTrue(np.array_equal(v1+1, v2))
        self.assertTrue(np.array_equal(v2-1, v1))
        v1 = Vector2Array(np.r_[1, 1.], np.r_[1, 1.])
        v2 = Vector2Array(np.r_[2, 2.], np.r_[2, 2.])
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
        v1 = Vector2Array(1, 1)
        v2 = Vector2Array(2, 2)
        self.assertTrue(v1.dot(v2) == 4)
        v1l = Vector2Array(np.r_[1, 1.], np.r_[1, 1.])
        v2l = Vector2Array(np.r_[2, 2.], np.r_[2, 2.])
        self.assertTrue(np.array_equal(v1l.dot(v2l), np.r_[4, 4]))
        self.assertTrue(np.array_equal(v1.dot(v2l), np.r_[4, 4]))
        self.assertTrue(np.array_equal(v1l.dot(v2), np.r_[4, 4]))
        v3 = Vector2Array([3]*4, [3]*4)

        def f(): v3.dot(v2l)
        self.assertRaises(ValueError, f)

        def f(): v3.dot(5)
        self.assertRaises(TypeError, f)

    def test_as_percent(self):
        v1 = Vector2Array(10, 0)
        v2 = Vector2Array(20, 0)
        self.assertTrue(np.array_equal(v1.as_percent(2), v2))
        self.assertTrue(np.array_equal(v1, Vector2Array(10, 0)))   # not copied
        v3 = Vector2Array(
            [0, 0, 2, 0, 0],
            [0, 2, 0, 0, 0]
        )
        v4 = v3 * .5
        self.assertTrue(np.array_equal(v3.as_percent(.5), v4))
        v5 = Vector2Array()
        self.assertTrue(np.array_equal(v5.as_percent(100), v5))
        v6 = Vector2Array(5, 5)
        self.assertTrue(np.array_equal(v6.as_percent(0), v5))

        def f(): v6.as_percent('One Hundred Percent')
        self.assertRaises(TypeError, f)

    def test_normalize(self):
        v1 = Vector2Array(5, 0)
        self.assertTrue(v1.length == 5)
        self.assertTrue(v1.normalize() is v1)
        self.assertTrue(v1.length == 1)
        v2 = Vector2Array()

        def f(): v2.normalize()
        self.assertRaises(ZeroDivisionError, f)
        v3 = Vector2Array(
            [0, 2],
            [2, 0]
        )
        self.assertTrue(np.array_equal(v3.length, [2, 2]))
        self.assertTrue(v3.normalize() is v3)
        self.assertTrue(np.array_equal(v3.length, [1, 1]))

    def test_as_length(self):
        v1 = Vector2Array(1, 1)
        v2 = v1.as_length(1)
        self.assertTrue(v1 is not v2)
        self.assertTrue(v1.length == np.sqrt(2))
        self.assertAlmostEqual(v2.length[0], 1)

        v3 = Vector2Array(np.r_[1, 2], np.r_[1, 2])
        v4 = v3.as_length([1, 2])
        self.assertTrue(np.allclose(v4.length, [1, 2]))

        def f(): v = v3.as_length(5)
        self.assertRaises(ValueError, f)
        v5 = Vector2Array(np.r_[1, 0], np.r_[0, 1])
        self.assertTrue(np.allclose(v5.length, [1, 1]))
        v6 = v5.as_length([-1, 3])
        self.assertTrue(np.allclose(v6, [[-1., -0.], [0., 3.]]))
        self.assertTrue(np.allclose(v6.length, [1, 3]))
        v7 = Vector2Array()

        def f(): v = v7.as_length(5)
        self.assertRaises(ZeroDivisionError, f)
        v8 = v7.as_length(0)
        self.assertTrue(v8.length == 0)
        v9 = Vector2Array(
            [0, 0, 1, 0, 0],
            [1, 1, 0, 0, 0]
        )
        length = [5, 5, 5, 5, 5]

        def f(): v = v9.as_length(length)
        self.assertRaises(ZeroDivisionError, f)
        length = [5, 5, 5, 0, 0]
        v10 = v9.as_length(length)
        self.assertTrue(np.array_equal(length, v10.length))

    def test_as_unit(self):
        v1 = Vector2Array(1, 0)
        v2 = v1.as_unit()
        self.assertTrue(v1 is not v2)
        self.assertTrue(np.array_equal(v1, v2))
        self.assertTrue(v2.length == 1)
        v3 = Vector2Array(np.r_[1, 2], np.r_[1, 2])
        v4 = v3.as_unit()
        self.assertTrue(np.allclose(v4.length, [1, 1]))
        v5 = Vector2Array(1, 1)
        v6 = v5.as_unit()
        self.assertAlmostEqual(v6.length[0], 1)
        self.assertTrue(v6.x == v6.y)
        v7 = Vector2Array()

        def f(): v = v7.as_unit()
        self.assertRaises(ZeroDivisionError, f)
        v9 = Vector2Array(
            [0, 0, 1, 0, 0],
            [0, 1, 0, 0, 0]
        )

        def f(): v = v9.as_unit()
        self.assertRaises(ZeroDivisionError, f)

    def test_view_types(self):
        v1 = Vector2Array(np.random.rand(100, 2))
        self.assertTrue(isinstance(v1, Vector2Array))
        self.assertTrue(isinstance(v1[1:2], Vector2Array))
        self.assertTrue(isinstance(v1[1:50:2], Vector2Array))
        self.assertTrue(isinstance(v1[4], Vector2))
        self.assertTrue(isinstance(v1[4, :], np.ndarray))
        self.assertTrue(isinstance(v1.x, np.ndarray))
        self.assertTrue(isinstance(v1[1:30, :], np.ndarray))

        a1 = np.array([1., 2., 3])
        with self.assertRaises(ValueError):
            a1.view(Vector2)
        with self.assertRaises(ValueError):
            a1.view(Vector2Array)
        a1 = np.array([1., 2.])
        self.assertTrue(isinstance(a1.view(Vector2), Vector2))
        with self.assertRaises(ValueError):
            a1.view(Vector2Array)
        a1 = np.array([[1., 2.]])
        with self.assertRaises(ValueError):
            a1.view(Vector2)
        self.assertTrue(isinstance(a1.view(Vector2Array), Vector2Array))

        with self.assertRaises(ValueError):
            v1.view(Vector3Array)
        self.assertTrue(isinstance(v1.view(Vector2Array), Vector2Array))
        with self.assertRaises(ValueError):
            v1.view(Vector3)
        with self.assertRaises(ValueError):
            v1.view(Vector2)
        v1 = Vector2([1., 2.])
        with self.assertRaises(ValueError):
            v1.view(Vector3Array)
        with self.assertRaises(ValueError):
            v1.view(Vector2Array)
        with self.assertRaises(ValueError):
            v1.view(Vector3)
        self.assertTrue(isinstance(v1.view(Vector2), Vector2))

        v1 = np.kron(Vector2([1., 0.]), np.atleast_2d(np.ones(10)).T)
        self.assertFalse(isinstance(v1, Vector2))

    def test_cross(self):
        vector2 = Vector2(5, 0)
        vector2_2 = Vector2(1, 0)
        crossResult = vector2.cross(vector2_2)
        self.assertEqual(crossResult[0], 0)
        self.assertEqual(crossResult[1], 0)
        self.assertEqual(crossResult[2], 0)
        with self.assertRaises(TypeError):
            dotResult2 = vector2.cross("Banana")

    def test_angle(self):

        # test a unit vector along each coordinate
        v1 = Vector2(1, 0)  # x-coordinate, use this as datum
        v = [Vector2(1, 0), Vector2(0, 1), Vector2(-1, 0), Vector2(0, -1)]
        angles_deg = [0, 90, 180, 90]
        angles_rad = [0, np.pi / 2, np.pi, np.pi / 2]
        for k in range(4):
            a_deg = v1.angle(v[k], unit='deg')
            a_rad0 = v1.angle(v[k], unit='rad')
            a_rad1 = v1.angle(v[k])
            self.assertEqual(a_deg, angles_deg[k])
            self.assertEqual(a_rad0, angles_rad[k])
            self.assertEqual(a_rad1, angles_rad[k])

            # verify the associative property
            self.assertEqual(v1.angle(v[k]), v[k].angle(v1))

        with self.assertRaises(TypeError):
            angleResult = v1.angle('anything but Vector2')
        with self.assertRaises(ValueError):
            angleResult = v1.angle(v[0], unit='invalid entry')
        with self.assertRaises(ZeroDivisionError):
            angleResult = v1.angle(Vector2(0, 0))

    def test_polar(self):
        # polar <-> cartesian conversions
        cases = [
            # ((rho, theta), (x, y))
            ((1, 0), (1, 0)),
            ((1, np.pi), (-1, 0)),
            ((2, -np.pi / 2), (0, -2)),
            ((1, np.pi * 3 / 4), (-1 / np.sqrt(2), 1 / np.sqrt(2))),
            ((3, np.pi / 4), (3 / np.sqrt(2), 3 / np.sqrt(2))),
        ]
        for polar, cartesian in cases:
            rho, theta = polar
            x, y = cartesian
            v = Vector2(rho, theta, polar=True)
            self.assertAlmostEqual(v.x, x)
            self.assertAlmostEqual(v.y, y)
            v = Vector2(x, y)
            self.assertAlmostEqual(v.rho, rho)
            self.assertAlmostEqual(v.theta, theta)

        # degrees -> radians
        cases = [
            # (degrees, radians)
            (0, 0),
            (90, np.pi/2),
            (-90, -np.pi/2),
            (45, np.pi/4),
            (180, np.pi),
        ]
        for deg, rad in cases:
            v = Vector2(1, deg, polar=True, unit='deg')
            self.assertAlmostEqual(v.theta, rad)
            self.assertAlmostEqual(v.theta_deg, deg)

        # faulty input
        with self.assertRaises(ValueError):
            Vector2(1, np.pi, polar=True, unit='invalid_unit')
        with self.assertRaises(ValueError):
            v = Vector2(1, np.pi, polar=True)
            # copying doesn't support polar=True
            Vector2(v, polar=True)

    def test_spherical(self):
        # cartesian -> sperical conversions
        cases = [
            # ((x, y, z), (rho, theta, phi))
            ((1, 0, 0), (1, 0, np.pi/2)),
            ((1, 0, 1), (np.sqrt(2), 0, np.pi/4)),
            ((1, 0, -1), (np.sqrt(2), 0, np.pi/4*3)),
        ]
        for cartesian, sperical in cases:
            rho, theta, phi = sperical
            x, y, z = cartesian
            v = Vector3(x, y, z)
            self.assertAlmostEqual(v.rho, rho)
            self.assertAlmostEqual(v.theta, theta)
            self.assertAlmostEqual(v.phi, phi)


if __name__ == '__main__':
    unittest.main()
