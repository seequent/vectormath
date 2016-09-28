from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import unittest
import numpy as np

from vectormath import Vector2


class TestVMathVector2(unittest.TestCase):

    def test_init_exceptions(self):
        self.assertRaises(TypeError, Vector2, np.r_[1], 1.0)
        self.assertRaises(ValueError, Vector2, np.r_[1, 2], np.r_[1])
        self.assertRaises(ValueError, Vector2, np.array([0, 0, 0]))
        self.assertRaises(ValueError, Vector2, 'Make', ' me a ')
        self.assertRaises(ValueError, Vector2, ([0, 0, 0], [0, 0, 0]))

    def test_init(self):
        v1 = Vector2()
        v2 = Vector2(0, 0)
        self.assertTrue(np.array_equal(v1, v2))
        v3 = Vector2(v1)
        self.assertTrue(np.array_equal(v1, v3))
        self.assertTrue(v1 is not v3)
        v4 = Vector2(np.r_[0, 0])
        self.assertTrue(np.array_equal(v1, v4))
        v5 = Vector2(np.c_[np.r_[1, 0], np.r_[0, 1]])
        self.assertTrue(np.array_equal(v5.length, np.r_[1, 1]))
        v6 = Vector2(np.r_[1, 0], np.r_[0, 1])
        self.assertTrue(np.array_equal(v6.length, np.r_[1, 1]))
        v7 = Vector2([0, 0])
        self.assertTrue(np.array_equal(v1, v7))
        v8 = Vector2(x=0, y=0)
        self.assertTrue(np.array_equal(v1, v8))
        v9 = Vector2(
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0]
        )
        v10 = Vector2([
            [0, 0],
            [0, 0],
            [0, 0],
            [0, 0],
            [0, 0]
        ])
        self.assertTrue(np.array_equal(v9, v10))
        v11 = Vector2([[[[[0]], [[0]], ]]])
        self.assertTrue(np.array_equal(v1, v11))
        v12 = Vector2([0]*5, [0]*5)
        self.assertTrue(np.array_equal(v10, v12))
        v13 = Vector2((0, 0))
        self.assertTrue(np.array_equal(v1, v13))
        v14 = Vector2(([0, 0], [0, 0]))
        self.assertTrue(np.array_equal(v14, Vector2([0]*2, [0]*2)))

    def test_indexing(self):
        v2 = Vector2(1, 2)
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
        self.assertTrue(np.array_equal(v2, Vector2(l)))
        l = []
        v3 = Vector2([[1, 2],
                     [2, 3]])
        for v in v3:
            l.append(v)
        self.assertTrue(np.array_equal(
            np.array(l),
            np.array([[1, 2], [2, 3]]))
        )
        self.assertTrue(np.array_equal(Vector2(l), v3))
        v4 = Vector2()
        v4[0, 0] = 1
        v4[0, 1] = 2
        self.assertTrue(np.array_equal(v2, v4))

    def test_copy(self):
        vOrig = Vector2()
        vCopy = vOrig.copy()
        self.assertTrue(np.array_equal(vOrig, vCopy))
        self.assertTrue(vOrig is not vCopy)

    def test_size(self):
        v1 = Vector2()
        self.assertTrue(v1.nV == 1)
        v2 = Vector2(np.c_[np.r_[1, 0, 0], np.r_[0, 1, 0]])
        self.assertTrue(v2.nV == 3)
        v3 = Vector2(
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0]
        )
        self.assertTrue(v3.nV == 5)
        v4 = Vector2(0, 0)
        self.assertTrue(v4.nV == 1)

    def test_setget(self):
        v1 = Vector2(1, 1)
        self.assertTrue(v1.x == 1)
        v1.x = 2
        self.assertTrue(v1.x == 2)
        self.assertTrue(v1.y == 1)
        v1.y = 2
        self.assertTrue(v1.y == 2)

        v2 = Vector2([[0, 1],
                     [1, 2]])
        self.assertTrue(np.array_equal(v2.x, [0, 1]))
        v2.x = [0, -1]
        self.assertTrue(np.array_equal(v2.x, [0, -1]))
        self.assertTrue(np.array_equal(v2.y, [1, 2]))
        v2.y = [-1, -2]
        self.assertTrue(np.array_equal(v2.y, [-1, -2]))

    def test_length(self):
        v1 = Vector2(1, 1)
        self.assertTrue(v1.length == np.sqrt(2))
        v2 = Vector2(np.r_[1, 2], np.r_[1, 2])
        self.assertTrue(np.array_equal(v2.length, np.sqrt(np.r_[2, 8])))
        v3 = Vector2(1, 0)
        v3.length = 5
        assert v3.x == 5
        self.assertTrue(v3.length == 5)
        v4 = Vector2(np.r_[1, 1], np.r_[0, 0])

        def f(): v4.length = 5
        self.assertRaises(ValueError, f)
        v5 = Vector2(np.r_[1, 0], np.r_[0, 1])
        self.assertTrue(np.array_equal(v5.length, [1, 1]))
        v5.length = [-1, 3]
        self.assertTrue(np.array_equal(v5, [[-1., -0.], [0., 3.]]))
        self.assertTrue(np.array_equal(v5.length, [1, 3]))
        v6 = Vector2()
        self.assertTrue(v6.length == 0)

        def f(): v6.length = 5
        self.assertRaises(ZeroDivisionError, f)
        v6.length = 0
        self.assertTrue(v6.length == 0)
        v7 = Vector2(
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
        v1 = Vector2(1, 1)
        v2 = Vector2(2, 2)
        self.assertTrue(np.array_equal(v2-v1, v1))
        self.assertTrue(np.array_equal(v1-v2, -v1))
        self.assertTrue(np.array_equal(v1+v1, v2))
        self.assertTrue(np.array_equal(v1*v2, v2))
        self.assertTrue(np.array_equal(v2/v1, v2))
        self.assertTrue(np.array_equal(2*v1, v2))
        self.assertTrue(np.array_equal(v2/2, v1))
        self.assertTrue(np.array_equal(v1+1, v2))
        self.assertTrue(np.array_equal(v2-1, v1))
        v1 = Vector2(np.r_[1, 1.], np.r_[1, 1.])
        v2 = Vector2(np.r_[2, 2.], np.r_[2, 2.])
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
        v1 = Vector2(1, 1)
        v2 = Vector2(2, 2)
        self.assertTrue(v1.dot(v2) == 4)
        v1l = Vector2(np.r_[1, 1.], np.r_[1, 1.])
        v2l = Vector2(np.r_[2, 2.], np.r_[2, 2.])
        self.assertTrue(np.array_equal(v1l.dot(v2l), np.r_[4, 4]))
        self.assertTrue(np.array_equal(v1.dot(v2l), np.r_[4, 4]))
        self.assertTrue(np.array_equal(v1l.dot(v2), np.r_[4, 4]))
        v3 = Vector2([3]*4, [3]*4)

        def f(): v3.dot(v2l)
        self.assertRaises(ValueError, f)

        def f(): v3.dot(5)
        self.assertRaises(TypeError, f)

    def test_as_percent(self):
        v1 = Vector2(10, 0)
        v2 = Vector2(20, 0)
        self.assertTrue(np.array_equal(v1.as_percent(2), v2))
        self.assertTrue(np.array_equal(v1, Vector2(10, 0)))   # not copied
        v3 = Vector2(
            [0, 0, 2, 0, 0],
            [0, 2, 0, 0, 0]
        )
        v4 = v3 * .5
        self.assertTrue(np.array_equal(v3.as_percent(.5), v4))
        v5 = Vector2()
        self.assertTrue(np.array_equal(v5.as_percent(100), v5))
        v6 = Vector2(5, 5)
        self.assertTrue(np.array_equal(v6.as_percent(0), v5))

        def f(): v6.as_percent('One Hundred Percent')
        self.assertRaises(TypeError, f)

    def test_normalize(self):
        v1 = Vector2(5, 0)
        self.assertTrue(v1.length == 5)
        self.assertTrue(v1.normalize() is v1)
        self.assertTrue(v1.length == 1)
        v2 = Vector2()

        def f(): v2.normalize()
        self.assertRaises(ZeroDivisionError, f)
        v3 = Vector2(
            [0, 2],
            [2, 0]
        )
        self.assertTrue(np.array_equal(v3.length, [2, 2]))
        self.assertTrue(v3.normalize() is v3)
        self.assertTrue(np.array_equal(v3.length, [1, 1]))

    def test_as_length(self):
        v1 = Vector2(1, 1)
        v2 = v1.as_length(1)
        self.assertTrue(v1 is not v2)
        self.assertTrue(v1.length == np.sqrt(2))
        self.assertAlmostEqual(v2.length, 1)

        v3 = Vector2(np.r_[1, 2], np.r_[1, 2])
        v4 = v3.as_length([1, 2])
        self.assertTrue(np.allclose(v4.length, [1, 2]))

        def f(): v = v3.as_length(5)
        self.assertRaises(ValueError, f)
        v5 = Vector2(np.r_[1, 0], np.r_[0, 1])
        self.assertTrue(np.allclose(v5.length, [1, 1]))
        v6 = v5.as_length([-1, 3])
        self.assertTrue(np.allclose(v6, [[-1., -0.], [0., 3.]]))
        self.assertTrue(np.allclose(v6.length, [1, 3]))
        v7 = Vector2()

        def f(): v = v7.as_length(5)
        self.assertRaises(ZeroDivisionError, f)
        v8 = v7.as_length(0)
        self.assertTrue(v8.length == 0)
        v9 = Vector2(
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
        v1 = Vector2(1, 0)
        v2 = v1.as_unit()
        self.assertTrue(v1 is not v2)
        self.assertTrue(np.array_equal(v1, v2))
        self.assertTrue(v2.length == 1)
        v3 = Vector2(np.r_[1, 2], np.r_[1, 2])
        v4 = v3.as_unit()
        self.assertTrue(np.allclose(v4.length, [1, 1]))
        v5 = Vector2(1, 1)
        v6 = v5.as_unit()
        self.assertAlmostEqual(v6.length, 1)
        self.assertTrue(v6.x == v6.y)
        v7 = Vector2()

        def f(): v = v7.as_unit()
        self.assertRaises(ZeroDivisionError, f)
        v9 = Vector2(
            [0, 0, 1, 0, 0],
            [0, 1, 0, 0, 0]
        )

        def f(): v = v9.as_unit()
        self.assertRaises(ZeroDivisionError, f)


if __name__ == '__main__':
    unittest.main()
