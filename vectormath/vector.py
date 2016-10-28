from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import numpy as np


class Vector3(np.ndarray):
    """Primitive 3D vector defined from the origin"""

    def __new__(cls, x=None, y=None, z=None):

        def read_array(X, Y, Z):
            if isinstance(X, cls) and Y is None and Z is None:
                return cls(X.x, X.y, X.z)
            if (isinstance(X, (list, tuple, np.ndarray)) and len(X) == 3 and
                    Y is None and Z is None):
                return cls(X[0], X[1], X[2])
            if X is None and Y is None and Z is None:
                return cls(0, 0, 0)
            if np.isscalar(X) and np.isscalar(Y) and np.isscalar(Z):
                xyz = np.r_[X, Y, Z]
                xyz = xyz.astype(float)
                return xyz.view(cls)
            raise ValueError('Invalid input for Vector3 - must be an instance '
                             'of a Vector3, a length-3 array, 3 scalars, or '
                             'nothing for [0., 0., 0.]')

        return read_array(x, y, z)

    def __array_finalize__(self, obj):
        if obj is None:
            return

    @property
    def x(self):
        return self[0]

    @x.setter
    def x(self, value):
        self[0] = value

    @property
    def y(self):
        return self[1]

    @y.setter
    def y(self, value):
        self[1] = value

    @property
    def z(self):
        return self[2]

    @z.setter
    def z(self, value):
        self[2] = value

    @property
    def length(self):
        """Vector3 length"""
        return float(np.sqrt(np.sum(self**2)))

    @length.setter
    def length(self, l):
        if not np.isscalar(l):
            raise ValueError('Length must be a scalar')
        l = float(l)
        if self.length != 0:
            new_length = l/self.length
            self.x *= new_length
            self.y *= new_length
            self.z *= new_length
            return
        if l != 0:
            raise ZeroDivisionError('Cannot resize vector of length 0 to '
                                    'nonzero length')

    def as_length(self, l):
        """Scale the length of a vector to a value"""
        V = self.copy()
        V.length = l
        return V

    def as_percent(self, p):
        """Scale the length of a vector by a percent"""
        V = self.copy()
        V.length = p * self.length
        return V

    def as_unit(self):
        """Scale the length of a vector to 1"""
        V = self.copy()
        V.normalize()
        return V

    def normalize(self):
        """Scale the length of a vector to 1 in place"""
        self.length = 1
        return self

    def dot(self, vec):
        """Dot product with another vector"""
        if not isinstance(vec, Vector3):
            raise TypeError('Dot product operand must be a vector')
        return float(self.x*vec.x + self.y*vec.y + self.z*vec.z)

    def cross(self, vec):
        """Cross product with another vector"""
        if not isinstance(vec, Vector3):
            raise TypeError('Cross product operand must be a vector')
        return Vector3(np.cross(self, vec))

    def __mul__(self, m):
        return Vector3(self.view(np.ndarray) * m)


class Vector2(np.ndarray):
    """Primitive 2D vector defined from the origin"""

    def __new__(cls, x=None, y=None):

        def read_array(X, Y):
            if isinstance(X, cls) and Y is None:
                return cls(X.x, X.y)
            if (isinstance(X, (list, tuple, np.ndarray)) and len(X) == 2 and
                    Y is None):
                return cls(X[0], X[1])
            if X is None and Y is None:
                return cls(0, 0)
            if np.isscalar(X) and np.isscalar(Y):
                xyz = np.r_[X, Y]
                xyz = xyz.astype(float)
                return xyz.view(cls)
            raise ValueError('Invalid input for Vector2 - must be an instance '
                             'of a Vector2, a length-2 array, 2 scalars, or '
                             'nothing for [0., 0.]')

        return read_array(x, y)

    def __array_finalize__(self, obj):
        if obj is None:
            return

    @property
    def x(self):
        return self[0]

    @x.setter
    def x(self, value):
        self[0] = value

    @property
    def y(self):
        return self[1]

    @y.setter
    def y(self, value):
        self[1] = value

    @property
    def length(self):
        """Vector3 length"""
        return float(np.sqrt(np.sum(self**2)))

    @length.setter
    def length(self, l):
        if not np.isscalar(l):
            raise ValueError('Length must be a scalar')
        l = float(l)
        if self.length != 0:
            new_length = l/self.length
            self.x *= new_length
            self.y *= new_length
            return
        if l != 0:
            raise ZeroDivisionError('Cannot resize vector of length 0 to '
                                    'nonzero length')

    def as_length(self, l):
        """Scale the length of a vector to a value"""
        V = self.copy()
        V.length = l
        return V

    def as_percent(self, p):
        """Scale the length of a vector by a percent"""
        V = self.copy()
        V.length = p * self.length
        return V

    def as_unit(self):
        """Scale the length of a vector to 1"""
        V = self.copy()
        V.normalize()
        return V

    def normalize(self):
        """Scale the length of a vector to 1 in place"""
        self.length = 1
        return self

    def dot(self, vec):
        """Dot product with another vector"""
        if not isinstance(vec, Vector2):
            raise TypeError('Dot product operand must be a vector')
        return float(self.x*vec.x + self.y*vec.y)

    def cross(self, vec):
        """Cross product with another vector"""
        if not isinstance(vec, Vector2):
            raise TypeError('Cross product operand must be a vector')
        return Vector2(np.cross(self, vec))

    def __mul__(self, m):
        return Vector2(self.view(np.ndarray) * m)


class Vector3Array(np.ndarray):
    """
        Primitive vectors, or list of primitive vectors,
        defined from the origin.
    """

    def __new__(cls, x=None, y=None, z=None):

        def read_array(X, Y, Z):
            if isinstance(X, cls) and Y is None and Z is None:
                X = np.atleast_2d(X)
                return cls(X.x.copy(), X.y.copy(), X.z.copy())
            if isinstance(X, (list, tuple)):
                X = np.array(X)
            if isinstance(Y, (list, tuple)):
                Y = np.array(Y)
            if isinstance(Z, (list, tuple)):
                Z = np.array(Z)
            if isinstance(X, np.ndarray) and Y is None and Z is None:
                X = np.squeeze(X)
                if X.size == 3:
                    X = X.flatten()
                    return cls(X[0], X[1], X[2])
                elif len(X.shape) == 2 and X.shape[1] == 3:
                    return cls(
                        X[:, 0].copy(), X[:, 1].copy(), X[:, 2].copy()
                    )
                raise ValueError(
                    'Unexpected shape for vector init: {shp}'.format(
                        shp=X.shape
                    )
                )
            if np.isscalar(X) and np.isscalar(Y) and np.isscalar(Z):
                X, Y, Z = float(X), float(Y), float(Z)
            elif not (isinstance(X, type(Y)) and isinstance(X, type(Z))):
                raise TypeError('Must be the same types for x, y, and '
                                'z for vector init')
            if isinstance(X, np.ndarray):
                if not (X.shape == Y.shape and X.shape == Z.shape):
                    raise ValueError('Must be the same shapes for x, y, '
                                     'and z in vector init')
                xyz = np.c_[X, Y, Z]
                xyz = xyz.astype(float)
                return xyz.view(cls)
            if X is None:
                X, Y, Z = 0.0, 0.0, 0.0
            xyz = np.r_[X, Y, Z].reshape((1, 3))
            return np.asarray(xyz).view(cls)

        return read_array(x, y, z)

    def __array_finalize__(self, obj):
        if obj is None:
            return

    def __getitem__(self, i):
        item_out = super(Vector3Array, self).__getitem__(i)
        if np.isscalar(i):
            return item_out.view(Vector3)
        if isinstance(i, slice):
            return item_out
        return item_out.view(np.ndarray)

    @property
    def x(self):
        return self[:, 0]

    @x.setter
    def x(self, value):
        self[:, 0] = value

    @property
    def y(self):
        return self[:, 1]

    @y.setter
    def y(self, value):
        self[:, 1] = value

    @property
    def z(self):
        return self[:, 2]

    @z.setter
    def z(self, value):
        self[:, 2] = value

    @property
    def nV(self):
        """Number of vectors"""
        return self.shape[0]

    @property
    def length(self):
        """Vector3 lengths"""
        return np.sqrt(np.sum(self**2, axis=1)).view(np.ndarray)

    @length.setter
    def length(self, l):
        l = np.array(l)
        if self.nV != l.size:
            raise ValueError('Length vector must be the same number of '
                             'elements as vector.')
        # This case resizes all vectors with nonzero length
        if np.all(self.length != 0):
            new_length = l/self.length
            self.x *= new_length
            self.y *= new_length
            self.z *= new_length
            return
        # This case only applies to single vectors
        # if self.length == 0 and l == 0
        if self.nV == 1 and l == 0:
            assert self.length == 0, \
                'Nonzero length should be resized in the first case'
            self.x, self.y, self.z = 0, 0, 0
            return
        # This case only applies if vectors with length == 0
        # in an array are getting resized to 0
        if self.nV > 1 and np.array_equal(self.length.nonzero(), l.nonzero()):
            new_length = l/[x if x != 0 else 1 for x in self.length]
            self.x *= new_length
            self.y *= new_length
            self.z *= new_length
            return
        # Error if length zero array is resized to nonzero value
        raise ZeroDivisionError('Cannot resize vector of length 0 to '
                                'nonzero length')

    def as_length(self, l):
        """Scale the length of a vector to a value"""
        V = self.copy()
        V.length = l
        return V

    def as_percent(self, p):
        """Scale the length of a vector by a percent"""
        V = self.copy()
        V.length = p * self.length
        return V

    def as_unit(self):
        """Scale the length of a vector to 1"""
        V = self.copy()
        V.normalize()
        return V

    def normalize(self):
        """Scale the length of a vector to 1 in place"""
        self.length = np.ones(self.nV)
        return self

    def dot(self, vec):
        """Dot product with another vector"""
        if not isinstance(vec, Vector3Array):
            raise TypeError('Dot product operand must be a vector')
        if self.nV != 1 and vec.nV != 1 and self.nV != vec.nV:
            raise ValueError('Dot product operands must have the same '
                             'number of elements.')
        return self.x*vec.x + self.y*vec.y + self.z*vec.z

    def cross(self, vec):
        """Cross product with another vector"""
        if not isinstance(vec, Vector3Array):
            raise TypeError('Cross product operand must be a Vector3Array')
        if self.nV != 1 and vec.nV != 1 and self.nV != vec.nV:
            raise ValueError('Cross product operands must have the same '
                             'number of elements.')
        return Vector3Array(np.cross(self, vec))

    def __mul__(self, m):
        return Vector3Array(self.view(np.ndarray) * m)


class Vector2Array(np.ndarray):
    """
        Primitive vectors, or list of primitive vectors,
        defined from the origin.
    """

    def __new__(cls, x=None, y=None):

        def read_array(X, Y):
            if isinstance(X, cls) and Y is None:
                X = np.atleast_2d(X)
                return cls(X.x.copy(), X.y.copy())
            if isinstance(X, (list, tuple)):
                X = np.array(X)
            if isinstance(Y, (list, tuple)):
                Y = np.array(Y)
            if isinstance(X, np.ndarray) and Y is None:
                X = np.squeeze(X)
                if X.size == 2:
                    X = X.flatten()
                    return cls(X[0], X[1])
                elif len(X.shape) == 2 and X.shape[1] == 2:
                    return cls(
                        X[:, 0].copy(), X[:, 1].copy()
                    )
                raise ValueError(
                    'Unexpected shape for vector init: {shp}'.format(
                        shp=X.shape
                    )
                )
            if np.isscalar(X) and np.isscalar(Y):
                X, Y = float(X), float(Y)
            elif not isinstance(X, type(Y)):
                raise TypeError('Must be the same types for x and y '
                                'for vector init')
            if isinstance(X, np.ndarray):
                if X.shape != Y.shape:
                    raise ValueError('Must be the same shapes for x and y '
                                     'in vector init')
                xy = np.c_[X, Y]
                xy = xy.astype(float)
                return xy.view(cls)
            if X is None:
                X, Y = 0.0, 0.0
            xy = np.r_[X, Y].reshape((1, 2))
            return np.asarray(xy).view(cls)

        return read_array(x, y)

    def __array_finalize__(self, obj):
        if obj is None:
            return

    def __getitem__(self, i):
        item_out = super(Vector2Array, self).__getitem__(i)
        if np.isscalar(i):
            return item_out.view(Vector2)
        if isinstance(i, slice):
            return item_out
        return item_out.view(np.ndarray)

    @property
    def x(self):
        return self[:, 0]

    @x.setter
    def x(self, value):
        self[:, 0] = value

    @property
    def y(self):
        return self[:, 1]

    @y.setter
    def y(self, value):
        self[:, 1] = value

    @property
    def z(self):
        raise Exception("Vector2 does not have a z property")

    @property
    def nV(self):
        """Number of vectors"""
        return self.shape[0]

    @property
    def length(self):
        """Vector3 length"""
        return np.sqrt(np.sum(self**2, axis=1)).view(np.ndarray)

    @length.setter
    def length(self, l):
        l = np.array(l)
        if self.nV != l.size:
            raise ValueError('Length vector must be the same number of '
                             'elements as vector.')
        # This case resizes all vectors with nonzero length
        if np.all(self.length != 0):
            new_length = l/self.length
            self.x *= new_length
            self.y *= new_length
            return
        # This case only applies to single vectors
        # if self.length == 0 and l == 0
        if self.nV == 1 and l == 0:
            assert self.length == 0, \
                'Nonzero length should be resized in the first case'
            self.x, self.y = 0, 0
            return
        # This case only applies if vectors with length == 0
        # in an array are getting resized to 0
        if self.nV > 1 and np.array_equal(self.length.nonzero(), l.nonzero()):
            new_length = l/[x if x != 0 else 1 for x in self.length]
            self.x *= new_length
            self.y *= new_length
            return
        # Error if length zero array is resized to nonzero value
        raise ZeroDivisionError('Cannot resize vector of length 0 to '
                                'nonzero length')

    def as_length(self, l):
        """Scale the length of a vector to a value"""
        V = self.copy()
        V.length = l
        return V

    def as_percent(self, p):
        """Scale the length of a vector by a percent"""
        V = self.copy()
        V.length = p * self.length
        return V

    def as_unit(self):
        """Scale the length of a vector to 1"""
        V = self.copy()
        V.normalize()
        return V

    def normalize(self):
        """Scale the length of a vector to 1 in place"""
        self.length = np.ones(self.nV)
        return self

    def dot(self, vec):
        """Dot product with another vector"""
        if not isinstance(vec, Vector2Array):
            raise TypeError('Dot product operand must be a Vector2Array')
        if self.nV != 1 and vec.nV != 1 and self.nV != vec.nV:
            raise ValueError('Dot product operands must have the same '
                             'number of elements.')
        return self.x*vec.x + self.y*vec.y

    def __mul__(self, m):
        return Vector2Array(self.view(np.ndarray) * m)
