vectormath
==========


.. image:: https://img.shields.io/pypi/v/vectormath.svg
    :target: https://pypi.python.org/pypi/vectormath
    :alt: Latest PyPI version

.. image:: https://img.shields.io/badge/license-MIT-blue.svg
    :target: https://github.com/3ptscience/vectormath/blob/master/LICENSE
    :alt: MIT license

.. image:: https://api.travis-ci.org/3ptscience/vectormath.svg?branch=master
    :target: https://travis-ci.org/3ptscience/vectormath
    :alt: Travis CI build status


Vector math utilities for Python.

.. code::

    import numpy as np
    import vectormath as vmath

    # Simple vectors
    v = vmath.Vector3(5, 0, 0)
    v.normalize()
    print(v) # > [[1, 0, 0]]
    print(v.x, type(v.x))  # > 1.0, float

    # Array based vectors are much faster than a for loop
    v_array = vmath.Vector3([[4,0,0],[0,2,0]])
    print(v_array.x) # > [4, 0]
    # we can
    print(v_array.length) # [4, 2]
    print(v_array.normalize())  # [[1,0,0],[0,1,0]]

    # These vectors are just numpy arrays
    assert isinstance(v, np.ndarray)


Major classes include :code:`Vector3`, :code:`Matrix3`, :code:`Plane`, :code:`Parallelogram`.


Current version: v0.0.3
