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

.. image:: https://codecov.io/gh/3ptscience/vectormath/branch/master/graph/badge.svg
    :target: https://codecov.io/gh/3ptscience/vectormath
    :alt: Code test coverage


Vector math utilities for Python built on `NumPy <http://www.numpy.org/>`_


Why
---

The :code:`vectormath` package provides a fast, simple library of vector math
utilities by leveraging NumPy. This allows explicit
geometric constructs to be created (for example, :code:`Vector3` and :code:`Plane`)
without redefining the underlying array math.

Scope
-----

The :code:`vectormath` package includes :code:`Vector3`/:code:`Vector2`,
:code:`Vector3Array`/:code:`Vector2Array`, :code:`Matrix3`, :code:`Plane`,
and :code:`Parallelogram`. The latter three classes build on Vectors to
simplify initialization and operations.


Goals
-----

* Speed: All low-level operations rely on NumPy arrays. These are densely packed,
  typed, and partially implemented in C. The :code:`VectorArray` classes in particular
  take advantage of this speed by performing vector operations on all Vectors at
  once, rather than in a loop.
* Simplicty: High-level operations are explicit and straight-forward.
  This library should be usable by Programmers, Mathematicians, and Geologists.


Alternatives
------------

* `NumPy <http://www.numpy.org/>`_ can be used for any array operations
* Many small libraries on PyPI (e.g. `vectors <https://github.com/allelos/vectors>`_)
  implement vector math operations but are are only built with single vectors
  in mind.

Connections
-----------

* `properties <https://github.com/3ptscience/properties>`_ uses :code:`vectormath`
  as the underlying framework for Vector properties.

Installation
------------

To install the repository, ensure that you have
`pip installed <https://pip.pypa.io/en/stable/installing/>`_ and run:

.. code::

    pip install vectormath

For the development version:

.. code::

    git clone https://github.com/3ptscience/vectormath.git
    cd vectormath
    pip install -e .


Examples
========

This example gives a brief demonstration of some of the notable features of
:code:`Vector3` and :code:`Vector3Array`

.. code:: python

    import numpy as np
    import vectormath as vmath

    # Single Vectors
    v = vmath.Vector3(5, 0, 0)
    v.normalize()
    print(v)                          # >> [1, 0, 0]
    print(v.x)                        # >> 1.0

    # VectorArrays are much faster than a for loop over Vectors
    v_array = vmath.Vector3Array([[4, 0, 0], [0, 2, 0], [0, 0, 3]])
    print(v_array.x)                  # >> [4, 0, 0]
    print(v_array.length)             # >> [4, 2, 3]
    print(v_array.normalize())        # >> [[1, 0, 0], [0, 1, 0], [0, 0, 1]]

    # Vectors can be accessed individually or in slices
    print(type(v_array[1:]))          # >> vectormath.Vector3Array
    print(type(v_array[2]))           # >> vectormath.Vector3

    # All these classes are just numpy arrays
    print(isinstance(v, np.ndarray))  # >> True
    print(type(v_array[1:, 1:]))      # >> numpy.ndarray


Current version: v0.1.0
