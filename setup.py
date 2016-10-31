#!/usr/bin/env python
"""
    Vector math utilities for Python
"""

from distutils.core import setup
from setuptools import find_packages

CLASSIFIERS = [
    'Development Status :: 4 - Beta',
    'Programming Language :: Python',
    'Topic :: Scientific/Engineering',
    'Topic :: Scientific/Engineering :: Mathematics',
    'Topic :: Scientific/Engineering :: Physics',
    'Operating System :: Microsoft :: Windows',
    'Operating System :: POSIX',
    'Operating System :: Unix',
    'Operating System :: MacOS',
    'Natural Language :: English',
]

with open('README.rst') as f:
    LONG_DESCRIPTION = ''.join(f.readlines())

setup(
    name='vectormath',
    version='0.1.0',
    packages=find_packages(),
    install_requires=[
        'future',
        'six',
        'numpy>=1.7',
    ],
    author='3point Science',
    author_email='info@3ptscience.com',
    description='vectormath',
    long_description=LONG_DESCRIPTION,
    keywords='linear algebra, vector, plane, math',
    url='https://github.com/3ptscience/vectormath',
    download_url='http://github.com/3ptscience/vectormath',
    classifiers=CLASSIFIERS,
    platforms=['Windows', 'Linux', 'Solaris', 'Mac OS-X', 'Unix'],
    license='MIT License',
    use_2to3=False,
)
