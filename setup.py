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
    version='0.1.4',
    packages=find_packages(),
    install_requires=[
        'numpy>=1.7',
    ],
    author='ARANZ Geo Limited',
    author_email='it@aranzgeo.com',
    description='vectormath: vector math utilities for Python',
    long_description=LONG_DESCRIPTION,
    keywords='linear algebra, vector, plane, math',
    url='https://github.com/aranzgeo/vectormath',
    download_url='https://github.com/aranzgeo/vectormath',
    classifiers=CLASSIFIERS,
    platforms=['Windows', 'Linux', 'Solaris', 'Mac OS-X', 'Unix'],
    license='MIT License',
    include_package_data=True,
    use_2to3=False,
)
