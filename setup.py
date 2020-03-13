from __future__ import absolute_import, print_function
from setuptools import setup, find_packages
from os import path

_dir = path.abspath(path.dirname(__file__))

with open(path.join(_dir,'phasenet','version.py')) as f:
    exec(f.read())

setup (
    name='phasenet',
    version=__version__,
    description='PhaseNet',
    license='BSD 3-Clause License',
    packages=find_packages(),
    python_requires='>=3.6',
    install_requires=[
        'csbdeep>=0.4.0',
    ],
)
