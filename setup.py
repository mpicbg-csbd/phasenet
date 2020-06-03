from __future__ import absolute_import, print_function
from setuptools import setup, find_packages
from os import path

_dir = path.abspath(path.dirname(__file__))

with open(path.join(_dir,'phasenet','version.py'), encoding='utf-8') as f:
    exec(f.read())

with open(path.join(_dir,'README.md'), encoding='utf-8') as f:
    long_description = f.read()


setup (
    name='phasenet',
    version=__version__,
    description='PhaseNet',
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='https://github.com/mpicbg-csbd/phasenet',
    author='Debayan Saha, Martin Weigert, Uwe Schmidt',
    author_email='dsaha@mpi-cbg.de, martin.weigert@epfl.ch, uschmidt@mpi-cbg.de',
    license='BSD 3-Clause License',
    packages=find_packages(),
    python_requires='>=3.6',
    install_requires=[
        'csbdeep>=0.4.0',
    ],
)
