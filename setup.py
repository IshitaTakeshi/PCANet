import os
from setuptools import setup
from Cython.Build import cythonize

setup(
    name="pcanet",
    version="0.0.1",
    author="Takeshi Ishita",
    py_modules=["pcanet"],
    ext_modules=cythonize("histogram.pyx"),
    install_requires=[
        'chainer',
        'numpy',
        'psutil',
        'python-mnist',
        'recommonmark',
        'scikit-learn',
        'scipy',
        'sphinx',
    ],
)
