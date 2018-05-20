import os
from setuptools import setup

setup(
    name="pcanet",
    version="0.0.1",
    author="Takeshi Ishita",
    py_modules=["pcanet"],
    install_requires=[
        'cupy==5.0.0a1',
        'chainer',
        'numpy',
        'psutil',
        'recommonmark',
        'scikit-learn',
        'scipy',
        'sphinx',
    ]
)
