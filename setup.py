import os
from setuptools import setup


setup(
    name="pcanet",
    version="0.0.1",
    author="Takeshi Ishita",
    py_modules=["pcanet"],
    install_requires=[
        'numpy',
        'scipy',
        'scikit-learn',
        'python-mnist',
        'sphinx',
        'recommonmark'
    ],
)
