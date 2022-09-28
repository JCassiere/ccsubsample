#!/usr/bin/env python
from setuptools import find_packages, setup

setup(
    name="ccsubsample",
    version="0.1.0",
    description="Computational Chemistry Subsampling",
    author="Joe Cassiere",
    author_email="jcassiere@gatech.edu",
    url="https://github.com/JCassiere/ccsubsample",
    packages=find_packages(exclude=["contrib", "docs", "tests"]),
    python_requires=">=3.6, <4"
)
