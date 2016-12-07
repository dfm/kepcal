#!/usr/bin/env python
# encoding: utf-8

import os
import glob
from setuptools import setup

import kepcal

# Execute the setup command.
desc = open("README.rst").read()
setup(
    name="kepcal",
    version=kepcal.__version__,
    author="Daniel Foreman-Mackey",
    author_email="foreman.mackey@gmail.com",
    packages=[
        "kepcal",
    ],
    scripts=list(glob.glob(os.path.join("scripts", "kepcal-*"))),
    url="http://github.com/dfm/kepcal",
    license="MIT",
    description="Self calibration using the Kepler FFIs",
    long_description=desc,
    package_data={"": ["README.rst", "LICENSE"]},
    include_package_data=True,
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python 3",
    ],
)
