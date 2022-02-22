#! /usr/bin/env python
#
# Copyright (C) 2018 Raphael Vallat
import os

def read(fname):
    return open(os.path.join(os.path.dirname(__file__), fname)).read()

DESCRIPTION = "Pingouin: statistical package for Python"
LONG_DESCRIPTION = """Pingouin is a statistical Python package based on Pandas.
"""

DISTNAME = 'pingouin'
MAINTAINER = 'Raphael Vallat'
MAINTAINER_EMAIL = 'raphaelvallat9@gmail.com'
URL = 'https://pingouin-stats.org/index.html'
DOWNLOAD_URL = 'https://github.com/raphaelvallat/pingouin/'
VERSION = '0.5.1'
LICENSE = 'GPL-3.0'
PACKAGE_DATA = {'pingouin.data.icons': ['*.svg']}

INSTALL_REQUIRES = [
    'numpy>=1.19',
    'scipy>=1.7',
    'pandas>=1.0',
    'matplotlib>=3.0.2',
    'seaborn>=0.11',
    'statsmodels>=0.13',
    'scikit-learn',
    'pandas_flavor>=0.2.0',
    'outdated',
    'tabulate'
]

PACKAGES = [
    'pingouin',
    'pingouin.datasets',
]

CLASSIFIERS = [
    'Intended Audience :: Science/Research',
    'Programming Language :: Python :: 3.7',
    'Programming Language :: Python :: 3.8',
    'Programming Language :: Python :: 3.9',
    'Topic :: Scientific/Engineering :: Mathematics',
    'Operating System :: POSIX',
    'Operating System :: Unix',
    'Operating System :: MacOS'
]

try:
    from setuptools import setup
    _has_setuptools = True
except ImportError:
    from distutils.core import setup

if __name__ == "__main__":

    setup(name=DISTNAME,
          author=MAINTAINER,
          author_email=MAINTAINER_EMAIL,
          maintainer=MAINTAINER,
          maintainer_email=MAINTAINER_EMAIL,
          description=DESCRIPTION,
          long_description=LONG_DESCRIPTION,
          license=LICENSE,
          url=URL,
          version=VERSION,
          download_url=DOWNLOAD_URL,
          install_requires=INSTALL_REQUIRES,
          include_package_data=True,
          packages=PACKAGES,
          package_data=PACKAGE_DATA,
          classifiers=CLASSIFIERS,
          )
