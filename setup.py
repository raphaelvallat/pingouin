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
VERSION = '0.2.9'
PACKAGE_DATA = {'pingouin.data.icons': ['*.svg']}

INSTALL_REQUIRES = [
    'numpy>=1.15',
    'scipy>=1.3',
    'pandas>=0.24',
    'matplotlib>=3.0.2',
    'seaborn>=0.9.0',
    'statsmodels>=0.10.0',
    'pandas_flavor>=0.1.2'
]

PACKAGES = [
    'pingouin',
    'pingouin.datasets',
    'pingouin.external',
]

CLASSIFIERS = [
    'Intended Audience :: Science/Research',
    'Programming Language :: Python :: 3.6',
    'Programming Language :: Python :: 3.7',
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
          license=read('LICENSE'),
          url=URL,
          version=VERSION,
          download_url=DOWNLOAD_URL,
          install_requires=INSTALL_REQUIRES,
          include_package_data=True,
          packages=PACKAGES,
          package_data=PACKAGE_DATA,
          classifiers=CLASSIFIERS,
          )
