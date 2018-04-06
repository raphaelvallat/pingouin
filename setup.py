#! /usr/bin/env python
#
# Copyright (C) 2018 Raphael Vallat
import os

DESCRIPTION = "Pingouin: statistical package for Python"
LONG_DESCRIPTION = """Pingouin is a statistical Python package based on Pandas.
"""

DISTNAME = 'pingouin'
MAINTAINER = 'Raphael Vallat'
MAINTAINER_EMAIL = 'raphaelvallat9@gmail.com'
URL = ''
LICENSE = 'BSD (3-clause)'
DOWNLOAD_URL = 'https://github.com/raphaelvallat/pingouin/'
VERSION = '0.1.dev'
PACKAGE_DATA = {'pingouin.data.icons': ['*.svg']}

try:
    from setuptools import setup
    _has_setuptools = True
except ImportError:
    from distutils.core import setup

def check_dependencies():
    install_requires = []

    try:
        import numpy
    except ImportError:
        install_requires.append('numpy')
    try:
        import scipy
    except ImportError:
        install_requires.append('scipy')
    try:
        import pandas
    except ImportError:
        install_requires.append('pandas')
    try:
        import tabulate
    except ImportError:
        install_requires.append('tabulate')


    return install_requires

if __name__ == "__main__":

    install_requires = check_dependencies()

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
        install_requires=install_requires,
        packages=['pingouin'],
        package_data=PACKAGE_DATA,
        classifiers=[
                     'Intended Audience :: Science/Research',
                     'Programming Language :: Python :: 3.6',
                     'License :: OSI Approved :: BSD License',
                     'Topic :: Scientific/Engineering :: Mathematics',
                     'Operating System :: POSIX',
                     'Operating System :: Unix',
                     'Operating System :: MacOS'],
          )
