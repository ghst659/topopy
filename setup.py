#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import setuptools
setuptools.setup(
    name='topopy',
    version='0.1',
    description='Python topo',
    long_description='Python Directed Graph',
    url='https://github.com/ghst659/topo_py.git',
    license='GPLv3',
    keywords='python topo',
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Development Status :: 4 - Beta',
        'Development Status :: 5 - Production/Stable',
        'Operating System :: MacOS :: MacOS X',
        'Operating System :: POSIX :: Linux',
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3 :: Only',
        'Topic :: Software Development :: Libraries :: Python Modules',
        'License :: GPLv3'
    ],
    author='ghst659',
    author_email='',
    packages=setuptools.find_packages(), # or manually as packages=['qe', 'qe.util', 'qe.tests', 'qe.trinity'],
    include_package_data=True,           # see MANIFEST.in on globs of files to include, e.g. json files
    scripts=[],
    install_requires=[]         # packages from http://pypi.python.org
)
