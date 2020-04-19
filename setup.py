#!/usr/bin/env python

from distutils.core import setup

with open('requirements.txt') as fp:
    requirements = fp.read().splitlines()

setup(name='measure-propagation',
      version='0.1.0',
      description='Python implementation of "Semi-supervised learning with measure propagation."',
      packages=['measure_propagation'],
      install_requires=requirements,
      )
