#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import sys
sys.path.append(os.getcwd())

try:
    from setuptools import setup
except ImportError:
    from distutils.core import setup

setup(
    name='VirtualMaterials',
    packages=['VirtualMaterials',
              'VirtualMaterials.SampleImages',
              'VirtualMaterials.Simulation',
              'VirtualMaterials.VirtualImages',
              'VirtualMaterials.Utilities'],
    install_requires = [
        'numpy',
        'vtk',
        'scipy',
        'scikit-image',
        'SimpleITK',
        'mayavi',
        'hdf5storage'
    ],        
    description="A framework for handling virtual images of porous media.",
    author='CEA LITEN',
    author_email='jeff.gostick@mcgill.ca',
)

