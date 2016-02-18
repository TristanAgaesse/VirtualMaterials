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
              'VirtualMaterials.VirtualImages',
              'VirtualMaterials.Simulation',
              'VirtualMaterials.Utilities',
              'VirtualMaterials.Examples',
              'VirtualMaterials.test',
              'VirtualMaterials.ImageCleaning'],
              
    install_requires = [
        'numpy',
        'vtk',
        'scipy',
        'scikit-image',
        'SimpleITK',
        'hdf5storage',
	      'mayavi'],     
        
    description="A Python toolbox to create virtual material images and to compute their physical properties..",
    author='CEA LITEN',
)

