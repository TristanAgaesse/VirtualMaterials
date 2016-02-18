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
    description="A Python toolbox to create virtual material images and to compute their physical properties..",
    
    packages=['VirtualMaterials',
              'VirtualMaterials.VirtualImages',
              'VirtualMaterials.Simulation',
              'VirtualMaterials.Utilities',
              'VirtualMaterials.Examples',
              'VirtualMaterials.test',
              'VirtualMaterials.ImageCleaning'],
              
    install_requires = [
        'numpy',
        'scipy',
        'scikit-image',
        'SimpleITK',
        'hdf5storage',
	      'mayavi'],     
        
    author='CEA LITEN',
)

