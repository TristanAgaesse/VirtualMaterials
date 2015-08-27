# -*- coding: utf-8 -*-
"""
Created on Wed Aug  5 11:08:55 2015

@author: greentoto
"""
import unittest
import numpy

from VirtualMaterials.VirtualImages import VirtualMaterialsGeneration
from VirtualMaterials.Simulation  import FullMorphology


class test_FullMorphology(unittest.TestCase):
    

    def test_FullMorphology(self):
    
        image = VirtualMaterialsGeneration.CreateTestImage_TetrahedronMedialAxis()
        image[image==50]=0
        image[image==100]=0 
        
        
        outputImage = FullMorphology.FullMorphology(image,inletFace=1,
                                     voxelLength=1,gamma=72e-3,
                                     pressureList=[10],pressureCode=[110])

        
#        self.assertTrue( == )
#        self.assertTrue(np.all( ==))
       
        print('Test_FullMorphology : need to improve coverage')
    
    
    
#----------------------------------------------------------------------------------------------
if __name__ == '__main__':
    unittest.main()   