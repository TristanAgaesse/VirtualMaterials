# -*- coding: utf-8 -*-
import numpy as np
import unittest

import VirtualMaterials.ImageAnalysis  as imageAnalysis


class test_ImageAnalysis(unittest.TestCase):
    

    
    def test_FastDilation(self):
        
        #image2
        image= np.array([0,0,1,1,0,0,0,1])
        structuringElement = [1,1,1]

        dilatedImage=imageAnalysis.Morphology.FastDilation(image,structuringElement) 
        solution = np.array([0,1,1,1,1,0,1,1])
        self.assertTrue( np.all(dilatedImage==solution) )
    
        #image2
        image= np.array([[0,0,1,1,0,0,0,1],[0,0,0,0,0,0,1,0],[0,1,0,0,0,0,0,0]])
        structuringElement = np.array([[0,1,0],[1,1,1],[0,1,0]])
        
        dilatedImage=imageAnalysis.Morphology.FastDilation(image,structuringElement) 
        solution = np.array([[0, 1,  1,  1,  1, 0,  1,  1], 
                             [0,  1,  1,  1, 0 , 1 , 1 , 1], 
                             [ 1 , 1 , 1 ,0, 0 ,0  ,1 ,0]])
        self.assertTrue( np.all(dilatedImage==solution) )
    
    
    
#----------------------------------------------------------------------------------------------
if __name__ == '__main__':
    unittest.main()   