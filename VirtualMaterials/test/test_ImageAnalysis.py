# -*- coding: utf-8 -*-
import numpy as np
import unittest

import VirtualMaterials.ImageAnalysis  as imageAnalysis


class test_ImageAnalysis(unittest.TestCase):
    
    def test_Dilation(self):
        image= np.array([[0,0,1,1,0,0,0,1],[0,0,0,0,0,0,1,0],[0,1,0,0,0,0,0,0]])
        
        ballRadius = 1
        dilatedImage=imageAnalysis.Morphology.Dilation(image,ballRadius) 
        solution = np.array([[0, 1,  1,  1,  1, 1,  1,  1], 
                             [1,  1,  1,  1, 1 , 1 , 1 , 1], 
                             [ 1 , 1 , 1 ,0, 0 ,1  ,1 ,1]])
        self.assertTrue( np.all(dilatedImage==solution) )
        
        
    def test_Erosion(self):
        image= np.array([[0, 1,  1,  1,  1, 1,  1,  1], 
                             [1,  1,  1,  1, 1 , 1 , 1 , 1], 
                             [ 1 , 1 , 1 ,0, 0 ,1  ,1 ,1]])
        
        ballRadius = 1
        erodedImage=imageAnalysis.Morphology.Erosion(image,ballRadius) 
        solution = np.array([[0, 0,  1,  1,  1, 1,  1,  1], 
                             [0,  0,  0,  0, 0 , 0 , 1 , 1], 
                             [ 1 , 1 , 0 ,0, 0 ,0  ,1 ,1]])
        self.assertTrue( np.all(erodedImage==solution) )    
    
    
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
    
    
    def test_ChordLength(self):
        
        image=np.ones((100,50,75))
        label=1
        
        clX = imageAnalysis.QuantifyGeometry.ChordLength(
            image,label,direction=(1,0,0),mode='meanLength')
        clY = imageAnalysis.QuantifyGeometry.ChordLength(
            image,label,direction=(0,1,0),mode='meanLength')    
        clZ = imageAnalysis.QuantifyGeometry.ChordLength(
            image,label,direction=(0,0,1),mode='meanLength')   
            
        self.assertTrue( np.all(np.asarray((clX,clY,clZ))==image.shape) )    
            
#----------------------------------------------------------------------------------------------
if __name__ == '__main__':
    unittest.main()   