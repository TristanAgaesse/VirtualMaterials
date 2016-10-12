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
            image,label,direction=(1,0,0))
        clX = np.mean(clX)    
        clY = imageAnalysis.QuantifyGeometry.ChordLength(
            image,label,direction=(0,1,0))   
        clY = np.mean(clY)    
        clZ = imageAnalysis.QuantifyGeometry.ChordLength(
            image,label,direction=(0,0,1))   
        clZ = np.mean(clZ)    
        self.assertTrue( np.all(np.asarray((clX,clY,clZ))==image.shape) )    
    

    def test_PoreSizeDistribution_Continuous(self):
        image=np.zeros((30,30,30),dtype=np.bool)
        image[1:-1,1:-1,1:-1]=True
        nPoint=10
        radiusList,cPSD = imageAnalysis.QuantifyGeometry.PoreSizeDistribution_Continuous(
                                                    image,nPoint=nPoint)
        
        solution_radiusList=np.linspace(1,30,nPoint).astype(np.uint8)
        solution_cPSD = np.array([]) #TODO : 
        self.assertTrue( np.all(radiusList==solution_radiusList) )
        self.assertTrue( np.all(cPSD==solution_cPSD) )
        
        
#----------------------------------------------------------------------------------------------
if __name__ == '__main__':
    unittest.main()   