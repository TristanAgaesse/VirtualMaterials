# -*- coding: utf-8 -*-
"""
Created on Wed Aug  5 11:08:55 2015

@author: greentoto
"""
import numpy as np
import unittest

from VirtualMaterials.VirtualImages import VirtualMaterialsGeneration
from VirtualMaterials.Simulation  import PoreNetworkExtraction


class test_PoreNetworkExtraction(unittest.TestCase):
    

    def test_ExtractNetwork(self):
    
        #Image 1
        image = VirtualMaterialsGeneration.CreateTestImage_TetrahedronMedialAxis()
        PNMGeometricData = PoreNetworkExtraction.ExtractNetwork(image=image,phases={'void0':0,'void1':50,'void2':100},seedMethod='hMaxima',seedParam=4) 

        nPore=5
        nInternalLink=2
        interfaceToPore = np.array([[3, 4] ,[4, 5]])
        porePhase = np.array([100 , 50 ,  0  , 0 ,  0])
        
        self.assertTrue(PNMGeometricData['imagePores'].max()==nPore)
        self.assertTrue(PNMGeometricData['imageLiens'].max()==nInternalLink)
        self.assertTrue(np.all(PNMGeometricData['interfaceToPore']==interfaceToPore))
        self.assertTrue(np.all(PNMGeometricData['porePhase']==porePhase)) 
       

        #Image 2
        image = VirtualMaterialsGeneration.CreateTestImage_TetrahedronMedialAxis()
        image[image==50]=0
        image[image==100]=0 
        PNMGeometricData = PoreNetworkExtraction.ExtractNetwork(image=image,phases={'void':0},seedMethod='hMaxima',seedParam=4)
       
        nPore=5
        nInternalLink=4
        interfaceToPore = np.array([[3, 5], [2, 3], [1, 3], [3, 4]])
        porePhase = np.array([0 , 0 ,  0  , 0 ,  0])

        self.assertTrue(PNMGeometricData['imagePores'].max()==nPore)
        self.assertTrue(PNMGeometricData['imageLiens'].max()==nInternalLink)
        self.assertTrue(np.all(PNMGeometricData['interfaceToPore']==interfaceToPore))
        self.assertTrue(np.all(PNMGeometricData['porePhase']==porePhase))
        
        
        #Image 3
        image = VirtualMaterialsGeneration.CreateTestImage_TwoBalls()
        PNMGeometricData = PoreNetworkExtraction.ExtractNetwork(image=image,phases={'void':0},seedMethod='hMaxima',seedParam=2)
       
        nPore=2
        nInternalLink=1
        interfaceToPore = np.array([[1, 2]])
        porePhase = np.array([0 , 0])
        
        self.assertTrue(PNMGeometricData['imagePores'].max()==nPore)
        self.assertTrue(PNMGeometricData['imageLiens'].max()==nInternalLink)
        self.assertTrue(np.all(PNMGeometricData['interfaceToPore']==interfaceToPore))
        self.assertTrue(np.all(PNMGeometricData['porePhase']==porePhase))
        
        #Image 4
        image = VirtualMaterialsGeneration.CreateTestImage_ThreeBalls()
        PNMGeometricData = PoreNetworkExtraction.ExtractNetwork(image=image,phases={'void':0},seedMethod='hMaxima',seedParam=2)         
       
        nPore=3
        nInternalLink=3
        interfaceToPore = np.array([[2, 3], [1, 3], [1, 2]])
        porePhase = np.array([0 , 0,0])
       
        self.assertTrue(PNMGeometricData['imagePores'].max()==nPore)
        self.assertTrue(PNMGeometricData['imageLiens'].max()==nInternalLink)
        self.assertTrue(np.all(PNMGeometricData['interfaceToPore']==interfaceToPore))
        self.assertTrue(np.all(PNMGeometricData['porePhase']==porePhase))
        
       
        print('Test_PoreNetworkExtraction_ExtractNetwork : need to improve coverage ')              
    
    
    
    def test_ParseLabeledImage(self):
        

        links=np.array([0, 0, 2, 1,2,1, 4])
        voxelLookUpTable = PoreNetworkExtraction.BuildVoxelLookUpTable(links)
        voxels0=PoreNetworkExtraction.GetVoxelOfLabel(0,voxelLookUpTable)
        voxels1=PoreNetworkExtraction.GetVoxelOfLabel(1,voxelLookUpTable)
        voxels2=PoreNetworkExtraction.GetVoxelOfLabel(2,voxelLookUpTable)
        voxels3=PoreNetworkExtraction.GetVoxelOfLabel(3,voxelLookUpTable)
        voxels4=PoreNetworkExtraction.GetVoxelOfLabel(4,voxelLookUpTable)
    
        #self.assertTrue( np.all(voxels0==np.array([0, 1])) )
        self.assertTrue( np.all(voxels1==np.array([3, 5])) ) 
        self.assertTrue( np.all(voxels2==np.array([2,4])) )
        self.assertTrue( np.all(voxels3==np.array([])) ) 
        self.assertTrue( np.all(voxels4==np.array([6])) )  
    
        links=np.array([0, 0])
        voxelLookUpTable = PoreNetworkExtraction.BuildVoxelLookUpTable(links)
        #voxels0=PoreNetworkExtraction.GetVoxelOfLabel(0,voxelLookUpTable)
        #self.assertTrue( np.all(voxels0==np.array([0, 1])) )
        
        
        print('Test_PoreNetworkExtraction_ParseLabeledImage : need to improve coverage')                
        
        
    def test_FastDilation(self):
        
        #image2
        image= np.array([0,0,1,1,0,0,0,1])
        structuringElement = [1,1,1]

        dilatedImage=PoreNetworkExtraction.__FastDilation__(image,structuringElement) 
        solution = np.array([0,1,1,1,1,0,1,1])
        self.assertTrue( np.all(dilatedImage==solution) )
    
        #image2
        image= np.array([[0,0,1,1,0,0,0,1],[0,0,0,0,0,0,1,0],[0,1,0,0,0,0,0,0]])
        structuringElement = np.array([[0,1,0],[1,1,1],[0,1,0]])
        
        dilatedImage=PoreNetworkExtraction.__FastDilation__(image,structuringElement) 
        solution = np.array([[0, 1,  1,  1,  1, 0,  1,  1], 
                             [0,  1,  1,  1, 0 , 1 , 1 , 1], 
                             [ 1 , 1 , 1 ,0, 0 ,0  ,1 ,0]])
        self.assertTrue( np.all(dilatedImage==solution) )

    
    
    
#----------------------------------------------------------------------------------------------
if __name__ == '__main__':
    unittest.main()   