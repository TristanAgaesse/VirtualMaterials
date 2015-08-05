# -*- coding: utf-8 -*-
"""
Created on Wed Aug  5 11:08:55 2015

@author: greentoto
"""

def TestExtractNetwork():

    from VirtualMaterials.VirtualImages import VirtualMaterialsGeneration
    from VirtualMaterials.Simulation  import PoreNetworkExtraction
    from VirtualMaterials.Utilities  import tifffile as tff

    image = VirtualMaterialsGeneration.CreateTestImage_TetrahedronMedialAxis()
    tff.imsave("temp_testImage.tif",image)
    
    inputFileName = "temp_testImage.tif"
    outputFileName = "temp_pnmExtract_testImage_allPhases.mat"
    
    PoreNetworkExtraction.ExtractNetwork(inputFileName,
                  outputFileName,phases={'void0':0,'void1':50,'void2':100},hContrast=4)
     
     
    image[image==50]=0
    image[image==100]=0 
    tff.imsave("temp_testImage2.tif",image)
    
    inputFileName = "temp_testImage2.tif"
    outputFileName = "temp_pnmExtract_testImage_uniquePhase.mat"
    
    PoreNetworkExtraction.ExtractNetwork(inputFileName,
                  outputFileName,phases={'void0':0},hContrast=4)
                  
    #Then process network with PNM_Matlab              
                  