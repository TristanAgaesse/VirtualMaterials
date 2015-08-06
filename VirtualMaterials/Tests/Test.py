# -*- coding: utf-8 -*-
"""
Created on Wed Aug  5 11:08:55 2015

@author: greentoto
"""

def TestExtractNetwork():

    from VirtualMaterials.VirtualImages import VirtualMaterialsGeneration
    from VirtualMaterials.Simulation  import PoreNetworkExtraction
    from VirtualMaterials.Utilities  import tifffile as tff
    import numpy as np
    
    #Image 1
    image = VirtualMaterialsGeneration.CreateTestImage_TetrahedronMedialAxis()
    imageFile="temp_testImageTetrahedronMedialAxis_allPhases.tif"
    tff.imsave(imageFile,image)
    outputFileName = "temp_pnmExtract_testImageTetrahedronMedialAxis_allPhases"
    PNMGeometricData = PoreNetworkExtraction.ExtractNetwork(imageFile,outputFileName,phases={'void0':0,'void1':50,'void2':100},hContrast=4) 
    tff.imsave(outputFileName+"_imageLiens.tif",PNMGeometricData["imageLiens"].astype(np.uint32))    
    tff.imsave(outputFileName+"_imagePores.tif",PNMGeometricData["imagePores"].astype(np.uint32))
    
    #Image 2
    image = VirtualMaterialsGeneration.CreateTestImage_TetrahedronMedialAxis()
    image[image==50]=0
    image[image==100]=0 
    imageFile = "temp_testImageTetrahedronMedialAxis_uniquePhase.tif"
    tff.imsave(imageFile,image)
    outputFileName = "temp_pnmExtract_testImageTetrahedronMedialAxis_uniquePhase"
    PNMGeometricData = PoreNetworkExtraction.ExtractNetwork(imageFile,outputFileName,phases={'void':0},hContrast=4)
    tff.imsave(outputFileName+"_imageLiens.tif",PNMGeometricData["imageLiens"].astype(np.uint32))    
    tff.imsave(outputFileName+"_imagePores.tif",PNMGeometricData["imagePores"].astype(np.uint32))
    
    #Image 3
    image = VirtualMaterialsGeneration.CreateTestImage_TwoBalls()
    imageFile = "temp_testImageTwoBalls.tif"
    tff.imsave(imageFile,image)
    outputFileName = "temp_pnmExtract_testImageTwoBalls"
    PNMGeometricData = PoreNetworkExtraction.ExtractNetwork(imageFile,outputFileName,phases={'void':0},hContrast=2)
    tff.imsave(outputFileName+"_imageLiens.tif",PNMGeometricData["imageLiens"].astype(np.uint32))    
    tff.imsave(outputFileName+"_imagePores.tif",PNMGeometricData["imagePores"].astype(np.uint32))
    
    #Image 4
    image = VirtualMaterialsGeneration.CreateTestImage_ThreeBalls()
    imageFile ="temp_testImageThreeBalls.tif"
    tff.imsave(imageFile,image)
    outputFileName = "temp_pnmExtract_testImageThreeBalls"
    PNMGeometricData = PoreNetworkExtraction.ExtractNetwork(imageFile,outputFileName,phases={'void':0},hContrast=2)         
    tff.imsave(outputFileName+"_imageLiens.tif",PNMGeometricData["imageLiens"].astype(np.uint32))    
    tff.imsave(outputFileName+"_imagePores.tif",PNMGeometricData["imagePores"].astype(np.uint32))
                  
                  