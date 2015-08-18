# -*- coding: utf-8 -*-
"""
Created on Wed Aug  5 11:08:55 2015

@author: greentoto
"""
import numpy as np


#----------------------------------------------------------------------------------------------
def RunAllTests():

    Test_PoreNetworkExtraction_ExtractNetwork()
    Test_PoreNetworkExtraction_ParseLabeledImage()
                

#----------------------------------------------------------------------------------------------
def Test_PoreNetworkExtraction_ExtractNetwork():

    from VirtualMaterials.VirtualImages import VirtualMaterialsGeneration
    from VirtualMaterials.Simulation  import PoreNetworkExtraction
    from VirtualMaterials.Utilities  import tifffile as tff
    
    
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
                  
    print('Test_PoreNetworkExtraction_ExtractNetwork : ok')              



#----------------------------------------------------------------------------------------------
def Test_PoreNetworkExtraction_ParseLabeledImage():
    
    from VirtualMaterials.Simulation  import PoreNetworkExtraction    
    
    links=np.array([0, 0, 2, 1,2,1, 4])
    voxelLookUpTable = PoreNetworkExtraction.BuildVoxelLookUpTable(links)
    voxels1=PoreNetworkExtraction.GetVoxelOfLabel(1,voxelLookUpTable)
    voxels2=PoreNetworkExtraction.GetVoxelOfLabel(2,voxelLookUpTable)
    voxels3=PoreNetworkExtraction.GetVoxelOfLabel(3,voxelLookUpTable)
    voxels4=PoreNetworkExtraction.GetVoxelOfLabel(4,voxelLookUpTable)

    assert( np.all(voxels1==np.array([3, 5])) ) 
    assert( np.all(voxels2==np.array([2,4])) )
    assert( np.all(voxels3==np.array([])) ) 
    assert( np.all(voxels4==np.array([6])) )  

    print('Test_PoreNetworkExtraction_ParseLabeledImage : ok')                
    







    
#----------------------------------------------------------------------------------------------
if __name__ == '__main__':
    RunAllTests()    