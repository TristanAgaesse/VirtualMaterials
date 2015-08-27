# -*- coding: utf-8 -*-

import numpy as np
from scipy import ndimage
import SimpleITK as sitk
import time

#from skimage import morphology

#----------------------------------------------------------------------------------------------

def FullMorphology(inputImage,inletFace=1,voxelLength=1,pressureList=[10],pressureCode=[110],gamma=72e-3):
    """inletFace=0 for Xmin, 1 for Xmax, 2 for Ymin, 3 for Ymax, 4 for Zmin, 5 for Zmax"""
    
    
    
        
    beginTime=time.time()
            
    myImg = inputImage

    inletVoxels = np.zeros(myImg.shape,dtype=bool)
    
    
    if inletFace==0:
        inletVoxels[0,:,:] = True
    elif inletFace==1:
        inletVoxels[-1,:,:] = True
    elif inletFace==2:
        inletVoxels[:,0,:] = True
    elif inletFace==3:
        inletVoxels[:,-1,:] = True
    elif inletFace==4:
        inletVoxels[:,:,0] = True
    elif inletFace==5:
        inletVoxels[:,:,-1] = True        
        
    #distanceMap = ndimage.distance_transform_edt(np.logical_not(myImg)).astype(np.float16)
    memoryType=np.float16
    itkimage = sitk.GetImageFromArray(np.logical_not(myImg==0).astype(np.uint8))
    itkdistanceMap = sitk.DanielssonDistanceMap( itkimage )
    distanceMap=sitk.GetArrayFromImage(itkdistanceMap).astype(memoryType)


    pressureCode=np.asarray(pressureCode).astype(np.uint8)
    pressureList = np.asarray(pressureList)
    ascendingOrder=np.argsort(pressureList)
    
    for i in range(len(pressureList)-1,-1,-1):
        print('begin '+str(i))
        pressure = pressureList[ascendingOrder[i]]
        water = FullMorphologyHydrophobicStep(distanceMap,2*gamma/(pressure*voxelLength),inletVoxels)
        myImg[water] = pressureCode[ascendingOrder[i]]
        del water  
    
      
      
    

    endTime=time.time()
    print("Time spent : {} s".format(endTime-beginTime))

    return myImg.astype(np.uint8)

#----------------------------------------------------------------------------------------------

def FullMorphologyHydrophobicStep(distanceMap,capillaryLength,inletVoxels):
    
    #Find the centers of the water balls
    indicesCenters=(distanceMap>capillaryLength).reshape(distanceMap.shape)
    invadedVoxels=np.zeros(distanceMap.shape,dtype=bool)
    invadedVoxels[indicesCenters]=True
    del indicesCenters
    
    
     
              
    #Keep only water connected with the inlet          
    structuringElement = np.ones((3,3,3))
    invadedVoxels=ndimage.measurements.label(invadedVoxels,structuringElement)[0]
    
    labelsToKeep=np.unique(invadedVoxels[inletVoxels])    
    labelsToKeep=labelsToKeep[labelsToKeep>0]
    
    foo=np.zeros(invadedVoxels.shape,dtype=bool)
    
    for iLabel in labelsToKeep :
        foo[invadedVoxels==iLabel] = True
    
    del labelsToKeep
    
    invadedVoxels=invadedVoxels.astype(np.bool)
    invadedVoxels[np.logical_not(foo)]=False
    invadedVoxels[foo]=True
    
    #Dilate with a ball of size capillaryLength
    
    #Scipy implementation of dilation
    #ball=morphology.ball(capillaryLength)
    #invadedVoxels=ndimage.morphology.binary_dilation(invadedVoxels, structure=ball)     
    
    #SimpleITK implementation of dilation
    itkInvadedVoxels = sitk.GetImageFromArray(invadedVoxels.astype(np.uint8))
    itkInvadedVoxels = sitk.BinaryDilate(itkInvadedVoxels, int(capillaryLength), sitk.sitkBall, 0.0, 1.0,  False)   
    invadedVoxels=sitk.GetArrayFromImage(itkInvadedVoxels)  
    invadedVoxels=invadedVoxels.astype(np.bool) 
    
    
    return invadedVoxels
    
    

