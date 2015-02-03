# -*- coding: utf-8 -*-
"""
Created on Thu Nov 27 19:53:34 2014

@author: greentoto
"""

import numpy as np
#from skimage import morphology
from scipy import ndimage
import SimpleITK as sitk
import sys
import os

sys.path.append(os.getcwd())
import tifffile as tff
import time

#----------------------------------------------------------------------------------------------

def FullMorphology(inputFileName,outputFileName,voxelLength=1,pressureList=[10],pressureCode=[110]):
    
    beginTime=time.time()
            
    myImg=tff.imread(inputFileName).astype(np.uint8)

    inletVoxels = np.zeros(myImg.shape,dtype=bool)
    inletVoxels[-1,:,:] = True
    
    #distanceMap = ndimage.distance_transform_edt(np.logical_not(myImg)).astype(np.float16)
    memoryType=np.float16
    itkimage = sitk.GetImageFromArray(np.logical_not(myImg==0).astype(np.uint8))
    itkdistanceMap = sitk.DanielssonDistanceMap( itkimage )
    distanceMap=sitk.GetArrayFromImage(itkdistanceMap).astype(memoryType)


    gamma=72e-3

    pressureCode=np.asarray(pressureCode).astype(np.uint8)
    
    for i in range(len(pressureList)-1,-1,-1):
        print('begin '+str(i))
        pressure = pressureList[i]
        water = FullMorphologyHydrophobicStep(distanceMap,2*gamma/(pressure*voxelLength),inletVoxels)
        myImg[water] = pressureCode[i]
        del water  
    
      
    tff.imsave(outputFileName,myImg.astype(np.uint8))

    endTime=time.time()
    print("Time spent : {} s".format(endTime-beginTime))


#----------------------------------------------------------------------------------------------

def FullMorphologyHydrophobicStep(distanceMap,capillaryLength,inletVoxels):
    
    #Find the centers of the water balls
    indicesCenters=(distanceMap>capillaryLength).reshape(distanceMap.shape)
    invadedVoxels=np.zeros(distanceMap.shape,dtype=bool)
    invadedVoxels[indicesCenters]=True
    del indicesCenters
    
    
    #Dilate with a ball of size capillaryLength
    
    #Scipy implementation of dilation
    #ball=morphology.ball(capillaryLength)
    #invadedVoxels=ndimage.morphology.binary_dilation(invadedVoxels, structure=ball)     
    
    #SimpleITK implementation of dilation
    itkInvadedVoxels = sitk.GetImageFromArray(invadedVoxels.astype(np.uint8))
    itkInvadedVoxels = sitk.BinaryDilate(itkInvadedVoxels, int(capillaryLength), sitk.sitkBall, 0.0, 1.0,  False)   
    invadedVoxels=sitk.GetArrayFromImage(itkInvadedVoxels)  
    invadedVoxels=invadedVoxels.astype(np.bool)  
              
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
    
    
    return invadedVoxels
    


    


