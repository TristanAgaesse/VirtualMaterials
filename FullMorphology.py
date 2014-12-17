# -*- coding: utf-8 -*-
"""
Created on Thu Nov 27 19:53:34 2014

@author: greentoto
"""

import numpy as np
from skimage import morphology
from scipy import ndimage
import sys
import os

sys.path.append(os.getcwd())
import tifffile as tff


#----------------------------------------------------------------------------------------------

def FullMorphologyHydrophobicStep(distanceMap,capillaryLength,inletVoxels):
    
    #Find the centers of the water balls
    indicesCenters=(distanceMap>capillaryLength).reshape(distanceMap.shape)
    invadedVoxels=np.zeros(distanceMap.shape,dtype=bool)
    invadedVoxels[indicesCenters]=True
    del indicesCenters
    
    
    #Dilate with a ball of size capillaryLength
    ball=morphology.ball(capillaryLength)
    invadedVoxels=ndimage.morphology.binary_dilation(invadedVoxels, structure=ball)     
      
              
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
    

#----------------------------------------------------------------------------------------------

def main(inputFileName,outputFileName,voxelLength):
            
    myImg=tff.imread(inputFileName).astype(np.uint8)
    
    inletVoxels = np.zeros(myImg.shape,dtype=bool)
    inletVoxels[-1,:,:] = True
    
    distanceMap = ndimage.distance_transform_edt(np.logical_not(myImg)).astype(np.float16)
    
#    saturation=np.zeros(distanceMap.max())
#    
#    for iStep in range(distanceMap.max()+1,1,-1):
#        print(iStep)
#        saturation[iStep-1]=np.count_nonzero(FullMorphologyHydrophobicStep(distanceMap,iStep,inletVoxels))
#
#
#    return saturation
    

    gamma=72e-3
    
    #Calculs de répartition de l'eau pour 53 mbar, 39 mbar, 28 mbar, 22 mbar, 14 mbar 
    #pressureList=[5300,3900,2800,2200,1400]
    pressureList=[2200,1400]
    
    for i in range(len(pressureList)):
        print('begin '+str(i))
        pressure = pressureList[i]
        water = FullMorphologyHydrophobicStep(distanceMap,2*gamma/(pressure*voxelLength),inletVoxels)
        myImg[water] = int(100+pressure/100)
        del water  
    
      
    tff.imsave(outputFileName,myImg.astype(np.uint8))
    


#----------------------------------------------------------------------------------------------
    
if __name__ == "__main__":
    main()