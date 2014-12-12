# -*- coding: utf-8 -*-
"""
Created on Thu Nov 27 19:53:34 2014

@author: greentoto
"""

import numpy as np
from skimage import morphology
from scipy import ndimage
import sys

foldername ='/home/270.12-Modeling_PEMFC_Li/theseTristan/PSI_drainage_python/'
sys.path.append(foldername)

import tifffile as tff

def main(inputFileName,outputFilename,voxelLength):
        
    #inputFileName = foldername+"PSI_sampleDrainage_2540.tif"
    
    myImg=tff.imread(inputFileName)
    
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
    
    #voxelLength=(2.2e-6)*2540/float(2540)
    gamma=72e-3
    
    
    pressure = 53*100
    water = FullMorphologyHydrophobicStep(distanceMap,2*gamma/(pressure*voxelLength),inletVoxels)
    myImg[water] = 153
    
    pressure = 39*100
    water = FullMorphologyHydrophobicStep(distanceMap,2*gamma/(pressure*voxelLength),inletVoxels)
    myImg[water] = 139
    
    pressure = 28*100
    water = FullMorphologyHydrophobicStep(distanceMap,2*gamma/(pressure*voxelLength),inletVoxels)
    myImg[water] = 128
    
    pressure = 22*100
    water = FullMorphologyHydrophobicStep(distanceMap,2*gamma/(pressure*voxelLength),inletVoxels)
    myImg[water] = 122
    
    pressure = 14*100
    water = FullMorphologyHydrophobicStep(distanceMap,2*gamma/(pressure*voxelLength),inletVoxels)
    myImg[water] = 114
    
    tff.imsave(outputFileName,myImg)
    

#----------------------------------------------------------------------------------------------

def FullMorphologyHydrophobicStep(distanceMap,capillaryLength,inletVoxels):
    
    indicesInvaded=(distanceMap>capillaryLength).reshape(distanceMap.shape)
    invadedVoxels=np.zeros(distanceMap.shape,dtype=bool)
    invadedVoxels[indicesInvaded]=True
    
    
    ball=morphology.ball(capillaryLength)
    #invadedVoxels=mahotas.morph.dilate(invadedVoxels,ball)
    invadedVoxels=morphology.binary_dilation(invadedVoxels, selem=ball)    
              
              
    structuringElement = np.ones((3,3,3))
    invadedVoxels=ndimage.measurements.label(invadedVoxels,structuringElement)[0]
    
    labelsToKeep=np.unique(invadedVoxels[inletVoxels])    
    labelsToKeep=labelsToKeep[labelsToKeep>0]
    
    foo=np.zeros(invadedVoxels.shape,dtype=bool)
    
    for iLabel in labelsToKeep :
        foo[invadedVoxels==iLabel] = True
    
    invadedVoxels=invadedVoxels.astype(np.bool)
    invadedVoxels[np.logical_not(foo)]=False
    invadedVoxels[foo]=True
    
    
    return invadedVoxels
    


    
#----------------------------------------------------------------------------------------------
    
if __name__ == "__main__":
    main()