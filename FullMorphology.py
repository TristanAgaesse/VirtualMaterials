# -*- coding: utf-8 -*-
"""
Created on Thu Nov 27 19:53:34 2014

@author: greentoto
"""

import numpy as np
import mahotas
import skimage
import tifffile as tff
from scipy import ndimage



def main():
    
    foldername ='/home/ta240184/Documents/MATLAB/PNM_Matlab_V0.1/Simulations/PSI_drainage_python/'
    inputFileName = foldername+"PSI_sampleDrainage.tif"
    
    myImg=tff.imread(inputFileName)
    
    inletVoxels = myImg[0,:,:]   
    
    distanceMap = ndimage.distance_transform_cdt(np.logical_not(myImg),metric='chessboard').astype(np.int8)
    
    saturation=np.zeros(distanceMap.max())
    
    for iStep in range(1,distanceMap.max()+1):
        saturation[iStep-1]=np.count_nonzero(FullMorphologyHydrophobicStep(distanceMap,iStep,inletVoxels))


    return saturation



#----------------------------------------------------------------------------------------------

def FullMorphologyHydrophobicStep(distanceMap,capillaryLength,inletVoxels):
    
    invadedVoxels=distanceMap[distanceMap>capillaryLength].astype(np.bool)    
    
    ball=skimage.morphology.ball(capillaryLength)
    invadedVoxels=mahotas.morph.dilate(invadedVoxels,ball)
    
    invadedVoxels=skimage.measure.label(invadedVoxels)
    
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