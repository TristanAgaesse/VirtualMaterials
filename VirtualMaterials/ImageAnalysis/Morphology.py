# -*- coding: utf-8 -*-
import SimpleITK as sitk
import numpy as np


#----------------------------------------------------------------------------------------------    
def FastDilation(image,structuringElement):    
    """Custom implementation of morphology dilation
    :param image : numpy image    
    :param: structuringElement
    :return: numpy image 
    
    :Example:    
    import VirtualMaterials as vmat
    structuringElement = np.ones((3,3,3),dtype=np.bool)
    sobelEdges = vmat.ImageAnalysis.Morphology.FastDilation(image,structuringElement)
    """
    structuringElement= np.asarray(structuringElement)
    center = structuringElement.shape[0]//2
    image=image.astype(np.bool)    
    
    biggerImage = np.zeros(np.asarray(image.shape)+2*center,dtype=np.bool)
    dilatedImage = np.zeros(biggerImage.shape,dtype=np.bool) 
    
    dim=image.ndim    
    
    #smallImageIndices=(center:-center-1,center:-center-1,center:-center-1)
#    smallImageIndices = [np.arange(center,center+image.shape[iDim]) 
#                            for iDim in range(dim)]
    
    smallImageIndices = [slice(center,center+image.shape[iDim]) 
                            for iDim in range(dim)]
    
    biggerImage[smallImageIndices] = image
    
    #X,Y,Z = np.nonzero(biggerImage)
    nnzIndices=np.nonzero(biggerImage)
    oneColumn=np.ones(nnzIndices[0].size,dtype=np.int)        
    
    for iSE in range(structuringElement.size):        
        #xIse,yIse,zIse = np.unravel_index(iSE,structuringElement.shape)
        iSEposition = np.unravel_index(iSE,structuringElement.shape)
        #if structuringElement[xIse,yIse,zIse]:
#            shiftX,shiftY,shiftZ = xIse-center,yIse-center,zIse-center
#            dilatedImage[X+shiftX*oneColumn,Y+shiftY*oneColumn,Z+shiftZ*oneColumn]=True
        if structuringElement[iSEposition]:    
            positionTrue=[]
            for iDim in range(dim):
                shift=iSEposition[iDim]-center
                positionTrue.append( nnzIndices[iDim]+shift*oneColumn )
                  
            dilatedImage[positionTrue]=True
                  
    dilatedImage = dilatedImage[smallImageIndices]
    
    
    return dilatedImage
    





    