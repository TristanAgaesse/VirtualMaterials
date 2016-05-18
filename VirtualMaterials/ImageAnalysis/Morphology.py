# -*- coding: utf-8 -*-
import SimpleITK as sitk
import numpy as np


#----------------------------------------------------------------------------------------------    
def Dilation(image,ballRadius):
    """ITK morphology binary dilation
    :param image : numpy image (boolean)   
    :param: ballRadius : radius of the structuring element (in voxels)
    :return: dilatedImage (numpy array) 
    
    :Example:    
    import VirtualMaterials as vmat
    ballRadius = 1
    dilatedImage = vmat.ImageAnalysis.Morphology.Dilation(image,ballRadius)
    """

    itkImage = sitk.GetImageFromArray(image.astype(np.uint8))
    foregroundValue = 1.0   
    backgroundValue = 0.0
    boundaryToForeGround=False
    itkDilated = sitk.BinaryDilate(itkImage, int(ballRadius), sitk.sitkBall,
                        backgroundValue, foregroundValue,  boundaryToForeGround)
    del itkImage 
    dilatedImage=sitk.GetArrayFromImage(itkDilated)
    del itkDilated
    dilatedImage=dilatedImage.astype(np.bool) 

    return dilatedImage
    
#----------------------------------------------------------------------------------------------    
def Erosion(image,ballRadius):
    """ITK morphology binary erosion
    :param image : numpy image (boolean)   
    :param: ballRadius : radius of the structuring element (in voxels)
    :return: erodedImage (numpy array) 
    
    :Example:    
    import VirtualMaterials as vmat
    ballRadius = 1
    erodedImage = vmat.ImageAnalysis.Morphology.Erosion(image,ballRadius)
    """

    itkImage = sitk.GetImageFromArray(image.astype(np.uint8))
    foregroundValue = 1.0   
    backgroundValue = 0.0
    boundaryToForeGround=True
    itkEroded = sitk.BinaryErode(itkImage, int(ballRadius), sitk.sitkBall,
                        backgroundValue, foregroundValue,  boundaryToForeGround) 
    del itkImage                    
    erodedImage=sitk.GetArrayFromImage(itkEroded)  
    del itkEroded
    erodedImage=erodedImage.astype(np.bool) 

    return erodedImage

#----------------------------------------------------------------------------------------------    
def FastDilation(image,structuringElement):    
    """Custom implementation of morphology dilation
    :param image : numpy image    
    :param: structuringElement
    :return: dilatedImage (numpy array) 
    
    :Example:    
    import VirtualMaterials as vmat
    structuringElement = np.ones((3,3,3),dtype=np.bool)
    dilatedImage = vmat.ImageAnalysis.Morphology.FastDilation(image,structuringElement)
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
    

#----------------------------------------------------------------------------   
def DistanceMap(image):
    """ITK Danielson distance map
    :param image : numpy image (boolean)   
    :return: distanceMap (numpy array)  
    
    :Example:    
    import VirtualMaterials as vmat
    distanceMap = vmat.ImageAnalysis.Morphology.DistanceMap(image)
    """

    memoryType=np.float16
    itkimage = sitk.GetImageFromArray(np.logical_not(image).astype(np.uint8))
    itkdistanceMap = sitk.DanielssonDistanceMap( itkimage )
    del itkimage
    distanceMap=sitk.GetArrayFromImage(itkdistanceMap).astype(memoryType) 
    del itkdistanceMap

    return distanceMap
    