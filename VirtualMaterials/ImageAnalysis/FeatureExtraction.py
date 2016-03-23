# -*- coding: utf-8 -*-
import SimpleITK as sitk
import numpy as np

#-----------------------------------------------------------------------------
def SobelEdgeDetection(image):    
    """Uses ITK sobel edge detector to detect boundaries between phases
    :param image : numpy image    
    :return: numpy image of the sobel edges
    
    :Example:    
    import VirtualMaterials as vmat
    sobelEdges = vmat.ImageAnalysis.FeatureExtraction.SobelEdgeDetection(image)
    """
        
    myItkImage = sitk.GetImageFromArray(image.astype(np.uint8))
    caster = sitk.CastImageFilter()
    caster.SetOutputPixelType(sitk.sitkFloat32)
    floatImage = caster.Execute( myItkImage )

    itkEdges = sitk.SobelEdgeDetection(floatImage)
    sobelEdges = sitk.GetArrayFromImage(itkEdges).astype(np.bool)

    del myItkImage,floatImage,itkEdges  

    return sobelEdges


#-----------------------------------------------------------------------------
def CannyEdgeDetection(image,variance=2):       
    """Uses ITK Canny edge detector to detect boundaries between phases. Canny 
    edge detector is the following :
      - Apply Gaussian filter to smooth the image in order to remove the noise
      - Find the intensity gradients of the image
      - Apply non-maximum suppression to get rid of spurious response to edge detection
      - Apply double threshold to determine potential edges
      - Track edge by hysteresis: Finalize the detection of edges by suppressing all the other edges that are weak and not connected to strong edges.
    
    :param image : numpy image  
    :param : variance : width of the gaussian filter  
    :return: numpy image of the sobel edges
    
    :Example:    
    import VirtualMaterials as vmat
    sobelEdges = vmat.ImageAnalysis.FeatureExtraction.CannyEdgeDetection(image,variance=2)
    """

    
    #Load image into SimpleITK
    myItkImage = sitk.GetImageFromArray(image.astype(np.uint8))
    caster = sitk.CastImageFilter()
    caster.SetOutputPixelType(sitk.sitkFloat32)
    floatImage = caster.Execute( myItkImage )
    
    #Canny edge detection
    canny = sitk.CannyEdgeDetectionImageFilter()
    variance=float(variance)
    canny.SetVariance( [ variance,variance,variance] )
    #canny.SetLowerThreshold( 10 )
    #canny.SetUpperThreshold( 1000 ) 
    myItkImage = canny.Execute( floatImage )
    
    #Go back to a numpy array image
    caster = sitk.CastImageFilter()
    caster.SetOutputPixelType(sitk.sitkInt8)
    myItkImage = caster.Execute( myItkImage )
    cannyEdges = sitk.GetArrayFromImage(myItkImage).astype(np.bool)

    del myItkImage, caster, canny, floatImage  

    return cannyEdges
