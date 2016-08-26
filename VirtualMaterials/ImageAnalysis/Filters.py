# -*- coding: utf-8 -*-
import SimpleITK as sitk
from VirtualMaterials.Utilities import Utilities as utils  
import numpy as np
#Apply filters to the image using the SimpleITK library

#---------------------------------------------------
def MedianFilter(image,radius = (3,3,3)):
    """MedianFilter : see SimpleITK.Median
    :param image : numpy image
    :param radius: size of neighborhood used to take median
    :return: numpy image after filtering
    
    :Example:    
    import VirtualMaterials as vmat
    filteredImage = vmat.ImageAnalysis.Filters.MedianFilter(image,radius = (3,3,3))
    """
        
    assert(image.min()>=0)
    memoryType = utils.BestMemoryType(image.max())    
    itkimage = sitk.GetImageFromArray(image.astype(memoryType))

    itkimage = sitk.Median(itkimage,radius = radius)
	
    image=sitk.GetArrayFromImage(itkimage).astype(memoryType) 
    return image
    
#---------------------------------------------------
def Threshold(image, lower=0.0,upper=1.0,outsideValue=0.0):
    """Threshold : see SimpleITK.Threshold
    :param image : numpy image
    :param lower: 
    :param upper: 
    :param outsideValue: 
    :return: numpy image after filtering
    
    :Example:    
    import VirtualMaterials as vmat
    filteredImage = vmat.ImageAnalysis.Filters.Threshold(image, lower=0.0,upper=1.0,outsideValue=0.0)
    """
    
    assert(image.min()>=0)
    memoryType = utils.BestMemoryType(image.max())    
    itkimage = sitk.GetImageFromArray(image.astype(memoryType))

    itkimage = sitk.Threshold(itkimage, 
                              lower=lower,
                              upper=upper,
                              outsideValue=outsideValue)
    
    image=sitk.GetArrayFromImage(itkimage).astype(memoryType) 
    return image
    

#---------------------------------------------------
def OtsuThreshold(image,
               insideValue = 100,
               outsideValue = 0,
               numberOfHistogramBins = 128,
               maskOutput = True,
               maskValue = 255   ):
    """OtsuThreshold : see SimpleITK.OtsuThreshold
    :param image : numpy image
    :param insideValue: 
    :param outsideValue: 
    :param numberOfHistogramBins: 
    :param maskOutput: 
    :param maskValue: 
    :return: numpy image after filtering
    
    :Example:    
    import VirtualMaterials as vmat
    filteredImage = vmat.ImageAnalysis.Filters.OtsuThreshold(image,
               insideValue = 100,
               outsideValue = 0,
               numberOfHistogramBins = 128,
               maskOutput = True,
               maskValue = 255   )
    """
    
    
    assert(image.min()>=0)
    memoryType = memoryType = utils.BestMemoryType(image.max())    
    itkimage = sitk.GetImageFromArray(image.astype(memoryType))
       
    #OTSU Threshold FILTER
    itkimage= sitk.OtsuThreshold(
                       itkimage,
                       np.uint8(insideValue),
                       np.uint8(outsideValue),
                       np.uint32(numberOfHistogramBins),
                       bool(maskOutput),
                       np.uint8(maskValue) 
                       )
                       
    image=sitk.GetArrayFromImage(itkimage).astype(memoryType) 
    return image
                       
#---------------------------------------------------
def AnisotropicDiffusionFilter_Gradient(image,
                          timeStep = 0.125,
                          conductanceParameter=3.0,
                          conductanceScalingUpdateInterval=1,
                          numberOfIterations = 20 ):
    """AnisotropicDiffusionFilter : see SimpleITK.GradientAnisotropicDiffusion
    :param image : numpy image
    :param : timseStep
    :param : conductanceParameter
    :param : conductanceScalingUpdateInterval
    :param : numberOfIterations
    :return: numpy image after filtering
    
    :Example:    
    import VirtualMaterials as vmat
    filteredImage = vmat.ImageAnalysis.Filters.AnisotropicDiffusionFilter_Gradient(image,
                          timeStep = 0.125,
                          conductanceParameter=3.0,
                          conductanceScalingUpdateInterval=1,
                          numberOfIterations = 20 )
    """          
    
    #https://arxiv.org/pdf/1503.00992.pdf
    assert(image.min()>=0)
    itkimage = sitk.GetImageFromArray(image.astype(np.uint8))
    
    caster = sitk.CastImageFilter()
    caster.SetOutputPixelType(sitk.sitkFloat32)
    floatImage = caster.Execute( itkimage )
    
    itkimage = sitk.GradientAnisotropicDiffusion(floatImage,
                      timeStep = timeStep,
                      conductanceParameter=conductanceParameter,
                      conductanceScalingUpdateInterval=conductanceScalingUpdateInterval,
                      numberOfIterations = numberOfIterations
                      )
               
    image=sitk.GetArrayFromImage(itkimage).astype(np.uint8) 
    return image         
               
               
#---------------------------------------------------
def AnisotropicDiffusionFilter_Curvature(image,
                          timeStep = 0.0625,
                          conductanceParameter=3.0,
                          conductanceScalingUpdateInterval=1,
                          numberOfIterations = 5 ):
    """AnisotropicDiffusionFilter : see SimpleITK.CurvatureAnisotropicDiffusion
    :param image : numpy image
    :param : timseStep
    :param : conductanceParameter
    :param : conductanceScalingUpdateInterval
    :param : numberOfIterations
    :return: numpy image after filtering
    
    :Example:    
    import VirtualMaterials as vmat
    filteredImage = vmat.ImageAnalysis.Filters.AnisotropicDiffusionFilter_Curvature(image,
                          timeStep = 0.0625,
                          conductanceParameter=3.0,
                          conductanceScalingUpdateInterval=1,
                          numberOfIterations = 5 )
    """          
    
    #https://arxiv.org/pdf/1503.00992.pdf
    assert(image.min()>=0)
    itkimage = sitk.GetImageFromArray(image.astype(np.uint8))
    
    caster = sitk.CastImageFilter()
    caster.SetOutputPixelType(sitk.sitkFloat32)
    floatImage = caster.Execute( itkimage )
    
    itkimage = sitk.CurvatureAnisotropicDiffusion(floatImage,
                      timeStep = timeStep,
                      conductanceParameter=conductanceParameter,
                      conductanceScalingUpdateInterval=conductanceScalingUpdateInterval,
                      numberOfIterations = numberOfIterations
                      )
               
    image=sitk.GetArrayFromImage(itkimage).astype(np.uint8) 
    return image                  