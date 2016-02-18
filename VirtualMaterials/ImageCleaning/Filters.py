# -*- coding: utf-8 -*-
import SimpleITK as sitk
from VirtualMaterials.Utilities import Utilities as utils  
#Apply filters to the image using the SimpleITK library

#---------------------------------------------------
def MedianFilter(image,radius = (3,3,3)):
    """MedianFilter : see SimpleITK.Median
    :param image : numpy image
    :param radius: size of neighborhood used to take median
    :return: numpy image after filtering
    
    :Example:    
    import VirtualMaterials as vmat
    filteredImage = vmat.ImageCleaning.Filters.MedianFilter(image,radius = (3,3,3))
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
    filteredImage = vmat.ImageCleaning.Filters.Threshold(image, lower=0.0,upper=1.0,outsideValue=0.0)
    """
    
    assert(image.min()>=0)
    memoryType = utils.BestMemoryType(image.max())    
    itkimage = sitk.GetImageFromArray(image.astype(memoryType))

    itkimage = sitk.Threshold(image, lower=lower,upper=upper,outsideValue=outsideValue)
    
    image=sitk.GetArrayFromImage(itkimage).astype(memoryType) 
    return image
    

#---------------------------------------------------
def OtsuThreshold(image,
               maskImage,
               insideValue = 100,
               outsideValue = 0,
               numberOfHistogramBins = 128,
               maskOutput = True,
               maskValue = 255   ):
    """OtsuThreshold : see SimpleITK.OtsuThreshold
    :param image : numpy image
    :param maskImage: 
    :param insideValue: 
    :param outsideValue: 
    :param numberOfHistogramBins: 
    :param maskOutput: 
    :param maskValue: 
    :return: numpy image after filtering
    
    :Example:    
    import VirtualMaterials as vmat
    filteredImage = vmat.ImageCleaning.Filters.OtsuThreshold(image,
               maskImage,
               insideValue = 100,
               outsideValue = 0,
               numberOfHistogramBins = 128,
               maskOutput = True,
               maskValue = 255   )
    """
    
    
    assert(image.min()>=0)
    memoryType = utils.BestMemoryType(image.max())    
    itkimage = sitk.GetImageFromArray(image.astype(memoryType))
    
    #OTSU Threshold FILTER
    itkimage= sitk.OtsuThreshold(itkimage,
                       maskImage,
                       insideValue = 100,
                       outsideValue = 0,
                       numberOfHistogramBins = 128,
                       maskOutput = True,
                       maskValue = 255 
                       )		
                       

    image=sitk.GetArrayFromImage(itkimage).astype(memoryType) 
    return image
                       
#---------------------------------------------------
def AnisotropicDiffusionFilter(image,
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
    filteredImage = vmat.ImageCleaning.Filters.AnisotropicDiffusionFilter(image,
                          timeStep = 0.125,
                          conductanceParameter=3.0,
                          conductanceScalingUpdateInterval=1,
                          numberOfIterations = 20 )
    """          

    assert(image.min()>=0)
    memoryType = utils.BestMemoryType(image.max())    
    itkimage = sitk.GetImageFromArray(image.astype(memoryType))
    
    itkimage = sitk.GradientAnisotropicDiffusion(itkimage,
                                      timeStep = 0.125,
                                      conductanceParameter=3.0,
                                      conductanceScalingUpdateInterval=1,
                                      numberOfIterations = 20
                                      )
               
    image=sitk.GetArrayFromImage(itkimage).astype(memoryType) 
    return image         
               
               
               