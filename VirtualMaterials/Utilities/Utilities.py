# -*- coding: utf-8 -*-
    
from VirtualMaterials.Utilities  import tifffile as tff
import numpy as np
import vtk
import SimpleITK as sitk    
    
#--------------------------------------------------------------------
#      Utilities
#--------------------------------------------------------------------    
    
#--------------------------------------------------------------------
def SaveImageTiff(image,filename):
    tff.imsave(filename,image.astype(np.uint8))

#--------------------------------------------------------------------
def LoadImageTiff(filename):
    
    image=tff.imread(filename)
    
    return image


#--------------------------------------------------------------------
def ExtractSurface(image):
        
        
    #Convert numpy array to vtk image
    shape=image.shape
    shape=[int(shape[i]) for i in range(len(shape))]
        
    dataImporter = vtk.vtkImageImport()
    dataImporter.CopyImportVoidPointer(image, image.nbytes)
    dataImporter.SetDataScalarTypeToUnsignedChar()
    dataImporter.SetNumberOfScalarComponents(1)
    w, d, h = shape
    dataImporter.SetDataExtent(0, h-1, 0, d-1, 0, w-1)
    dataImporter.SetWholeExtent(0, h-1, 0, d-1, 0, w-1)
    dataImporter.Update() 
    
    #Use VTK Marching Cubes algorithm
    dmc = vtk.vtkDiscreteMarchingCubes()
    dmc.SetInputConnection(dataImporter.GetOutputPort())
    dmc.GenerateValues(1, 1, 1)
    dmc.Update()
    
    return dmc.GetOutput()
    
    
#--------------------------------------------------------------------
def Resize(image,scale=(2,2,2),interpolatorType='NearestNeighbor'):  
    
    sitkImage = sitk.GetImageFromArray(image)
    
    interpolator =   sitk.sitkNearestNeighbor
#            sitk.sitkLinear
#            sitk.sitkBSpline

    dim = sitkImage.GetDimension()        
    
    transform = sitk.Transform(dim,sitk.sitkScale)
    transform.SetParameters(scale)    
       
#    inputSize = sitkImage.GetSize() 
#    oldNx,oldNy,oldNz = inputSize  
#    outputSize = inputSize
    
    Nx,Ny,Nz=scale
    
    inputSize,inputSpacing,inputOrigin = image.GetSize(),image.GetSpacing(), image.GetOrigin()
    outputSize = [int(n/scale) for n in inputSize]
    outputSpacing = [n*scale for n in sp1] 
    outputOrigin =  [inputOrigin[i]/float(scale[i]) for i in range(2)]   
    
    inputSpacing = sitkImage.GetSpacing()
    sx = inputSpacing[0] * oldNx / float(Nx)
    sy = inputSpacing[1] * oldNy / float(Ny)
    sz = inputSpacing[2] * oldNz / float(Nz)
    outputSpacing = sx, sy, sz
    
#    inputOrigin=sitkImage.GetOrigin()
#    ox = inputOrigin[0] / float(Nx)
#    oy = inputOrigin[1] / float(Ny)
#    oz = inputOrigin[2] / float(Nz)    
#    outputOrigin = ox,oy,oz
    
    resampleFilter = sitk.ResampleImageFilter()
    
    #resampleFilter.SetInput(sitkImage)
    resampleFilter.SetTransform(transform)
    resampleFilter.SetInterpolator(interpolator)   
    resampleFilter.SetSize(outputSize)
    resampleFilter.SetOutputSpacing(outputSpacing)    
    resampleFilter.SetOutputOrigin(outputOrigin)
    
    resampleFilter.SetDefaultPixelValue(0) 
    #resampleFilter.Update()     
   
    outputSitkImage = resampleFilter.Execute(sitkImage)
    
#    origin = sitkImage.GetOrigin()
#    direction = sitkImage.GetDirection()
#    defaultPixelValue=0.0
#    
#    outputSitkImage = sitk.Resample(image,outputSize,transform,interpolator,
#                                    origin,outputSpacing,direction,
#                                    defaultPixelValue,sitk.sitkFloat32) 
    
    #outputSitkImage = sitk.Shrink(sitkImage,[Nx,Ny,Nz])
    
    
    return sitk.GetArrayFromImage(outputSitkImage)
    
    
    
    
        #outputSitkImage = sitk.Resample(sitkImage,transform,interpolator,defaultPixelValue)

    
#    PixelType = itk.UC
#    ScalarType = itk.D
#    Dimension = 2
#    
#    ImageType = itk.Image[PixelType, Dimension]
#    
#    scale = 
#    
#    inputImage = 
#    
#    size = inputImage.GetLargestPossibleRegion().GetSize()
#    spacing = inputImage.GetSpacing()
#    
#    centralPixel = itk.Index[Dimension]()
#    centralPixel[0] = size[0] / 2
#    centralPixel[1] = size[1] / 2
#    centralPoint = itk.Point[ScalarType, Dimension]()
#    centralPoint[0] = centralPixel[0]
#    centralPoint[1] = centralPixel[1]
#    
#    scaleTransform = itk.ScaleTransform[ScalarType, Dimension].New()
#    
#    parameters = scaleTransform.GetParameters()
#    parameters[0] = scale
#    parameters[1] = scale
#    
#    scaleTransform.SetParameters(parameters)
#    scaleTransform.SetCenter(centralPoint)
#    
#    interpolatorType = itk.LinearInterpolateImageFunction[ImageType, ScalarType]
#    interpolator = interpolatorType.New()
#    
#    resamplerType = itk.ResampleImageFilter[ImageType, ImageType]
#    resampleFilter = resamplerType.New()
#    
#    resampleFilter.SetInput(inputImage)
#    resampleFilter.SetTransform(scaleTransform)
#    resampleFilter.SetInterpolator(interpolator)
#    resampleFilter.SetSize(size)
#    resampleFilter.SetOutputSpacing(spacing)    
##        
##        
#http://itk.org/ITKExamples/src/Filtering/ImageGrid/ResampleAnImage/Documentation.html
#http://itk-users.7.n7.nabble.com/SimpleITK-Downsampling-results-in-blank-image-td34923.html
#http://www.itk.org/Wiki/ITK/Examples/ImageProcessing/ResampleImageFilter
#http://erie.nlm.nih.gov/~blowek1/SimpleITK/master/classitk_1_1simple_1_1Transform.html
#http://simpleitk.github.io/SimpleITK-Notebooks/21_Transform_and_Resampling.html
#http://itk-users.7.n7.nabble.com/SimpleITK-Downsampling-results-in-blank-image-td34923.html
#http://insightsoftwareconsortium.github.io/SimpleITK-Notebooks/55_VH_Resample.html