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
def Resize(image,scale=(2,2,2),interpolator='NearestNeighbor'):  
    
    if interpolator=='NearestNeighbor':
        sitkInterpolator =   sitk.sitkNearestNeighbor
        
    elif interpolator=='Linear':
        sitkInterpolator = sitk.sitkLinear
        
    elif interpolator == 'BSpline':
        sitkInterpolator = sitk.sitkBSpline


    sitkImage = sitk.GetImageFromArray(image)

    dim = sitkImage.GetDimension()        
    
    transform = sitk.Transform(dim,sitk.sitkScale)
    transform.SetParameters(scale)    
       
    
    inputSize,inputSpacing,inputOrigin = ( sitkImage.GetSize(),
                                    sitkImage.GetSpacing(),sitkImage.GetOrigin() )
    
    outputSize = [int(inputSize[i]/float(scale[i])) for i in range(dim)]
    outputSpacing = [inputSpacing[i] for i in range(dim)] 
    outputOrigin =  [inputOrigin[i]/float(scale[i]) for i in range(dim)]   
    
    
    resampleFilter = sitk.ResampleImageFilter()
    resampleFilter.SetTransform(transform)
    resampleFilter.SetInterpolator(sitkInterpolator)   
    resampleFilter.SetSize(outputSize)
    resampleFilter.SetOutputSpacing(outputSpacing)    
    resampleFilter.SetOutputOrigin(outputOrigin)
    resampleFilter.SetDefaultPixelValue(0) 
    outputSitkImage = resampleFilter.Execute(sitkImage)
    
    #Shrink is another sitk function to reduce image size using BSPline interpolation
    #outputSitkImage = sitk.Shrink(sitkImage,[Nx,Ny,Nz])
    
    outputImage = sitk.GetArrayFromImage(outputSitkImage)
    
    return outputImage
    
    
    
    