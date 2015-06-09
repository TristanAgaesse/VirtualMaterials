# -*- coding: utf-8 -*-
    
from VirtualMaterials.Utilities  import tifffile as tff
import numpy as np
import vtk
import SimpleITK as sitk    
from vtk.util import numpy_support    

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
    """ interpolator= 'NearestNeighbor', 'Linear' or 'BSpline'  """
    
    
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
    
    

#--------------------------------------------------------------------
def NumpyToVTKImage(numpyImage): 

    dataImporter = vtk.vtkImageImport()
    
    shape=numpyImage.shape
    shape=[int(shape[i]) for i in range(len(shape))]
    w, d, h = shape
    dataImporter.SetDataExtent(0, h-1, 0, d-1, 0, w-1)
    dataImporter.SetWholeExtent(0, h-1, 0, d-1, 0, w-1)
    
    numpyImage=numpyImage.astype(np.uint8)
    dataImporter.SetDataScalarTypeToUnsignedChar()
    
    
    
    dataImporter.CopyImportVoidPointer(numpyImage, numpyImage.nbytes)
    dataImporter.SetNumberOfScalarComponents(1)
        
    dataImporter.Update()
    vtkImage = dataImporter.GetOutput()

    return vtkImage



#--------------------------------------------------------------------
def VTKImageToNumpy(vtkImage): 
    
    vtkImageBis = vtkImage.NewInstance()
    vtkImageBis.DeepCopy(vtkImage)    
    
    vtkarray = vtkImageBis.GetPointData().GetScalars()
    numpyImage = numpy_support.vtk_to_numpy(vtkarray)
    
    h1, h2, d1, d2, w1, w2 = vtkImageBis.GetExtent()
    numpyImageShape = (w2-w1+1,d2-d1+1,h2-h1+1)
    numpyImage = numpyImage.reshape(numpyImageShape)
    
    return numpyImage    
       
    
    
    
    