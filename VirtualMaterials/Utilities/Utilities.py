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
def WriteTiff(image,filename):
    """ :param : image,filename """
    
    tff.imsave(filename,image.astype(np.uint8))

#--------------------------------------------------------------------
def ReadTiff(filename):
    """ :param : filename """
    
    image=tff.imread(filename)
    
    return image

#--------------------------------------------------------------------
def ReadVTKFile(filename):

    reader = vtk.vtkPolyDataReader()
    reader.SetFileName(filename)
    reader.Update()
    polydata = reader.GetOutput()

    return polydata

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
def DistanceMap(image):   
    """ Returns the Danielsson Distance Map of the image : for each 
    zero voxel, the min distance to a non zero voxel"""
    
    memoryType=np.float16
    itkimage = sitk.GetImageFromArray(image.astype(np.uint8))
    itkdistanceMap = sitk.DanielssonDistanceMap( itkimage )
    del itkimage
    distanceMap=sitk.GetArrayFromImage(itkdistanceMap).astype(memoryType) 
    del itkdistanceMap 
    
    return distanceMap
    
    
#--------------------------------------------------------------------    
def CoveringRadiusMap(image):    
    """ Returns the Covering Radius Map of the image """
    distanceMap = DistanceMap(image)
    
    
    coveringRadiusMap = 1
    return coveringRadiusMap

#--------------------------------------------------------------------
def NumpyToVTKImage(numpyImage): 

    dataImporter = vtk.vtkImageImport()
    
    shape=numpyImage.shape
    shape=[int(shape[i]) for i in range(len(shape))]
    w, d, h = shape
    dataImporter.SetDataExtent(0, h-1, 0, d-1, 0, w-1)
    dataImporter.SetWholeExtent(0, h-1, 0, d-1, 0, w-1)
    
    if numpyImage.dtype == np.uint8 :
        dataImporter.SetDataScalarTypeToUnsignedChar()
    elif numpyImage.dtype == np.int16 :
        dataImporter.SetDataScalarTypeToShort()
    elif numpyImage.dtype == np.int32 :
        dataImporter.SetDataScalarTypeToInt()        
    elif numpyImage.dtype == np.uint16 :
        dataImporter.SetDataScalarTypeToUnsignedShort()    
    elif numpyImage.dtype == np.float32 :
        dataImporter.SetDataScalarTypeToFloat()    
    elif numpyImage.dtype == np.float64 :
        dataImporter.SetDataScalarTypeToDouble()
    else:
        numpyImage=numpyImage.astype(np.uint8)
        dataImporter.SetDataScalarTypeToUnsignedChar()
        
#np_vtk = {numpy.character:vtk.VTK_UNSIGNED_CHAR,
#                numpy.uint8:vtk.VTK_UNSIGNED_CHAR,
#                numpy.uint16:vtk.VTK_UNSIGNED_SHORT,
#                numpy.uint32:vtk.VTK_UNSIGNED_INT,
#                numpy.uint64:vtk.VTK_UNSIGNED_LONG_LONG,
#                numpy.int8:vtk.VTK_CHAR,
#                numpy.int16:vtk.VTK_SHORT,
#                numpy.int32:vtk.VTK_INT,
#                numpy.int64:vtk.VTK_LONG_LONG,
#                numpy.float32:vtk.VTK_FLOAT,
#                numpy.float64:vtk.VTK_DOUBLE,
#                numpy.complex64:vtk.VTK_FLOAT,
#                numpy.complex128:vtk.VTK_DOUBLE}

#    vtk_typecode = numpy_support.get_vtk_array_type(numpyImage.dtype)
#    dataImporter.SetDataScalarType(vtk_typecode)	
        
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
       
    
#--------------------------------------------------------------------    
def BestMemoryType(number):   
    
    if number>=0:
        if abs(number)<np.iinfo(np.uint8).max:
            memoryType=np.uint8
        elif abs(number)<np.iinfo(np.uint16).max:
            memoryType=np.uint16
        elif abs(number)<np.iinfo(np.uint32).max:
            memoryType=np.uint32     
        else:
            memoryType=np.uint64   

    else:
        if abs(number)<np.iinfo(np.int8).max:
            memoryType=np.int8
        elif abs(number)<np.iinfo(np.int16).max:
            memoryType=np.int16
        elif abs(number)<np.iinfo(np.int32).max:
            memoryType=np.int32     
        else:
            memoryType=np.int64        
        
        
    return memoryType     
            
            
            