# -*- coding: utf-8 -*-


    
from VirtualMaterials.Utilities  import tifffile as tff
import numpy as np
import vtk
    
    
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
    
    
    