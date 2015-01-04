# -*- coding: utf-8 -*-
"""
Created on Fri Jan  2 18:50:02 2015

@author: greentoto
"""
import numpy as np
import vtk
from vtk.util import numpy_support
from scipy import ndimage
import math

import sys
sys.path.append('C:/Users/Tristan/Documents/Python/workspace/Image_Computations/')
import tifffile as tff


def main():
    imageShape = (50,50,50)
    image=np.zeros(imageShape).astype(np.uint8)
    imageBounds=(-1.5, 1.5, -1.5, 1.5, -1.5, 1.5)
    
    gridX
    gridY
    gridZ
    
    center = (0,0,0)
    radius = 0.5
    mesh = CreateSphere(center,radius)
    
    image=Voxelize(mesh,gridX,gridY,gridZ)
    
    
    print(np.count_nonzero(image))
    tff.imsave('test2.tif',image)




#--------------------------------------------------------------------






#--------------------------------------------------------------------
def CreateSphere(center,radius):
    
    sphereModel = vtk.vtkSphereSource()
    sphereModel.SetCenter(center[0],center[1],center[2])
    sphereModel.SetRadius(radius)
    sphereModel.SetThetaResolution(10)
    sphereModel.SetPhiResolution(10)
    mesh=sphereModel.GetOutput()
    mesh.Update()    

    return mesh
    
#--------------------------------------------------------------------
def CreateCylinder(center,radius,height):
    
    cylinderModel = vtk.vtkCylinderSource()
    cylinderModel.SetCenter(center[0],center[1],center[2])
    cylinderModel.SetRadius(radius)
    cylinderModel.SetHeight(height)
    cylinderModel.SetResolution(30)
    
    mesh=cylinderModel.GetOutput()
    mesh.Update()

    return mesh
    
#--------------------------------------------------------------------
def CreatePolyhedron(points):
    
    nPoint = len(points)
#    P0 = [0.0, 0.0, 0.0] 
#    P1 = [1.0, 0.0, 0.0]
     
    # Create the points
    points = vtk.vtkPoints()
    for iPoint in range(nPoint):
        points.InsertNextPoint(points[iPoint])

    # Create a hexahedron from the points
    hex = vtk.vtkHexahedron()
    for iPoint in range(nPoint):
        hex.GetPointIds().SetId(iPoint,iPoint)

    # Add the hexahedron to a cell array
    hexs = vtk.vtkCellArray()
    hexs.InsertNextCell(hex)
    
    # Add the points and hexahedron to an unstructured grid
    uGrid = vtk.vtkUnstructuredGrid()
    uGrid.SetPoints(points)
    uGrid.InsertNextCell(hex.GetCellType(), hex.GetPointIds())
    
    # Convert to vtk polydata object 
    geometryFilter =  vtk.vtkGeometryFilter()
    geometryFilter.SetInput(uGrid)
    geometryFilter.Update()
    polydata = geometryFilter.GetOutput()
    
    polydata.Update()
    
    return polydata
    
#--------------------------------------------------------------------
def CreateEllipsoid(center,xRadius,yRadius,zRadius):

    ellipsoid = vtk.vtkParametricEllipsoid()
    ellipsoid.SetXRadius(xRadius)
    ellipsoid.SetYRadius(yRadius)
    ellipsoid.SetZRadius(zRadius)
    
    source = vtk.vtkParametricFunctionSource()
    source.SetParametricFunction(ellipsoid)
    source.SetUResolution(30)
    source.SetVResolution(30)
    source.SetWResolution(30)
    source.Update()
    polydata = source.GetOutput()
    polydata.Update()
    
    center=(0,0,0)    
    
    
    return polydata
    
#--------------------------------------------------------------------
def CreateRandomHills():

    randomHills = vtk.vtkParametricRandomHills()
    randomHills.AllowRandomGenerationOff()        
    
    source = vtk.vtkParametricFunctionSource()
    source.SetParametricFunction(randomHills)
    source.Update()
    polydata = source.GetOutput()
    polydata.Update()
    
    return polydata
    
#--------------------------------------------------------------------
def CreateSpline():   
    
    npts      = 100
    vtkPoints = vtk.vtkPoints()
    vtkPoints.SetNumberOfPoints(100)
    for i in range (npts):
      x = math.sin(math.pi*i/20.)
      y = math.cos(math.pi*i/20.)
      z = 2*i/float(npts)
      vtkPoints.SetPoint(i, (x,y,z))
    
    vtkCellArray = vtk.vtkCellArray()
    vtkCellArray.InsertNextCell(npts)
    for i in range(npts):
      vtkCellArray.InsertCellPoint(i)
    
    value = lambda i: math.fabs(math.sin(math.pi*i/30.))
    vtkFloatArray = vtk.vtkFloatArray()
    vtkFloatArray.SetNumberOfValues(npts)
    for i in range(npts):
      vtkFloatArray.SetValue(i, value(i))
    
    vtkPolyData = vtk.vtkPolyData()
    vtkPolyData.SetPoints(vtkPoints)
    vtkPolyData.SetLines(vtkCellArray)
    vtkPolyData.GetPointData().SetScalars(vtkFloatArray)
    
    vtkSplineFilter = vtk.vtkSplineFilter()
    vtkSplineFilter.SetInput(vtkPolyData)
    vtkSplineFilter.SetNumberOfSubdivisions(5*npts)
    vtkSplineFilter.Update()
    
    vtkTubeFilter = vtk.vtkTubeFilter()
    vtkTubeFilter.SetInputConnection(vtkSplineFilter.GetOutputPort())
    vtkTubeFilter.SetRadius(0.15)
    vtkTubeFilter.SetNumberOfSides(10)
    vtkTubeFilter.CappingOn()    
    
    polydata = vtkTubeFilter.GetOutput()
    polydata.Update()
    
    return polydata    
    
#--------------------------------------------------------------------
def CreateTorus():
    
    torus = vtk.vtkParametricTorus()

    source = vtk.vtkParametricFunctionSource()
    source.SetParametricFunction(torus)
    source.Update()
    
    polydata = source.GetOutput()
    polydata.Update()
    
    return polydata
    
    
#--------------------------------------------------------------------
def Voxelize(vtkPolyDataObject,gridX,gridY,gridZ):
    
    #Voxelize the object on a window adapted to its bounds
    subGridX
    subGridY
    subGridZ
    
    
    #Use VTK VoxelModel to Voxelize the surface
    voxelModel = vtk.vtkVoxelModeller()
    voxelModel.SetInput(vtkPolyDataObject)
    voxelModel.SetSampleDimensions(sampleDimensions[0],sampleDimensions[1],sampleDimensions[2])
    voxelModel.SetModelBounds(bounds[0],bounds[1],bounds[2],bounds[3],bounds[4],bounds[5])
    voxelModel.SetScalarTypeToChar()
    voxelModel.SetForegroundValue(255)
    voxelModel.SetBackgroundValue(0)
    voxelModel.Update()
    
    voxelizedSurface = numpy_support.vtk_to_numpy(voxelModel.GetOutput().GetPointData().GetScalars())
    voxelizedSurface = voxelizedSurface.reshape(sampleDimensions).astype(np.uint8)
    
#    insideVoxel=( math.ceil(sampleDimensions[0]*(insidePoint[0]-bounds[0])/(bounds[1]-bounds[0])), 
#                  math.ceil(sampleDimensions[1]*(insidePoint[1]-bounds[2])/(bounds[3]-bounds[2])), 
#                  math.ceil(sampleDimensions[2]*(insidePoint[2]-bounds[4])/(bounds[5]-bounds[4]))
#                  )

    subImage=FillInside(voxelizedSurface)
    
    #Get back to the window of the original image
    objectImage    
    
    return objectImage
    
#--------------------------------------------------------------------    
def FillInside(voxelizedSurface):    
    
    surface = voxelizedSurface==255
    sampleDimensions = voxelizedSurface.shape
    image=np.zeros(sampleDimensions).astype(np.uint8)
    
    correctionLIST = []
    zVoxels=range(sampleDimensions[2])
    
    for ix in range(sampleDimensions[0]):
        for iy in range(sampleDimensions[1]):
            zSurfaceVoxels=np.nonzero(surface[ix,iy,:])
            if zSurfaceVoxels.size%2 == 0:
                for i in range(zSurfaceVoxels.size/2):
                    voxelsINSIDE = (zVoxels>zSurfaceVoxels[2*i] 
                                    & zVoxels<zSurfaceVoxels[2*i+1])
                    image[ix,iy,voxelsINSIDE] = 1
            else:
                correctionLIST.append([ix,iy])
    
    countCORRECTIONLIST = len(correctionLIST)

    if countCORRECTIONLIST>0:
        for loopC in range(countCORRECTIONLIST):
            voxelsforcorrection = np.squeeze( np.sum( [ 
                image[correctionLIST[loopC][0]-1,correctionLIST[loopC][1]-1,:],
                image[correctionLIST[loopC][0]-1,correctionLIST[loopC][1],:],
                image[correctionLIST[loopC][0]-1,correctionLIST[loopC][1]+1,:],
                image[correctionLIST[loopC][0],correctionLIST[loopC][1]-1,:],
                image[correctionLIST[loopC][0],correctionLIST[loopC][1]+1,:],
                image[correctionLIST[loopC][0]+1,correctionLIST[loopC][1]-1,:],
                image[correctionLIST[loopC][0]+1,correctionLIST[loopC][1],:],
                image[correctionLIST[loopC][0]+1,correctionLIST[loopC][1]+1,:],
                ] ) )
            voxelsforcorrection = (voxelsforcorrection>=4)
            image[correctionLIST[loopC][0],correctionLIST[loopC][1],voxelsforcorrection] = 1
        
        
#    labels=ndimage.measurements.label(np.logical_not(voxelizedSurface))[0]
#    inside = labels==labels[insideVoxel]    
#    image=np.zeros(sampleDimensions).astype(np.uint8)
#    image[inside]=255
#    image[surface]=255
    
    return image    
    
#--------------------------------------------------------------------
 
if __name__ == "__main__":
    main()