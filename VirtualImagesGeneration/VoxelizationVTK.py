# -*- coding: utf-8 -*-
"""
Created on Fri Jan  2 18:50:02 2015

@author: greentoto
"""
import numpy as np
import vtk
from vtk.util import numpy_support
import math

import sys
sys.path.append('C:/Users/Tristan/Documents/Python/workspace/Image_Computations/')
import tifffile as tff


def main():
    voxelNumbers = (50,50,50)
    image=np.zeros(voxelNumbers).astype(np.uint8)
    bounds=(-1.5, 1.5, -1.5, 1.5, -1.5, 1.5)
    
    gridX=np.linspace(bounds[0],bounds[1],voxelNumbers[0]+1)
    gridY=np.linspace(bounds[3],bounds[2],voxelNumbers[1]+1)
    gridZ=np.linspace(bounds[5],bounds[4],voxelNumbers[2]+1)
    
    for i in range(1):
        center = (i/10.0 ,i/10.0,i/10.0)
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
    
    #Voxelize the object on a window adapted to its bounds. This windows should
    #be a subsample or an extention of gridX,Y,Z
    bounds=vtkPolyDataObject.GetBounds()    
    nVoxGridX = len(gridX)-1  
    nVoxGridY = len(gridY)-1
    nVoxGridZ = len(gridZ)-1    
    
    Xmin = min(gridX)
    deltaX = (max(gridX)-Xmin)/float(nVoxGridX)    
    subNxMin = math.floor((bounds[0]-Xmin)/deltaX)-1   
    subNxMax = math.ceil((Xmin-bounds[1])/deltaX)+1
    subgridXmin = Xmin+deltaX*subNxMin
    subgridXmax = Xmin+deltaX*subNxMax
    nVoxSubgridX = subNxMax - subNxMin
    
    Ymin = min(gridY)
    deltaY = (max(gridY)-Ymin)/float(nVoxGridY)
    subNyMin = math.floor((bounds[2]-Ymin)/deltaY)-1
    subNyMax = math.ceil((Ymin-bounds[3])/deltaY)+1
    subgridYmin = Ymin+deltaY*subNyMin
    subgridYmax = Ymin+deltaY*subNyMax
    nVoxSubgridY = (subgridYmax-subgridYmin)/deltaY
    
    Zmin = min(gridZ)
    deltaZ = (max(gridZ)-Zmin)/float(nVoxGridZ)
    subNzMin = math.floor((bounds[4]-Zmin)/deltaZ)-1
    subNzMax = math.ceil((Zmin-bounds[5])/deltaZ)+1
    subgridZmin = Zmin+deltaZ*subNzMin
    subgridZmax = Zmin+deltaZ*subNzMax
    nVoxSubgridZ = (subgridZmax-subgridZmin)/deltaZ
    
    
    #Use VTK VoxelModel to Voxelize the surface
    voxelModel = vtk.vtkVoxelModeller()
    voxelModel.SetInput(vtkPolyDataObject)
    voxelModel.SetSampleDimensions(nVoxSubgridX,nVoxSubgridY,nVoxSubgridZ)
    voxelModel.SetModelBounds(subgridXmin,subgridXmax,subgridYmin,subgridYmax,subgridZmin,subgridZmax)
    voxelModel.SetScalarTypeToUnsignedChar()
    voxelModel.SetForegroundValue(1)
    voxelModel.SetBackgroundValue(0)
    voxelModel.Update()
    
    voxelizedSurface = numpy_support.vtk_to_numpy(voxelModel.GetOutput().GetPointData().GetScalars())
    voxelizedSurface = voxelizedSurface.reshape((nVoxSubgridX,nVoxSubgridY,nVoxSubgridZ)).astype(np.uint8)
    
    subImage=FillInside(voxelizedSurface)
    
    #Get back to the original window
    windowX = range(max(0,subNxMin),min(len(gridX),subNxMax))
    subWindowX = range(min(0,-subNxMin),min(len(gridX)-subNxMin,subNxMax-subNxMin))   
    
    objectImage = np.zeros((nVoxGridX,nVoxGridY,nVoxGridZ)).astype(np.uint8)
    objectImage[windowX,windowY,windowZ] = subImage[subWindowX,subWindowY,subWindowZ]    
    
    
    return objectImage
    
#--------------------------------------------------------------------    
def FillInside(voxelizedSurface):    
    
    surface = voxelizedSurface==1
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