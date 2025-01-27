# -*- coding: utf-8 -*-

import numpy as np
import vtk
from vtk.util import numpy_support
import math
from scipy import ndimage
import random
from skimage import morphology
import SimpleITK as sitk
from scipy.spatial import Voronoi
import time

from VirtualMaterials.Utilities  import tifffile as tff
from VirtualMaterials.Simulation  import FullMorphology
from VirtualMaterials.VirtualImages import Voxelization

#--------------------------------------------------------------------
#      Basic Shapes
#--------------------------------------------------------------------

#--------------------------------------------------------------------
def CreateBall(center,radius):
    
    source = vtk.vtkSphereSource()
    source.SetCenter(center[0],center[1],center[2])
    source.SetRadius(radius)
    source.SetThetaResolution(10)
    source.SetPhiResolution(10)
    source.Update()
    
    polydata=source.GetOutput()
#    polydata.Update()    

    return polydata


#--------------------------------------------------------------------
def CreateCylinder(center,axis,radius,height,resolution=8):
    
    source = vtk.vtkCylinderSource()
    source.SetCenter(0,0,0)
    source.SetRadius(radius)
    source.SetHeight(height)
    source.SetResolution(int(resolution))
    source.Update()
    
    polydata=source.GetOutput()
    #polydata.Update()

    #Perform rotation to get the rigth axis
    oldAxis = (0,1,0)
    polydata = MeshRotate(polydata,oldAxis,axis)
    
    #Perform translation to get the rigth center
    translationVector = center
    polydata = MeshTranslate(polydata,translationVector)
    
    triFilter=vtk.vtkTriangleFilter()
    
    if vtk.vtkVersion.GetVTKMajorVersion()==6:
        triFilter.SetInputData(polydata)
    else:
        triFilter.SetInput(polydata)
    triFilter.Update()
    
    polydata=triFilter.GetOutput() 
    #polydata.Update()
    
    return polydata
    
#--------------------------------------------------------------------
def CreatePolyhedron(points):
    
    nPoint = len(points)
#    P0 = [0.0, 0.0, 0.0] 
#    P1 = [1.0, 0.0, 0.0]
    
    points=[[float(points[i][j]) for j in range(3)] for i in range(nPoint)]
    
    # Create the points
    VTKpoints = vtk.vtkPoints()
    VTKpoints.SetNumberOfPoints(nPoint)
    for iPoint in range(nPoint):
        VTKpoints.SetPoint(iPoint,points[iPoint])

    # Create a hexahedron from the points
    hex = vtk.vtkHexahedron()
    for iPoint in range(nPoint):
        hex.GetPointIds().SetId(iPoint,iPoint)

    # Add the hexahedron to a cell array
    hexs = vtk.vtkCellArray()
    hexs.InsertNextCell(hex)
    
    # Add the points and hexahedron to an unstructured grid
    uGrid = vtk.vtkUnstructuredGrid()
    uGrid.SetPoints(VTKpoints)
    uGrid.InsertNextCell(hex.GetCellType(), hex.GetPointIds())
    
    # Convert to vtk polydata object 
    geometryFilter =  vtk.vtkGeometryFilter()
    if vtk.vtkVersion.GetVTKMajorVersion()==6:
        geometryFilter.SetInputData(uGrid)
    else:
        geometryFilter.SetInput(uGrid)
    
    geometryFilter.Update()
    polydata = geometryFilter.GetOutput()
    
    #polydata.Update()
    
    return polydata
    
#--------------------------------------------------------------------
def CreateEllipsoid(center,axis,xRadius,yRadius,zRadius):

    ellipsoid = vtk.vtkParametricEllipsoid()
    ellipsoid.SetXRadius(xRadius)
    ellipsoid.SetYRadius(yRadius)
    ellipsoid.SetZRadius(zRadius)
    
    source = vtk.vtkParametricFunctionSource()
    source.SetParametricFunction(ellipsoid)
    source.SetUResolution(15)
    source.SetVResolution(15)
    source.SetWResolution(15)
    source.Update()
    
    polydata=source.GetOutput()
    #polydata.Update()
    
    #Perform rotation to get the rigth axis
    oldAxis = (0,1,0)
    polydata = MeshRotate(polydata,oldAxis,axis)
    
    #Perform translation to get the rigth center
    translationVector = center
    polydata = MeshTranslate(polydata,translationVector)
    
    return polydata
    
#--------------------------------------------------------------------
def CreateRandomHills():

    randomHills = vtk.vtkParametricRandomHills()
    randomHills.AllowRandomGenerationOff()        
    
    source = vtk.vtkParametricFunctionSource()
    source.SetParametricFunction(randomHills)
    source.Update()
    polydata = source.GetOutput()
    #polydata.Update()
    
    return polydata
    
#--------------------------------------------------------------------
def CreateSpline():   
    
    npts      = 25
    vtkPoints = vtk.vtkPoints()
    vtkPoints.SetNumberOfPoints(100)
    for i in range (npts):
        x = math.sin(math.pi*i/5.)
        y = math.cos(math.pi*i/5.)
        z = 2*i/float(npts)
        vtkPoints.SetPoint(i, (x,y,z))
    
    vtkCellArray = vtk.vtkCellArray()
    vtkCellArray.InsertNextCell(npts)
    for i in range(npts):
        vtkCellArray.InsertCellPoint(i)
    
#    value = lambda i: math.fabs(math.sin(math.pi*i/30.))
#    vtkFloatArray = vtk.vtkFloatArray()
#    vtkFloatArray.SetNumberOfValues(npts)
#    for i in range(npts):
#        vtkFloatArray.SetValue(i, value(i))
        
    vtkPolyData = vtk.vtkPolyData()
    vtkPolyData.SetPoints(vtkPoints)
    vtkPolyData.SetLines(vtkCellArray)
#    vtkPolyData.GetPointData().SetScalars(vtkFloatArray)
    
    vtkSplineFilter = vtk.vtkSplineFilter()
    
    if vtk.vtkVersion.GetVTKMajorVersion()==6:
        vtkSplineFilter.SetInputData(vtkPolyData)
    else:
        vtkSplineFilter.SetInput(vtkPolyData)
    
    vtkSplineFilter.SetNumberOfSubdivisions(5*npts)
    vtkSplineFilter.Update()
    
    vtkTubeFilter = vtk.vtkTubeFilter()
    vtkTubeFilter.SetInputConnection(vtkSplineFilter.GetOutputPort())
    vtkTubeFilter.SetRadius(0.15)
    vtkTubeFilter.SetNumberOfSides(10)
    vtkTubeFilter.CappingOn()    
    vtkTubeFilter.Update()
    
    polydata = vtkTubeFilter.GetOutput()
    #polydata.Update()
    
    return polydata    
    
#--------------------------------------------------------------------
def CreateTorus(center,axis):
    
    torus = vtk.vtkParametricTorus()
    
    source = vtk.vtkParametricFunctionSource()
    source.SetParametricFunction(torus)
    source.Update()
    
    polydata=source.GetOutput()
    #polydata.Update()    
    
    #Perform rotation to get the rigth axis
    oldAxis = (0,1,0)
    polydata = MeshRotate(polydata,oldAxis,axis)
    
    #Perform translation to get the rigth center
    translationVector = center
    polydata = MeshTranslate(polydata,translationVector)
    
    return polydata

#--------------------------------------------------------------------
def CreateLine(start, end):
    """
    N-D Bresenham line algo
    Returns a list of points from (start, end] by ray tracing a line b/w the
    points.
    :param start: start point (numpy array)
    :param end:  end points (numpy array)
    :return: linevox :  array of all points traversed by the line.      
    
    Example :
    >>>start = np.array([3, 1, 9])
    >>>end  = np.array([0, 0, 0])
    >>>CreateLine(start, end)
     array([[ 3,  1,  8],
     [ 2,  1,  7],
     [ 2,  1,  6],
     [ 2,  1,  5],
     [ 1,  0,  4],
     [ 1,  0,  3],
     [ 1,  0,  2],
     [ 0,  0,  1],
     [ 0,  0,  0]])
    """
    
    def _bresenhamline_nslope(slope):
        """
        Normalize slope for Bresenham's line algorithm.
        """
        scale = np.amax(np.abs(slope), axis=1).reshape(-1, 1)
        zeroslope = (scale == 0).all(1)
        scale[zeroslope] = np.ones(1)
        normalizedslope = np.array(slope, dtype=np.double) / scale
        normalizedslope[zeroslope] = np.zeros(slope[0].shape)
            
        return normalizedslope
  
    def _bresenhamlines(start, end, max_iter):
        """
        Returns npts lines of length max_iter each. (npts x max_iter x dimension) 
        """
        if max_iter == -1:
            max_iter = np.amax(np.amax(np.abs(end - start), axis=1))
        npts, dim = start.shape
        nslope = _bresenhamline_nslope(end - start)
    
        # steps to iterate on
        stepseq = np.arange(1, max_iter + 1)
        stepmat = np.tile(stepseq, (dim, 1)).T
        
        # some hacks for broadcasting properly
        bline = start[:, np.newaxis, :] + nslope[:, np.newaxis, :] * stepmat
        
        # Approximate to nearest int
        return np.array(np.rint(bline), dtype=start.dtype)
  

    # Return the points as a single array
    max_iter=-1
    start=np.array([start])
    end=np.array([end])
    return _bresenhamlines(start, end, max_iter).reshape(-1, start.shape[-1])





    
#--------------------------------------------------------------------
def CreateVoxelizedBallFast(center,radius,imageVoxelNumber,imageBounds):
    
    bounds = imageBounds
    gridX=np.linspace(bounds[0],bounds[1],imageVoxelNumber[0]+1)
    gridY=np.linspace(bounds[2],bounds[3],imageVoxelNumber[1]+1)
    gridZ=np.linspace(bounds[4],bounds[5],imageVoxelNumber[2]+1)    
    
    #Prepare a subwindows zooming on the object
    subWindowBound = (center[0]-2*radius,center[0]+2*radius,
                      center[1]-2*radius,center[1]+2*radius,
                      center[2]-2*radius,center[2]+2*radius)    
    nVoxSubImage,boundSubgrid,gridRelativePosition = Voxelization.GetSubWindowInformation(
                                                        subWindowBound,gridX,gridY,gridZ)
    
    #Create a voxelized ball
    voxRadius = int(imageVoxelNumber[0]*radius/float(imageBounds[1]-imageBounds[0]))
    myBall = morphology.ball(voxRadius)
    diameter = myBall.shape[0]
    assert diameter <= min(nVoxSubImage[0],nVoxSubImage[1],nVoxSubImage[2])
    
    subImage = np.zeros(nVoxSubImage,dtype=np.bool)
    nxMin = int((nVoxSubImage[0]-diameter)/2.0)
    nyMin = int((nVoxSubImage[1]-diameter)/2.0)
    nzMin = int((nVoxSubImage[2]-diameter)/2.0)
    subImage[nxMin:nxMin+diameter,nyMin:nyMin+diameter,nzMin:nzMin+diameter] = myBall
    
    #Get back to the original window
    objectImage = Voxelization.InsertSubimageInImage(subImage,imageVoxelNumber,gridRelativePosition)
    
    return objectImage
    
    
    
#--------------------------------------------------------------------    
def MeshTranslate(polydata,translationVector):    
    
    #Perform translation 
    transform = vtk.vtkTransform()
    transform.Translate(translationVector[0],translationVector[1],translationVector[2])    
    transformFilter=vtk.vtkTransformPolyDataFilter()
    transformFilter.SetTransform(transform)
    
    if vtk.vtkVersion.GetVTKMajorVersion()==6:
        transformFilter.SetInputData(polydata)
    else:
        transformFilter.SetInput(polydata)    
    
    
       
    transformFilter.Update()    
    
    polydata = transformFilter.GetOutput()
    #polydata.Update()

    return polydata

#--------------------------------------------------------------------    
def MeshRotate(polydata,oldAxis,newAxis): 
    
    #Perform rotation 
    oldAxis = np.asarray(oldAxis)
    oldAxis = oldAxis/np.linalg.norm(oldAxis)
    newAxis = np.asarray(newAxis)
    newAxis = newAxis/np.linalg.norm(newAxis)    
       
    if np.linalg.norm(newAxis-oldAxis)<0.0000001:
        rotationAxis=newAxis
    else:
        rotationAxis = (newAxis-oldAxis)/2.0
        rotationAxis = rotationAxis/np.linalg.norm(rotationAxis)
    
    transform = vtk.vtkTransform()    
    transform.RotateWXYZ(180,rotationAxis[0],rotationAxis[1],rotationAxis[2])
    
    transformFilter=vtk.vtkTransformPolyDataFilter()
    transformFilter.SetTransform(transform)
    
    if vtk.vtkVersion.GetVTKMajorVersion()==6:
        transformFilter.SetInputData(polydata)
    else:
        transformFilter.SetInput(polydata)     
    
    
       
    transformFilter.Update()
    
    polydata = transformFilter.GetOutput()
    #polydata.Update()
    
    return polydata
    
#--------------------------------------------------------------------    
def MeshRotateX(polydata,rotationCenter,angle):    
    #Rotate about the center, rotation axis=x

    transform = vtk.vtkTransform()    
    
    center = rotationCenter
    
    polydata=MeshTranslate(polydata,(-center[0], -center[1], -center[2]))
    
#    transform.PostMultiply()
#    transform.Translate(-center[0], -center[1], -center[2])
    transform.RotateX(angle)
#    transform.Translate(center[0], center[1], center[2])
    
    transformFilter=vtk.vtkTransformPolyDataFilter()
    transformFilter.SetTransform(transform)
    
    if vtk.vtkVersion.GetVTKMajorVersion()==6:
        transformFilter.SetInputData(polydata)
    else:
        transformFilter.SetInput(polydata)     
       
    transformFilter.Update()
    
    polydata = transformFilter.GetOutput()
    #polydata.Update()
    
    polydata=MeshTranslate(polydata,(center[0], center[1], center[2]))    
    
    return polydata
    
#--------------------------------------------------------------------    
def MeshRotateY(polydata,rotationCenter,angle):    
    #Rotate about the center, rotation axis=y

    transform = vtk.vtkTransform()   
    
    center = rotationCenter

    polydata=MeshTranslate(polydata,(-center[0], -center[1], -center[2]))

    transform.RotateY(angle)
    
    
    transformFilter=vtk.vtkTransformPolyDataFilter()
    transformFilter.SetTransform(transform)
    
    if vtk.vtkVersion.GetVTKMajorVersion()==6:
        transformFilter.SetInputData(polydata)
    else:
        transformFilter.SetInput(polydata)     
       
    transformFilter.Update()
    
    polydata = transformFilter.GetOutput()
    #polydata.Update()
    polydata=MeshTranslate(polydata,(center[0], center[1], center[2])) 
    
    return polydata

#--------------------------------------------------------------------    
def MeshRotateZ(polydata,rotationCenter,angle):    
    #Rotate about the center, rotation axis=z

    transform = vtk.vtkTransform()    
    
    center = rotationCenter
    
    polydata=MeshTranslate(polydata,(-center[0], -center[1], -center[2]))

    transform.RotateZ(angle)
    
    transformFilter=vtk.vtkTransformPolyDataFilter()
    transformFilter.SetTransform(transform)
    
    if vtk.vtkVersion.GetVTKMajorVersion()==6:
        transformFilter.SetInputData(polydata)
    else:
        transformFilter.SetInput(polydata)     
       
    transformFilter.Update()
    
    polydata = transformFilter.GetOutput()
    #polydata.Update()
    
    polydata=MeshTranslate(polydata,(center[0], center[1], center[2]))     
    
    return polydata    
    
    
    
    
#--------------------------------------------------------------------
def ComputeVoronoiPoints(nPoint,anisotropy,imageBounds,randomSeed=0):    
    
    random.seed(randomSeed)
    assert anisotropy>0
    
    points=[]
    for i in range(nPoint):
        points.append([random.uniform(imageBounds[0], imageBounds[1]),
                  random.uniform(imageBounds[2], imageBounds[3]),
                  random.uniform(imageBounds[4], imageBounds[5])] )
    
    zCoord = [points[i][2] for i in range(nPoint)]
    zmin = min(zCoord)
    
    points=[[points[i][0],points[i][1],((points[i][2]-zmin)*anisotropy+zmin)] 
                                                            for i in range(nPoint)]
    
    vor = Voronoi(points)
    
    vertices = vor.vertices
    nVertice = len(vertices)
    vertices=[[vertices[i][0],vertices[i][1],((vertices[i][2]-zmin)/anisotropy+zmin)] 
                                                          for i in range(nVertice)]
    
    #remove vertices outside imageBounds
    isinside = [(vertices[i][0]>=imageBounds[0]) & (vertices[i][0]<=imageBounds[1]) &
                (vertices[i][1]>=imageBounds[2]) & (vertices[i][1]<=imageBounds[3]) &
                (vertices[i][2]>=imageBounds[4]) & (vertices[i][2]<=imageBounds[5]) 
                for i in range(nVertice) ]
     
    vertices = [vertices[i] for i in range(nVertice) if isinside[i]]
    oldNum = [i for i in range(nVertice) if isinside[i]]
    newNum = np.zeros(nVertice,dtype=np.uint)    
    newNum[np.asarray(oldNum)]=np.arange(0,len(oldNum))   
    
    #Get edges
    ridges = vor.ridge_vertices
    fibres = [ sorted([ridges[i][j],ridges[i][(j+1)%len(ridges[i])]]) 
                for i in range(len(ridges)) for j in range(len(ridges[i]))]
    goodfibre=[(fibres[i][0]!=-1) & (isinside[fibres[i][0]]) & (isinside[fibres[i][1]])
                        for i in range(len(fibres))]            
    fibres = [[newNum[fibres[i][0]],newNum[fibres[i][1]]] 
                            for i in range(len(fibres)) if goodfibre[i]]
    fibres = np.array([np.array(x) for x in set(tuple(x) for x in fibres)])
     
    return vertices, fibres

