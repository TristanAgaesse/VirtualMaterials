# -*- coding: utf-8 -*-
"""
Created on Fri Jan  2 18:50:02 2015

@author: Tristan Agaesse
"""
import numpy as np
import vtk
from vtk.util import numpy_support
import math
from scipy import ndimage

import os
import sys
sys.path.append(os.path.pardir)
import tifffile as tff


def Test():
    voxelNumbers = (100,100,100)
    image=np.zeros(voxelNumbers).astype(np.bool)
    bounds=(-5.0, 5.0, -5.0, 5.0, -5.0, 5.0)
    
    gridX=np.linspace(bounds[0],bounds[1],voxelNumbers[0]+1)
    gridY=np.linspace(bounds[3],bounds[2],voxelNumbers[1]+1)
    gridZ=np.linspace(bounds[5],bounds[4],voxelNumbers[2]+1)
    
    for i in range(1):
        center = (i/30.0 ,i/30.0,i/30.0)
        radius = 1
        mesh = CreateEllipsoid(center,2,1,0.5)
        
        objImage=Voxelize(mesh,gridX,gridY,gridZ)
        image=np.logical_or(image,objImage)
    
    print(np.count_nonzero(image))
    tff.imsave('TestEllipsoid.tif',(255*image).astype(np.uint8))

#--------------------------------------------------------------------
def CreateVirtualGDL():
    
    #Put cylinders according to a given random law
    1
    
    
    
    #Add binder
    
    
    
#--------------------------------------------------------------------
def CreateVirtualActiveLayer():  
    
    #Choose spherical voids between agglomerates
    1
    
    #Add carbon spheres to form agglomerates


#--------------------------------------------------------------------
def CreateVoronoi(voxelNumbers,imageBounds,outputFile,fiberFile,radiusFile,pointFile,verticeFile):

    image=np.zeros(voxelNumbers).astype(np.bool)
    
    gridX=np.linspace(imageBounds[0],imageBounds[1],voxelNumbers[0]+1)
    gridY=np.linspace(imageBounds[3],imageBounds[2],voxelNumbers[1]+1)
    gridZ=np.linspace(imageBounds[5],imageBounds[4],voxelNumbers[2]+1)

    def readMatrix(filename,elementType):
        f = open ( filename , 'r')
        if elementType is 'float':
            l = [ map(float,line.split(' ')) for line in f ]
        else :
            l = [ map(int,line.split(' ')) for line in f ]
        return l
        f.close()
                                  
    fibres=readMatrix(fiberFile,"int")
    radius=readMatrix(radiusFile,"float")
    points=readMatrix(pointFile,"float")
    vertices=readMatrix(verticeFile,"int")
    
    nFibre=len(fibres)
    print nFibre
    nVertice=len(vertices)
    print nVertice
    
    sphereRadii = np.zeros(len(points))    
    
    for iFibre in range(nFibre):
        
      iPoint1=fibres[iFibre][1]
      iPoint2=fibres[iFibre][2]
      origin=tuple(points[iPoint1])
      end=tuple(points[iPoint2])
      thisRadius=radius[iPoint1][0]
      #construction of radii of the sperical capings of cylinders
      sphereRadii[iPoint1]=max(thisRadius,sphereRadii[iPoint1])
      sphereRadii[iPoint2]=max(thisRadius,sphereRadii[iPoint2])
      #heigth=(end-origin).Length
      
      mesh = CreateCylinder(origin,end,thisRadius,)
      objImage=Voxelize(mesh,gridX,gridY,gridZ)
      image=np.logical_or(image,objImage)
      
    #spherical capings of cylinders      
    for iVertice in range(nVertice):
        
        iPoint = vertices[iVertice][1]-1
        thisRadius = sphereRadii[iPoint]
        center = tuple(points[iPoint])
        
        mesh = CreateSphere(center,thisRadius)
        objImage=Voxelize(mesh,gridX,gridY,gridZ)
        image=np.logical_or(image,objImage)

    tff.imsave(outputFile,(255*image).astype(np.uint8))


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
    
    npts      = 25
    vtkPoints = vtk.vtkPoints()
    vtkPoints.SetNumberOfPoints(100)
    for i in range (npts):
        x = (1.0+i/float(npts))*math.sin(math.pi*i/5.)
        y = (1.0+i/float(npts))*math.cos(math.pi*i/5.)
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
    #Voxelize the object on a window adapted to its bounds. The windows will 
    #be chosed to be a subsample or an extention of the whole image voxel 
    #gridX,Y,Z. This function uses VTK VoxelModel to voxelize the surface, 
    #then the FillInside function to fill the inside.

    bounds=vtkPolyDataObject.GetBounds()    
    nVoxGridX = len(gridX)-1  
    nVoxGridY = len(gridY)-1
    nVoxGridZ = len(gridZ)-1    
    
    Xmin = min(gridX)
    deltaX = (max(gridX)-Xmin)/float(nVoxGridX)    
    subNxMin = int(math.floor((bounds[0]-Xmin)/deltaX)-1)   
    subNxMax = int(math.ceil((bounds[1]-Xmin)/deltaX)+1)
    subgridXmin = Xmin+deltaX*subNxMin
    subgridXmax = Xmin+deltaX*subNxMax
    nVoxSubgridX = subNxMax - subNxMin
    
    Ymin = min(gridY)
    deltaY = (max(gridY)-Ymin)/float(nVoxGridY)
    subNyMin = int(math.floor((bounds[2]-Ymin)/deltaY)-1)
    subNyMax = int(math.ceil((bounds[3]-Ymin)/deltaY)+1)
    subgridYmin = Ymin+deltaY*subNyMin
    subgridYmax = Ymin+deltaY*subNyMax
    nVoxSubgridY = subNyMax - subNyMin
    
    Zmin = min(gridZ)
    deltaZ = (max(gridZ)-Zmin)/float(nVoxGridZ)
    subNzMin = int(math.floor((bounds[4]-Zmin)/deltaZ)-1)
    subNzMax = int(math.ceil((bounds[5]-Zmin)/deltaZ)+1)
    subgridZmin = Zmin+deltaZ*subNzMin
    subgridZmax = Zmin+deltaZ*subNzMax
    nVoxSubgridZ = subNzMax - subNzMin
    
    
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
    voxelizedSurface = voxelizedSurface.reshape((nVoxSubgridX,nVoxSubgridY,nVoxSubgridZ),order='F').astype(np.uint8)
    
    subImage=FillInside(voxelizedSurface)
    
    #Get back to the original window
    wXmin = max(0,subNxMin)
    wXmax = min(len(gridX)-1,subNxMax)
    subwXmin = max(0,-subNxMin)
    subwXmax = min(len(gridX)-1-subNxMin,subNxMax-subNxMin)
    assert subwXmax>subwXmin &  (wXmax>wXmin), "%r  %r  %r  %r" % (subwXmax,subwXmin,wXmax,wXmin)     
    
    wYmin = max(0,subNyMin)
    wYmax = min(len(gridY)-1,subNyMax)
    subwYmin = max(0,-subNyMin)
    subwYmax = min(len(gridY)-1-subNyMin,subNyMax-subNyMin)
    assert subwYmax>subwYmin &  (wYmax>wYmin)    
    
    wZmin = max(0,subNzMin)
    wZmax = min(len(gridZ)-1,subNzMax)
    subwZmin = max(0,-subNzMin)
    subwZmax = min(len(gridZ)-1-subNzMin,subNzMax-subNzMin)
    assert subwZmax>subwZmin &  (wZmax>wZmin)
    
    objectImage = np.zeros((nVoxGridX,nVoxGridY,nVoxGridZ)).astype(np.uint8)
    objectImage[wXmin:wXmax,wYmin:wYmax,wZmin:wZmax] = subImage[subwXmin:subwXmax,
                                            subwYmin:subwYmax,subwZmin:subwZmax]    
    
    
    return objectImage.astype(np.bool)


#--------------------------------------------------------------------    
def FillInside(voxelizedSurface,raydirection='xyz'): 

    #Count the number of voxels in each direction:
    sampleDimensions = voxelizedSurface.shape     
    voxcountX = sampleDimensions[0]
    voxcountY = sampleDimensions[1]
    voxcountZ = sampleDimensions[2]

    
    # Prepare logical array to hold the voxelised data:
    gridOUTPUT      = np.zeros( (voxcountX,voxcountY,voxcountZ,len(raydirection)) ).astype(np.bool)
    countdirections = 0;
    
    if raydirection.find('x')>-1:
      countdirections = countdirections + 1;
      gridOUTPUT[:,:,:,countdirections-1] = np.transpose( FillInsideInternal(np.transpose(voxelizedSurface,axes=[1,2,0])) ,axes=[2,0,1] )
    
    if raydirection.find('y')>-1:
      countdirections = countdirections + 1;
      gridOUTPUT[:,:,:,countdirections-1] = np.transpose( FillInsideInternal(np.transpose(voxelizedSurface,axes=[2,0,1])) ,axes=[1,2,0] )
    
    if raydirection.find('z')>-1:
      countdirections = countdirections + 1;
      gridOUTPUT[:,:,:,countdirections-1] = FillInsideInternal(voxelizedSurface)
    
    # Combine the results of each ray-tracing direction:
    if len(raydirection)>1:
      gridOUTPUT = np.sum(gridOUTPUT,axis=3)>=len(raydirection)/2.0

    return gridOUTPUT 
    
#--------------------------------------------------------------------    
def FillInsideInternal(voxelizedSurface):    
    #Fills the inside of a voxelized closed surface. This function is inspired 
    #by some parts of the Matlab file exchange function VOXELISE (AUTHOR  
    #Adam H. Aitkenhead, The Christie NHS Foundation Trust) 

    surface = voxelizedSurface==1
    sampleDimensions = voxelizedSurface.shape
    image=np.zeros(sampleDimensions).astype(np.uint8)
    
    correctionLIST = []
    zVoxels=np.asarray(range(sampleDimensions[2]))
    
    for ix in range(sampleDimensions[0]):
        for iy in range(sampleDimensions[1]):
            #zSurfaceVoxels=np.flatnonzero(surface[ix,iy,:])
            zSurfaceVoxels=ndimage.measurements.label(surface[ix,iy,:] , structure=np.ones(3))[0]
            if zSurfaceVoxels.max()>0:
                labelCenters=ndimage.measurements.center_of_mass(zSurfaceVoxels, labels=zSurfaceVoxels ,
                                                         index=range(1,zSurfaceVoxels.max()+1))
#                labelLength=ndimage.measurements.labeled_comprehension(
#                                                zSurfaceVoxels, zSurfaceVoxels, 
#                                                range(1,zSurfaceVoxels.max()+1),
#                                                np.size,np.uint8,0)
                if zSurfaceVoxels.max()%2 == 0: #& max(labelLength)<3:
                    for i in range(zSurfaceVoxels.max()/2):
                        voxelsINSIDE = np.logical_and(np.greater(zVoxels,labelCenters[i][0]*np.ones(sampleDimensions[2])), 
                                                      np.less(zVoxels,labelCenters[i+1][0]*np.ones(sampleDimensions[2])))
                        image[ix,iy,voxelsINSIDE] = 1
                else:
                    correctionLIST.append([ix,iy])
    
    
    # USE INTERPOLATION TO FILL IN THE RAYS WHICH COULD NOT BE VOXELISED
    #For rays where the voxelisation did not give a clear result, the ray is
    #computed by interpolating from the surrounding rays.    
    
    countCORRECTIONLIST = len(correctionLIST)
    
    #If necessary, add a one-pixel border around the x and y edges of the
    #array.  This prevents an error if the code tries to interpolate a ray at
    #the edge of the x,y grid.
    if min(correctionLIST[:][0])==0 or max([correctionLIST[i][0] for i in range(len(correctionLIST))] )==sampleDimensions[0]-1 or min(correctionLIST[:][1])==0 or max([correctionLIST[i][1] for i in range(len(correctionLIST))] )==sampleDimensions[1]-1:
        image = np.hstack( (np.zeros((sampleDimensions[0],1,sampleDimensions[2])),
                            image,np.zeros((sampleDimensions[0],1,sampleDimensions[2]))))
        image = np.vstack( (np.zeros((1,sampleDimensions[1]+2,sampleDimensions[2])),
                            image,np.zeros((1,sampleDimensions[1]+2,sampleDimensions[2]))))
        correctionLIST = [ [correctionLIST[i][0]+1,correctionLIST[i][1]+1] 
                                        for i in range(len(correctionLIST)) ]
      
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
                ], axis=0 ) )
            voxelsforcorrection = (voxelsforcorrection>=4)
            image[correctionLIST[loopC][0],correctionLIST[loopC][1],voxelsforcorrection] = 1
        
  #Remove the one-pixel border surrounding the array, if this was added
  #previously.
    if image.shape[0]>sampleDimensions[0] or image.shape[1]>sampleDimensions[1]:
        image = image[1:-1,1:-1,:]
    
    image[surface]=1
#    labels=ndimage.measurements.label(np.logical_not(voxelizedSurface))[0]
#    inside = labels==labels[insideVoxel]    
#    image=np.zeros(sampleDimensions).astype(np.uint8)
#    image[inside]=255
#    image[surface]=255
    
    return image    
    
#--------------------------------------------------------------------
 
if __name__ == "__main__":
    Test()