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
import random
from skimage import morphology
import SimpleITK as sitk

import os
import sys
sys.path.append(os.path.pardir)
import tifffile as tff



#--------------------------------------------------------------------
#      Virtual Materials
#--------------------------------------------------------------------


#--------------------------------------------------------------------
def CreateVirtualGDL(voxelNumbers,nFiber,fiberRadius,fiberLength,
                             binderThickness,anisotropy=1,randomSeed=1):
    
    print('Create Virtual GDL')
    random.seed(randomSeed)
    
    image=np.zeros(voxelNumbers,dtype=np.bool)
    bounds=(0.0, float(voxelNumbers[0]), 
            0.0, float(voxelNumbers[1]), 
            0.0, float(voxelNumbers[2]))
    
    gridX=np.linspace(bounds[0],bounds[1],voxelNumbers[0]+1)
    gridY=np.linspace(bounds[2],bounds[3],voxelNumbers[1]+1)
    gridZ=np.linspace(bounds[4],bounds[5],voxelNumbers[2]+1)
    
    #Put cylinders according to a given random law
    for i in range(nFiber):
        center = (random.uniform(bounds[0], bounds[1]),
                  random.uniform(bounds[2], bounds[3]),
                  random.uniform(bounds[4], bounds[5]))
        
        axis = (anisotropy*random.random(),
                anisotropy*random.random(),
                random.random())        
        
        mesh = CreateCylinder(center,axis,fiberRadius,fiberLength)
        objImage = Voxelize(mesh,gridX,gridY,gridZ)
        image = np.logical_or(image,objImage)
    
    #Add binder
#    ball = morphology.ball(binderThickness)
#    binder = ndimage.morphology.binary_closing(image, structure=ball)
    
    myItkImage = sitk.GetImageFromArray(image.astype(np.uint8))
    myItkImage = sitk.BinaryDilate(myItkImage, int(binderThickness), sitk.sitkBall, 0.0, 1.0,  False)   
    myItkImage = sitk.BinaryErode(myItkImage, int(binderThickness), sitk.sitkBall, 0.0, 1.0,  False)   
    binder=sitk.GetArrayFromImage(myItkImage).astype(np.bool)     
    
    binder = np.logical_and(binder,np.logical_not(image))    
    
    image = image.astype(np.uint8)
    image[binder] = 2    
    
    return image
    
    
    
#--------------------------------------------------------------------
def CreateVirtualCatalystLayer(voxelNumbers,carbonGrainRadius,nCarbonGrain,voidRadius,nVoid,
                             nafionThickness=1,nafionCoveragePercentage=0,randomSeed=1):  
    
    print('Create Virtual Active Layer')
    random.seed(randomSeed)
    
    image=np.zeros(voxelNumbers,dtype=np.bool)
    bounds=(0.0, float(voxelNumbers[0]), 
            0.0, float(voxelNumbers[1]), 
            0.0, float(voxelNumbers[2]))
    
    gridX=np.linspace(bounds[0],bounds[1],voxelNumbers[0]+1)
    gridY=np.linspace(bounds[2],bounds[3],voxelNumbers[1]+1)
    gridZ=np.linspace(bounds[4],bounds[5],voxelNumbers[2]+1)
    
    #Choose spherical voids between agglomerates
    void=np.zeros(voxelNumbers,dtype=np.bool)
    for i in range(nVoid):
        
        center = (random.uniform(bounds[0], bounds[1]),
                  random.uniform(bounds[2], bounds[3]),
                  random.uniform(bounds[4], bounds[5]))
#        void[int(center[0]),int(center[1]),int(center[2])]=1
        objImage=CreateVoxelizedBallFast(center,voidRadius,voxelNumbers,bounds)            
#        mesh = CreateBall(center,voidRadius)
#        objImage=Voxelize(mesh,gridX,gridY,gridZ)
        void=np.logical_or(image,objImage)
    
#    myItkImage = sitk.GetImageFromArray(void.astype(np.uint8))
#    myItkImage = sitk.BinaryDilate(myItkImage, int(voidRadius), sitk.sitkBall, 0.0, 1.0,  False)   
#    void = sitk.GetArrayFromImage(myItkImage).astype(np.bool)     
    
    #Add carbon spheres with Nafion outside voids
    iGrain, loop = 0, 0
    assert 0 <= nafionCoveragePercentage <= 100
    nGrainWithNafion = int(nCarbonGrain*nafionCoveragePercentage/100.0)
    while iGrain < nGrainWithNafion : 
        center = (random.uniform(bounds[0], bounds[1]),
                  random.uniform(bounds[2], bounds[3]),
                  random.uniform(bounds[4], bounds[5]))
        
        if void[int(center[0]),int(center[1]),int(center[2])]==0:
#            mesh = CreateBall(center,grainRadius)
#            objImage=Voxelize(mesh,gridX,gridY,gridZ)
            image[int(center[0]),int(center[1]),int(center[2])]=1
#            objImage=CreateVoxelizedBallFast(center,grainRadius,voxelNumbers,bounds)            
#            image=np.logical_or(image,objImage)
            iGrain = iGrain+1
        
        loop=loop+1
        if loop > 10000000:
            break
    
    myItkImage = sitk.GetImageFromArray(image.astype(np.uint8))
    myItkImage = sitk.BinaryDilate(myItkImage, int(carbonGrainRadius), sitk.sitkBall, 0.0, 1.0,  False)   
    image = sitk.GetArrayFromImage(myItkImage).astype(np.bool)  
    
    #Add Nafion
    myItkImage = sitk.GetImageFromArray(image.astype(np.uint8))
    myItkImage = sitk.BinaryDilate(myItkImage, int(nafionThickness), sitk.sitkBall, 0.0, 1.0,  False)   
    nafion = sitk.GetArrayFromImage(myItkImage).astype(np.bool)  
    nafion = np.logical_and(nafion,np.logical_not(image))
    
    image = image.astype(np.bool)
    
    #Add carbon spheres without Nafion outside voids
    carbon = np.zeros(voxelNumbers,dtype=np.bool)
    nGrainWithoutNafion = nCarbonGrain-nGrainWithNafion
    iGrain, loop = 0, 0
    while iGrain < nGrainWithoutNafion : 
        center = (random.uniform(bounds[0], bounds[1]),
                  random.uniform(bounds[2], bounds[3]),
                  random.uniform(bounds[4], bounds[5]))
        
        if void[int(center[0]),int(center[1]),int(center[2])]==0:
#            mesh = CreateBall(center,grainRadius)
#            objImage=Voxelize(mesh,gridX,gridY,gridZ)
            carbon[int(center[0]),int(center[1]),int(center[2])]=1
#            objImage=CreateVoxelizedBallFast(center,grainRadius,voxelNumbers,bounds)            
#            image=np.logical_or(imGDLage,objImage)
            
            iGrain = iGrain+1
        
        loop=loop+1
        if loop > 10000000:
            break
    
    myItkImage = sitk.GetImageFromArray(carbon.astype(np.uint8))
    myItkImage = sitk.BinaryDilate(myItkImage, int(carbonGrainRadius), sitk.sitkBall, 0.0, 1.0,  False)   
    carbon = sitk.GetArrayFromImage(myItkImage).astype(np.bool)    
    
    image = image.astype(np.uint8)
    image[nafion] = 2        
    image[carbon] = 1 
    
    return image

#--------------------------------------------------------------------
def CreateVirtualAgglomerate(voxelNumbers,grainRadius,nGrain,voidRadius,nVoid,
                             nafionThickness=1,nafionCoveragePercentage=0,randomSeed=1):


    return 1


#--------------------------------------------------------------------
def CreateVirtualInterfaceGDLMPL(voxelNumbers,grainRadius,nGrain,voidRadius,nVoid,
                             nafionThickness=1,nafionCoveragePercentage=0,randomSeed=1):

    #Create GDL Image


    #Zoom on the bottom of GDL with resampling


    #Create MPL Image


    #Add the two images

    return 1


#--------------------------------------------------------------------
def CreateVirtualInterfaceCatalystLayerMembrane(voxelNumbers,grainRadius,nGrain,voidRadius,nVoid,
                             nafionThickness=1,nafionCoveragePercentage=0,randomSeed=1):

    #Create Catalyst Layer Image



    #Create a piece of Membrane



    #Add the two images


    return 1


#--------------------------------------------------------------------
def CreateVirtualVoronoiFoam(voxelNumbers,imageBounds,nPoint,fiberRadius,anisotropy,randomSeed=0):
    
    print('Create Voronoi')
    
    #Prepare the structure image
    image=np.zeros(voxelNumbers,dtype=np.bool)
    
    gridX=np.linspace(imageBounds[0],imageBounds[1],voxelNumbers[0]+1)
    gridY=np.linspace(imageBounds[2],imageBounds[3],voxelNumbers[1]+1)
    gridZ=np.linspace(imageBounds[4],imageBounds[5],voxelNumbers[2]+1)
    
    
    #Read the positions of cylinders and balls in files
#    def readMatrix(filename,elementType):
#        f = open ( filename , 'r')
#        if elementType is 'float':
#            l = [ map(float,line.split(' ')) for line in f ]
#        else :
#            l = [ map(int,line.split(' ')) for line in f ]
#        return l
#        f.close()
#                                  
#    fibres=readMatrix(fiberFile,"int")
#    radius=readMatrix(radiusFile,"float")
#    points=readMatrix(pointFile,"float")
#    vertices=readMatrix(verticeFile,"int")
#    
#    nFibre=len(fibres)
#    nVertice=len(vertices)
#    sphereRadii = np.zeros(len(points))  
    
    vertices,fibres = VoronoiPoints(nPoint,anisotropy,imageBounds,randomSeed)
    
#    xCoord = [vertices[i][0] for i in range(nPoint)]
#    xmin,xmax = min(xCoord),max(xCoord)
#    yCoord = [vertices[i][1] for i in range(nPoint)]
#    ymin,ymax = min(yCoord),max(yCoord)    
#    zCoord = [vertices[i][2] for i in range(nPoint)]
#    zmin,zmax = min(zCoord),max(zCoord)    
#    
#    subWindowBounds = (xmin,xmax,ymin,ymax,zmin,zmax)   
#    
#    nVoxSubImage,boundSubgrid,gridRelativePosition = GetSubWindowInformation(
#                                            subWindowBounds,gridX,gridY,gridZ)
#    
#    subgridX=np.linspace(boundSubgrid[0],boundSubgrid[1],nVoxSubImage[0]+1)
#    subgridY=np.linspace(boundSubgrid[2],boundSubgrid[3],nVoxSubImage[1]+1)
#    subgridZ=np.linspace(boundSubgrid[4],boundSubgrid[5],nVoxSubImage[2]+1)    
#    
    nVertice, nFibre = len(vertices),len(fibres)
    #Add cylinders and balls to the structure image
    print(str(nFibre)+' fibers')
    print(str(nVertice)+' vertices')
    
    for iFibre in range(nFibre):
        print iFibre,         
        iPoint1=fibres[iFibre][0]
        iPoint2=fibres[iFibre][1]
        origin=np.array(vertices[iPoint1])
        end=np.array(vertices[iPoint2])
        height=np.linalg.norm(end-origin)
        axis=end-origin
        center=tuple((end+origin)/2)
        mesh = CreateCylinder(center,axis,fiberRadius,height)
        objImage=Voxelize(mesh,gridX,gridY,gridZ)
        image=np.logical_or(image,objImage)
      
      
    #spherical capings of cylinders      
    for iVertice in range(nVertice):
        center = vertices[iVertice]
        
        mesh = CreateBall(tuple(center),fiberRadius)
        objImage=Voxelize(mesh,gridX,gridY,gridZ)
        image=np.logical_or(image,objImage)

    #image = InsertSubimageInImage(image,voxelNumbers,gridRelativePosition)
    
    return image

#--------------------------------------------------------------------
def CreateVirtualCeramicElectrode():
    
    #Create Higth temperature Fuel Cell Electrode
    
    #Add polyhedra
    
    return 1


#--------------------------------------------------------------------
def PutLaserHolesInGDL(gdlImage,nHole,holeRadius,holeHeight):
    
    voxelNumbers = gdlImage.shape
    imageBounds=(0.0, float(voxelNumbers[0]), 
            0.0, float(voxelNumbers[1]), 
            0.0, float(voxelNumbers[2]))
    gridX=np.linspace(imageBounds[0],imageBounds[1],voxelNumbers[0]+1)
    gridY=np.linspace(imageBounds[2],imageBounds[3],voxelNumbers[1]+1)
    gridZ=np.linspace(imageBounds[4],imageBounds[5],voxelNumbers[2]+1)
    
    holeAxis=(0,0,1)    
    
    for iHole in range(nHole):
        
        center = (random.uniform(imageBounds[0], imageBounds[1]),
                  random.uniform(imageBounds[2], imageBounds[3]),
                  math.ceil(voxelNumbers[2]-holeHeight/2))
                  
        mesh = CreateCylinder(center,holeAxis,holeRadius,holeHeight)
        objImage=Voxelize(mesh,gridX,gridY,gridZ)
        gdlImage=np.logical_and(gdlImage,np.logical_not(objImage))
        
    return gdlImage








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
    polydata=source.GetOutput()
    polydata.Update()    

    return polydata


#--------------------------------------------------------------------
def CreateCylinder(center,axis,radius,height):
    
    source = vtk.vtkCylinderSource()
    source.SetCenter(0,0,0)
    source.SetRadius(radius)
    source.SetHeight(height)
    source.SetResolution(10)
    
    polydata=source.GetOutput()
    polydata.Update()

    #Perform rotation to get the rigth axis
    oldAxis = (0,1,0)
    polydata = MeshRotate(polydata,oldAxis,axis)
    
    #Perform translation to get the rigth center
    translationVector = center
    polydata = MeshTranslate(polydata,translationVector)
    
    return polydata
    
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
    polydata.Update()
    
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
    polydata.Update()
    
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
def CreateTorus(center,axis):
    
    torus = vtk.vtkParametricTorus()
    
    source = vtk.vtkParametricFunctionSource()
    source.SetParametricFunction(torus)
    source.Update()
    
    polydata=source.GetOutput()
    polydata.Update()    
    
    #Perform rotation to get the rigth axis
    oldAxis = (0,1,0)
    polydata = MeshRotate(polydata,oldAxis,axis)
    
    #Perform translation to get the rigth center
    translationVector = center
    polydata = MeshTranslate(polydata,translationVector)
    
    return polydata
    
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
    nVoxSubImage,boundSubgrid,gridRelativePosition = GetSubWindowInformation(
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
    objectImage = InsertSubimageInImage(subImage,imageVoxelNumber,gridRelativePosition)
    
    return objectImage
    
    
    
#--------------------------------------------------------------------    
def MeshTranslate(polydata,translationVector):    
    
    #Perform translation 
    transform = vtk.vtkTransform()
    transform.Translate(translationVector[0],translationVector[1],translationVector[2])    
    transformFilter=vtk.vtkTransformPolyDataFilter()
    transformFilter.SetTransform(transform)
    transformFilter.SetInput(polydata)   
    
    polydata = transformFilter.GetOutput()
    polydata.Update()

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
    transformFilter.SetInput(polydata)   
    
    polydata = transformFilter.GetOutput()
    polydata.Update()
    
    return polydata


#--------------------------------------------------------------------    
def VisualizeMesh(polydata):

    mapper = vtk.vtkPolyDataMapper()
    mapper.SetInput(polydata)
    actor = vtk.vtkActor()
    actor.SetMapper(mapper)
     
    renderer = vtk.vtkRenderer()
    renderWindow = vtk.vtkRenderWindow()
    renderWindow.AddRenderer(renderer)
    renderWindowInteractor = vtk.vtkRenderWindowInteractor()
    renderWindowInteractor.SetRenderWindow(renderWindow)
     
    renderer.AddActor(actor)
     
    renderWindow.Render()
    renderWindowInteractor.Start()










#--------------------------------------------------------------------
#      Voxelisation
#--------------------------------------------------------------------

    
#--------------------------------------------------------------------
def Voxelize(vtkPolyDataObject,gridX,gridY,gridZ):
    #Voxelize the object on a window adapted to its bounds. The windows will 
    #be chosed to be a subsample or an extention of the whole image voxel 
    #gridX,Y,Z. This function uses VTK VoxelModel to voxelize the surface, 
    #then the FillInside function to fill the inside.

    #Prepare a subwindows zooming on the object
    subWindowBound=vtkPolyDataObject.GetBounds()    
    nVoxSubImage,boundSubgrid,gridRelativePosition = GetSubWindowInformation(
                                                        subWindowBound,gridX,gridY,gridZ)
    
    #Voxelize the surface
    voxelizedSurface = VoxelizeSurface(vtkPolyDataObject,nVoxSubImage,boundSubgrid)
    
    #Fill the inside
    subImage=FillInside(voxelizedSurface)
    
    #Get back to the original window
    nVoxImage = (len(gridX)-1,len(gridY)-1,len(gridZ)-1)    
    wholeImage = InsertSubimageInImage(subImage,nVoxImage,gridRelativePosition)
    
    return wholeImage.astype(np.bool)


#--------------------------------------------------------------------
def VoxelizeSurface(vtkPolyDataObject,nVoxSubImage,boundSubgrid):
    #Use VTK VoxelModel to Voxelize the surface

    voxelModel = vtk.vtkVoxelModeller()
    voxelModel.SetInput(vtkPolyDataObject)
    voxelModel.SetSampleDimensions(nVoxSubImage[0],nVoxSubImage[1],nVoxSubImage[2])
    voxelModel.SetModelBounds(boundSubgrid[0],boundSubgrid[1],boundSubgrid[2],
                              boundSubgrid[3],boundSubgrid[4],boundSubgrid[5])
    voxelModel.SetScalarTypeToUnsignedChar()
    voxelModel.SetForegroundValue(1)
    voxelModel.SetBackgroundValue(0)
    voxelModel.Update()
    voxelizedSurface = numpy_support.vtk_to_numpy(voxelModel.GetOutput().GetPointData().GetScalars())
    voxelizedSurface = voxelizedSurface.reshape(nVoxSubImage,order='F').astype(np.uint8)
    
    return voxelizedSurface
    
    
#--------------------------------------------------------------------    
def FillInside(voxelizedSurface,raydirection='xyz'): 

    #Count the number of voxels in each direction:
    sampleDimensions = voxelizedSurface.shape     
    voxcountX = sampleDimensions[0]
    voxcountY = sampleDimensions[1]
    voxcountZ = sampleDimensions[2]

    
    # Prepare logical array to hold the voxelised data:
    gridOUTPUT = np.zeros((voxcountX,voxcountY,voxcountZ,len(raydirection))).astype(np.bool)
    countdirections = 0;
    
    if raydirection.find('x')>-1:
      countdirections = countdirections + 1;
      gridOUTPUT[:,:,:,countdirections-1] = np.transpose( FillInsideZDirection(
                                  np.transpose(voxelizedSurface,axes=[1,2,0])), 
                                  axes=[2,0,1] )
    
    if raydirection.find('y')>-1:
      countdirections = countdirections + 1;
      gridOUTPUT[:,:,:,countdirections-1] = np.transpose( FillInsideZDirection(
                                  np.transpose(voxelizedSurface,axes=[2,0,1])),
                                  axes=[1,2,0] )
    
    if raydirection.find('z')>-1:
      countdirections = countdirections + 1;
      gridOUTPUT[:,:,:,countdirections-1] = FillInsideZDirection(voxelizedSurface)
    
    # Combine the results of each ray-tracing direction:
    if len(raydirection)>1:
      gridOUTPUT = np.sum(gridOUTPUT,axis=3)>=len(raydirection)/2.0

    return gridOUTPUT 
    
#--------------------------------------------------------------------    
def FillInsideZDirection(voxelizedSurface):    
    #Fills the inside of a voxelized closed surface. This function is inspired 
    #by some parts of the Matlab file exchange function VOXELISE (AUTHOR  
    #Adam H. Aitkenhead, The Christie NHS Foundation Trust) 

    surface = voxelizedSurface.astype(np.int8)
    
    sampleDimensions = voxelizedSurface.shape
    image=np.zeros(sampleDimensions,dtype=np.uint8)
    
    correctionLIST = []
    zVoxels=np.arange(sampleDimensions[2])
    
    for ix in range(sampleDimensions[0]):
        for iy in range(sampleDimensions[1]):
            rolledIndices=[i+1 for i in range(sampleDimensions[2]-1)]
            rolledIndices.append(0)
            a=surface[ix,iy,rolledIndices]-surface[ix,iy,:]
            #print(a.max(),a.min())
            labelEnds = a<0
            if surface[ix,iy,-1]==1:
                labelEnds[-1]=1
            labelEnds = np.flatnonzero(labelEnds)
            #zSurfaceVoxels=ndimage.measurements.label(surface[ix,iy,:] , structure=np.ones(3))[0]
            
            if labelEnds.size>0:
#                labelCenters=ndimage.measurements.center_of_mass(
#                                          zSurfaceVoxels, labels=zSurfaceVoxels,
#                                          index=range(1,zSurfaceVoxels.max()+1))

                if labelEnds.size%2 == 0: 
                    for i in range(labelEnds.size/2):
                        voxelsINSIDE = np.logical_and(
                                np.greater(zVoxels,labelEnds[i]*np.ones(sampleDimensions[2])), 
                                np.less(zVoxels,labelEnds[i+1]*np.ones(sampleDimensions[2])))
                                
                        image[ix,iy,voxelsINSIDE] = 1
                else:
                    correctionLIST.append([ix,iy])
    
    
    # USE INTERPOLATION TO FILL IN THE RAYS WHICH COULD NOT BE VOXELISED
    #For rays where the voxelisation did not give a clear result, the ray is
    #computed by interpolating from the surrounding rays.    
    
    countCORRECTIONLIST = len(correctionLIST)
    
      
    if countCORRECTIONLIST>0:
        
        #If necessary, add a one-pixel border around the x and y edges of the
        #array.  This prevents an error if the code tries to interpolate a ray at
        #the edge of the x,y grid.
        cond0 = min([correctionLIST[i][0] for i in range(len(correctionLIST))])==0
        cond1 = max([correctionLIST[i][0] for i in range(len(correctionLIST))])==sampleDimensions[0]-1
        cond2 = min([correctionLIST[i][1] for i in range(len(correctionLIST))])==0
        cond3 = max([correctionLIST[i][1] for i in range(len(correctionLIST))])==sampleDimensions[1]-1
    
        if cond0 or cond1 or cond2 or cond3:
            image = np.hstack( (np.zeros((sampleDimensions[0],1,sampleDimensions[2])),
                                image,np.zeros((sampleDimensions[0],1,sampleDimensions[2]))))
            image = np.vstack( (np.zeros((1,sampleDimensions[1]+2,sampleDimensions[2])),
                                image,np.zeros((1,sampleDimensions[1]+2,sampleDimensions[2]))))
            correctionLIST = [ [correctionLIST[i][0]+1,correctionLIST[i][1]+1] 
                                            for i in range(len(correctionLIST)) ]
        
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
    
    image[surface.astype(np.bool)]=1
#    labels=ndimage.measurements.label(np.logical_not(voxelizedSurface))[0]
#    inside = labels==labels[insideVoxel]    
#    image=np.zeros(sampleDimensions).astype(np.uint8)
#    image[inside]=255
#    image[surface]=255
    
    return image    

#--------------------------------------------------------------------    
def GetSubWindowInformation(subWindowBounds,gridX,gridY,gridZ): 

    nVoxGridX = len(gridX)-1  
    nVoxGridY = len(gridY)-1
    nVoxGridZ = len(gridZ)-1    
    bounds=subWindowBounds    
    
    
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

    nVoxSubImage = (nVoxSubgridX,nVoxSubgridY,nVoxSubgridZ)
    
    boundSubgrid = (subgridXmin,subgridXmax,
                    subgridYmin,subgridYmax,
                    subgridZmin,subgridZmax,)
                    
    gridRelativePosition=(subNxMin,subNxMax,
                          subNyMin,subNyMax,
                          subNzMin,subNzMax,)
    
    return nVoxSubImage,boundSubgrid,gridRelativePosition

#--------------------------------------------------------------------    
def InsertSubimageInImage(subImage,nVoxImage,gridRelativePosition): 
    
    subNxMin,subNxMax,subNyMin,subNyMax,subNzMin,subNzMax = gridRelativePosition   
    
    wXmin = max(0,subNxMin)
    wXmax = min(nVoxImage[0],subNxMax)
    subwXmin = max(0,-subNxMin)
    subwXmax = min(nVoxImage[0]-subNxMin,subNxMax-subNxMin)
    #print subwXmax, subwXmin , wXmax,wXmin
    assert subwXmax>=subwXmin &  (wXmax>=wXmin)    
    
    wYmin = max(0,subNyMin)
    wYmax = min(nVoxImage[1],subNyMax)
    subwYmin = max(0,-subNyMin)
    subwYmax = min(nVoxImage[1]-subNyMin,subNyMax-subNyMin)
    assert subwYmax>=subwYmin &  (wYmax>=wYmin)    
    
    wZmin = max(0,subNzMin)
    wZmax = min(nVoxImage[2],subNzMax)
    subwZmin = max(0,-subNzMin)
    subwZmax = min(nVoxImage[2]-subNzMin,subNzMax-subNzMin)
    assert subwZmax>=subwZmin &  (wZmax>=wZmin)
    
    objectImage = np.zeros(nVoxImage,dtype=np.uint8)
    objectImage[wXmin:wXmax,wYmin:wYmax,wZmin:wZmax] = subImage[subwXmin:subwXmax,
                                            subwYmin:subwYmax,subwZmin:subwZmax]

    return objectImage
    




#--------------------------------------------------------------------
#      Utilities
#--------------------------------------------------------------------
    
    
#--------------------------------------------------------------------
def SaveImage(image,filename):
    tff.imsave(filename,image.astype(np.uint8))


#--------------------------------------------------------------------
def VoronoiPoints(nPoint,anisotropy,imageBounds,randomSeed=0):    
    from scipy.spatial import Voronoi
    import numpy as np
    
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

#--------------------------------------------------------------------
#      Tests
#--------------------------------------------------------------------
     
#--------------------------------------------------------------------
def Test():
#    voxelNumbers = (100,100,100)
#    image=np.zeros(voxelNumbers).astype(np.bool)
#    bounds=(-5.0, 5.0, -5.0, 5.0, -5.0, 5.0)
#    
#    gridX=np.linspace(bounds[0],bounds[1],voxelNumbers[0]+1)
#    gridY=np.linspace(bounds[3],bounds[2],voxelNumbers[1]+1)
#    gridZ=np.linspace(bounds[5],bounds[4],voxelNumbers[2]+1)
#   
    return 1  
    
#--------------------------------------------------------------------    
def TestVirtualVoronoi():    
    voxelNumbers=(200,200,200)
    imageBounds = (0.0,1.0,0.0,1.0,0.0,1.0)
    nPoint=100
    fiberRadius=0.04
    anisotropy=3
    randomSeed=0
    image=CreateVirtualVoronoiFoam(voxelNumbers,imageBounds,nPoint,fiberRadius,anisotropy,randomSeed)
    SaveImage(255*(image.astype(np.uint8)),'TestVoronoi.tif')

#--------------------------------------------------------------------
def TestVirtualGDL():
    image = CreateVirtualGDL((200,200,100),27,4,200,10,5) 
    SaveImage(100*(image.astype(np.uint8)),'TestGDL.tif')
    
#--------------------------------------------------------------------    
def TestVirtualCCL():
    image = CreateVirtualCatalystLayer((200,200,200),4,5000,50,200,2,30)
    SaveImage(100*(image.astype(np.uint8)),'TestCCL.tif')
    
    
    
    
    
#--------------------------------------------------------------------
if __name__ == "__main__":
    TestVirtualVoronoi()