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
from mayavi import mlab
import time
import os
import sys
sys.path.append(os.path.pardir+"/Utilities")
import tifffile as tff



#--------------------------------------------------------------------
#      Virtual Materials
#--------------------------------------------------------------------


#--------------------------------------------------------------------
def CreateVirtualGDL(voxelNumbers=(200,200,200),fiberContent=0.5,fiberRadius=10,fiberLength=100,
                             binderContent=0.3,anisotropy=1,randomSeed=1):
    
    #TODO : change algorithm and input parameters :
    #add fibers until a given fiber content is achieved. Add binder until a given porosity is achieved    
    
    
    print('Create Virtual GDL')
    beginTime=time.time()
        
    random.seed(randomSeed)

    assert(fiberContent+binderContent<=1)
    
    image=np.zeros(voxelNumbers,dtype=np.bool)
    bounds=(0.0, float(voxelNumbers[0]), 
            0.0, float(voxelNumbers[1]), 
            0.0, float(voxelNumbers[2]))
    
    gridX=np.linspace(bounds[0],bounds[1],voxelNumbers[0]+1)
    gridY=np.linspace(bounds[2],bounds[3],voxelNumbers[1]+1)
    gridZ=np.linspace(bounds[4],bounds[5],voxelNumbers[2]+1)
    
    #Put cylinders according to a given random law
    print('Adding fibers until fiberContent is reached')
    fiberTotalVolume = 0.0
    while fiberTotalVolume<fiberContent:
        center = (random.uniform(bounds[0], bounds[1]),
                  random.uniform(bounds[2], bounds[3]),
                  random.uniform(bounds[4], bounds[5]))
        
        axis = (anisotropy*random.random(),
                anisotropy*random.random(),
                random.random())        
        
        mesh = CreateCylinder(center,axis,fiberRadius,fiberLength)
        objImage = Voxelize(mesh,gridX,gridY,gridZ,raydirection='z')
        image = np.logical_or(image,objImage)
        
        fiberTotalVolume = float(np.count_nonzero(image))/np.size(image)        
        print(fiberTotalVolume)
        
    #Add binder
#    ball = morphology.ball(binderThickness)
#    binder = ndimage.morphology.binary_closing(image, structure=ball)
    
    print('Adding binder until binderContent is reached')
    binderTotalVolume = 0.0
    binderThickness=1
    
    myItkImage = sitk.GetImageFromArray(image.astype(np.uint8))
    
    while binderTotalVolume<binderContent :
        print(binderTotalVolume)
        
        myItkImage = sitk.BinaryDilate(myItkImage, int(binderThickness), sitk.sitkBall, 0.0, 1.0,  False)   
        myItkImage = sitk.BinaryErode(myItkImage, int(binderThickness), sitk.sitkBall, 0.0, 1.0,  False)   
        binder=sitk.GetArrayFromImage(myItkImage).astype(np.bool)     
    
        binder = np.logical_and(binder,np.logical_not(image))    
        
        binderTotalVolume = float(np.count_nonzero(binder))/np.size(image)
        
        binderThickness += 1
        
#TODO : faire recherche dichotomique du bon binderThickness        
#                =(a+b)/2. # Prendre le milieu de [a,b]
#if f(c)==0:
#    return c
#    if f(a)>0:
#        if f(c)>0:
#            a, b = c, b # dichotomie a droite
#            else:
#                a, b = a, c # dichotomie a gauche
#                else:
#                    if f(c)>0:
#                        a, b = a, c # dichotomie  gauche
#                    else:
#                        a, b = c, b
        
    image = image.astype(np.uint8)
    image[binder] = 2    
    
    endTime=time.time()
    print("Time spent : {} s".format(endTime-beginTime))
    
    return image
    
    
    
#--------------------------------------------------------------------
def CreateVirtualCatalystLayer(voxelNumbers,carbonGrainRadius,nCarbonGrain,voidRadius,nVoid,
                             nafionThickness=1,nafionCoveragePercentage=0,randomSeed=1):  
    
    print('Create Virtual Active Layer')
    beginTime=time.time()
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
    
    endTime=time.time()
    print("Time spent : {} s".format(endTime-beginTime))
    
    return image


#--------------------------------------------------------------------
def CreateVirtualAgglomerate(voxelNumbers,grainRadius,nGrain,voidRadius,nVoid,
                             nafionThickness=1,nafionCoveragePercentage=0,randomSeed=1):


    return 1


#--------------------------------------------------------------------
def CreateVirtualLayerWithCracks(voxelNumbers,voidRadius,nVoid,crackLength,
                                 nCrack,randomSeed=1):
    
    #couche uniforme avec des vides aléatoires
    
    
    #ajouter cracks    
    
    
    return 1




#--------------------------------------------------------------------
def CreateVirtualInterfaceGDLMPL(randomSeed=0):

    #Create GDL Image
    gdl = CreateVirtualGDL(voxelNumbers=(500,500,200),
                           nFiber=100,fiberRadius=9,fiberLength=500,
                           binderThickness=8,anisotropy=5,randomSeed=randomSeed)

    #Zoom on the bottom of GDL with resampling


    #Create MPL image
   
    
    #Add the two images

    #cas juxtaposition :
    #mettre une image au dessus de l'autre
#
#    mpl = CreateVirtualLayerWithCracks(voxelNumbers=voxelNumbers,
#                                       voidRadius,nVoid,
#                                       crackLength,nCrack,
#                                       randomSeed=1)
#                             


    #cas superposition : 
    #mettre la mpl, superposer la GDL. Superposition sur une epaisseur x




    #cas intrusion de la MPL dans les pores exterieurs de la GDL :
    #trouver les pores exterieurs avec full morphology. Ajouter la MPL dans le 
    #masque défini par ces pores.

#    wholeImage = 
#    top = 
    
    wholeImage[top] = gdl

    import FullMorphology
    invadedVoxels = FullMorphology(wholeImage,voxelLength=2,
                                   pressureList=1.0/10,pressureCode=110,gamma=1)

    mpl = invadedVoxels==110

    wholeImage[mpl] = mplCode
    
        #add micro porosity






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
    beginTime=time.time()
    
    #Prepare the structure image
    image=np.zeros(voxelNumbers,dtype=np.bool)
    
    gridX=np.linspace(imageBounds[0],imageBounds[1],voxelNumbers[0]+1)
    gridY=np.linspace(imageBounds[2],imageBounds[3],voxelNumbers[1]+1)
    gridZ=np.linspace(imageBounds[4],imageBounds[5],voxelNumbers[2]+1)
    

    vertices,fibres = ComputeVoronoiPoints(nPoint,anisotropy,imageBounds,randomSeed)
    

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
    
    endTime=time.time()
    print("Time spent : {} s".format(endTime-beginTime))
    
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
    source.SetResolution(8)
    
    polydata=source.GetOutput()
    polydata.Update()

    #Perform rotation to get the rigth axis
    oldAxis = (0,1,0)
    polydata = MeshRotate(polydata,oldAxis,axis)
    
    #Perform translation to get the rigth center
    translationVector = center
    polydata = MeshTranslate(polydata,translationVector)
    
    triFilter=vtk.vtkTriangleFilter()
    triFilter.SetInput(polydata)
    triFilter.Update()
    polydata=triFilter.GetOutput() 
    polydata.Update()
    
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
def Voxelize(vtkPolyDataObject,gridX,gridY,gridZ,raydirection='xyz'):
    #Voxelize the object on a window adapted to its bounds. The windows will 
    #be chosed to be a subsample or an extention of the whole image voxel 
    #gridX,Y,Z. This function uses VTK VoxelModel to voxelize the surface, 
    #then the FillInside function to fill the inside.

    #Prepare a subwindows zooming on the object
    subWindowBound=vtkPolyDataObject.GetBounds()    
    nVoxSubImage,boundSubgrid,gridRelativePosition = GetSubWindowInformation(
                                                        subWindowBound,gridX,gridY,gridZ)
    
    
    subImage=VoxelizeRayTracing(vtkPolyDataObject,nVoxSubImage,boundSubgrid,raydirection)
    

    #Get back to the original window
    nVoxImage = (len(gridX)-1,len(gridY)-1,len(gridZ)-1)    
    wholeImage = InsertSubimageInImage(subImage,nVoxImage,gridRelativePosition)
    
    return wholeImage.astype(np.bool)


#-----------------------------------------------------------------------------
def VoxelizeRayTracing(vtkPolyDataObject,nVoxSubImage,boundSubgrid,raydirection='z'):

    #Count the number of voxels in each direction:

#    points = numpy_support.vtk_to_numpy(vtkPolyDataObject.GetPoints().GetData())
#    polys = numpy_support.vtk_to_numpy(vtkPolyDataObject.GetPolys().GetData())
#    
#    nTriangle=len(polys)/4
#    meshXYZ=np.zeros((nTriangle,3,3),dtype=np.float)
#    for iTri in range(nTriangle):
#        for iVert in range(3):
#                assert polys[4*iTri]==3
#                meshXYZ[iTri,:,iVert]=points[polys[4*iTri+iVert+1]]
    
    gridCOx=np.linspace(boundSubgrid[0],boundSubgrid[1],nVoxSubImage[0])
    gridCOy=np.linspace(boundSubgrid[2],boundSubgrid[3],nVoxSubImage[1])
    gridCOz=np.linspace(boundSubgrid[4],boundSubgrid[5],nVoxSubImage[2])


    voxcountX,voxcountY,voxcountZ = nVoxSubImage[0],nVoxSubImage[1],nVoxSubImage[2]

    
    # Prepare logical array to hold the voxelised data:
    gridOUTPUT = np.zeros((voxcountX,voxcountY,voxcountZ,len(raydirection)),dtype=np.bool)
    countdirections = 0;
    
    if raydirection.find('x')>-1:
      countdirections = countdirections + 1
      rotatedMesh = meshXYZ[:,[1,2,0],:]
      gridOUTPUT[:,:,:,countdirections-1] = np.transpose( 
          VoxelizeRayTracingZDirection(rotatedMesh,gridCOy,gridCOz,gridCOx),
                                  axes=[2,0,1] )
    
    if raydirection.find('y')>-1:
      countdirections = countdirections + 1
      rotatedMesh = meshXYZ[:,[2,0,1],:]
      gridOUTPUT[:,:,:,countdirections-1] = np.transpose( 
          VoxelizeRayTracingZDirection(rotatedMesh,gridCOz,gridCOx,gridCOy),
                                  axes=[1,2,0] )
    
    if raydirection.find('z')>-1:
      countdirections = countdirections + 1
      gridOUTPUT[:,:,:,countdirections-1] = VoxelizeRayTracingZDirectionVTK(
                                  vtkPolyDataObject,gridCOx,gridCOy,gridCOz)
    
    # Combine the results of each ray-tracing direction:
    if len(raydirection)>1:
        gridOUTPUT = np.sum(gridOUTPUT,axis=3)>=len(raydirection)/2.0
    else:
        gridOUTPUT = gridOUTPUT[:,:,:,0]
        
    return gridOUTPUT
    


#-----------------------------------------------------------------------------
def VoxelizeRayTracingZDirectionVTK(polydata,gridCOx,gridCOy,gridCOz):
    
    #Identify the min and max x,y coordinates (cm) of the mesh:
#    meshXmin = meshXYZ[:,0,:].min()
#    meshXmax = meshXYZ[:,0,:].max()
#    meshYmin = meshXYZ[:,1,:].min()
#    meshYmax = meshXYZ[:,1,:].max()
#    meshZmin = meshXYZ[:,2,:].min()
#    meshZmax = meshXYZ[:,2,:].max()
    
    meshXmin,meshXmax,meshYmin,meshYmax,meshZmin,meshZmax = polydata.GetBounds()
    assert (meshXmin<meshXmax and meshYmin<meshYmax and meshZmin<meshZmax)
    
    #Identify the min and max x,y coordinates (pixels) of the mesh:
    meshXminp = np.nonzero(abs(gridCOx-meshXmin)==min(abs(gridCOx-meshXmin)))[0][0]
    meshXmaxp = np.nonzero(abs(gridCOx-meshXmax)==min(abs(gridCOx-meshXmax)))[0][0]
    meshYminp = np.nonzero(abs(gridCOy-meshYmin)==min(abs(gridCOy-meshYmin)))[0][0]
    meshYmaxp = np.nonzero(abs(gridCOy-meshYmax)==min(abs(gridCOy-meshYmax)))[0][0]
        
    #meshXYZmin = np.min(meshXYZ,axis=2)
    #meshXYZmax = np.max(meshXYZ,axis=2)    
        
    sampleDimensions=(len(gridCOx),len(gridCOy),len(gridCOz))
    image=np.zeros(sampleDimensions,dtype=np.bool)
    #Identify the min and max x,y,z coordinates of each facet:
    
    
    correctionLIST = []    
    
    #Construct octree of the mesh for fast access to its geometry
    obbTree = vtk.vtkOBBTree()
    obbTree.SetDataSet(polydata)
    obbTree.BuildLocator()
    
    for loopY in xrange(meshYminp,meshYmaxp+1):
        for loopX in xrange(meshXminp,meshXmaxp+1):
            
            epsilon=(meshZmax-meshZmin)/10
            pSource=[gridCOx[loopX],gridCOy[loopY],meshZmax+epsilon]
            pTarget=[gridCOx[loopX],gridCOy[loopY],meshZmin-epsilon]
            pointsIntersection=VTKRayCasting(obbTree,pSource,pTarget)
            
            
            if len(pointsIntersection)>0: 

              gridCOzCROSS=np.asarray([pointsIntersection[i][2] for i in range(len(pointsIntersection))])
                
              #Remove values of gridCOzCROSS which are outside of the mesh limits (including a 1e-12 margin for error).
              gridCOzCROSS = gridCOzCROSS[ np.logical_and(
                  np.greater_equal(gridCOzCROSS,(meshZmin-1e-12)*np.ones(gridCOzCROSS.shape)),
                  np.less_equal(gridCOzCROSS,(meshZmax+1e-12)*np.ones(gridCOzCROSS.shape))) ]
          
              #Round gridCOzCROSS to remove any rounding errors, and take only the unique values:
              gridCOzCROSS = np.round(gridCOzCROSS*1e12)/1e12
              gridCOzCROSS = np.unique(gridCOzCROSS)
    
              if gridCOzCROSS.size%2 == 0: 
                  for loopASSIGN in range(gridCOzCROSS.size/2):
                      voxelsINSIDE = np.logical_and(
                                    np.greater(gridCOz,gridCOzCROSS[2*loopASSIGN]), 
                                    np.less(gridCOz,gridCOzCROSS[2*loopASSIGN+1]))
                                    
                      image[loopX,loopY,voxelsINSIDE] = 1
              elif len(gridCOzCROSS)>0:
                  correctionLIST.append([loopX,loopY])
        
    
    # USE INTERPOLATION TO FILL IN THE RAYS WHICH COULD NOT BE VOXELISED
    #For rays where the voxelisation did not give a clear result, the ray is
    #computed by interpolating from the surrounding rays.    

    image=InterpolateRemainingVoxels(correctionLIST,sampleDimensions,image)
    
    return image





#-----------------------------------------------------------------------------
def VTKRayCasting(polydataObbTree,pSource,pTarget):
#https://pyscience.wordpress.com/2014/09/21/ray-casting-with-python-and-vtk-intersecting-linesrays-with-surface-meshes/


#    pSource = [loopX, loopY, +10]
#    pTarget = [loopX, loopY, -10]
    
    pointsVTKintersection = vtk.vtkPoints()
    code = polydataObbTree.IntersectWithLine(pSource, pTarget, pointsVTKintersection, None)

    #check code
#    if code==0:
#        1
#    elif code==-1:
#        1
#    elif code==1:
#        1
   #"Take the passed line segment and intersect it with the data set. This method 
   #assumes that the data set is a vtkPolyData that describes a closed surface, 
   #and the intersection points that are returned in ‘points’ alternate between 
   #entrance points and exit points. The return value of the function is 0 if no 
   #intersections were found, –1 if point ‘a0’ lies inside the closed surface, or 
   #+1 if point ‘a0’ lies outside the closed surface. Either ‘points’ or ‘cellIds’ 
   #can be set to NULL if you don’t want to receive that information."

    pointsVTKIntersectionData = pointsVTKintersection.GetData()
    noPointsVTKIntersection = pointsVTKIntersectionData.GetNumberOfTuples()
#   pointsIntersection = []
#    for idx in range(noPointsVTKIntersection):
#        _tup = pointsVTKIntersectionData.GetTuple3(idx)
#        pointsIntersection.append(_tup)

    pointsIntersection=[pointsVTKIntersectionData.GetTuple3(idx) 
                                for idx in range(noPointsVTKIntersection)]
    
    return pointsIntersection


#-----------------------------------------------------------------------------
def InterpolateRemainingVoxels(correctionLIST,sampleDimensions,image):
    
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
    return image





#
#
#
##-----------------------------------------------------------------------------
#def VoxelizeRayTracingZDirection(meshXYZ,gridCOx,gridCOy,gridCOz):
#
#    #% Loop through each x,y pixel.
#    #% The mesh will be voxelised by passing rays in the z-direction through
#    #% each x,y pixel, and finding the locations where the rays cross the mesh.
#    #This function is inspired by some parts of the Matlab file exchange function
#    #VOXELISE (AUTHOR Adam H. Aitkenhead, The Christie NHS Foundation Trust) 
#    
#    
#    #Identify the min and max x,y coordinates (cm) of the mesh:
#    meshXmin = meshXYZ[:,0,:].min()
#    meshXmax = meshXYZ[:,0,:].max()
#    meshYmin = meshXYZ[:,1,:].min()
#    meshYmax = meshXYZ[:,1,:].max()
#    meshZmin = meshXYZ[:,2,:].min()
#    meshZmax = meshXYZ[:,2,:].max()
#    
#    #Identify the min and max x,y coordinates (pixels) of the mesh:
#    meshXminp = np.nonzero(abs(gridCOx-meshXmin)==min(abs(gridCOx-meshXmin)))[0][0]
#    meshXmaxp = np.nonzero(abs(gridCOx-meshXmax)==min(abs(gridCOx-meshXmax)))[0][0]
#    meshYminp = np.nonzero(abs(gridCOy-meshYmin)==min(abs(gridCOy-meshYmin)))[0][0]
#    meshYmaxp = np.nonzero(abs(gridCOy-meshYmax)==min(abs(gridCOy-meshYmax)))[0][0]
#        
#    meshXYZmin = np.min(meshXYZ,axis=2)
#    meshXYZmax = np.max(meshXYZ,axis=2)    
#        
#    sampleDimensions=(len(gridCOx),len(gridCOy),len(gridCOz))
#    image=np.zeros(sampleDimensions,dtype=np.bool)
#    #Identify the min and max x,y,z coordinates of each facet:
#    
#    
#    correctionLIST = []    
#    
#    
#    for loopY in range(meshYminp,meshYmaxp+1):
#    
#        #- 1a - Find which mesh facets could possibly be crossed by the ray:
#        possibleCROSSLISTy = np.nonzero(np.logical_and(
#                                      np.less_equal(meshXYZmin[:,1],gridCOy[loopY]), 
#                                      np.greater_equal(meshXYZmax[:,1],gridCOy[loopY]) ))[0]
#        
#        for loopX in range(meshXminp,meshXmaxp+1):
#        
#            #- 1b - Find which mesh facets could possibly be crossed by the ray:
#            possibleCROSSLIST = possibleCROSSLISTy[ np.logical_and(
#                       np.less_equal(meshXYZmin[possibleCROSSLISTy,0],gridCOx[loopX]),
#                       np.greater_equal(meshXYZmax[possibleCROSSLISTy,0],gridCOx[loopX]))]
#        
#            if len(possibleCROSSLIST)>0:  #Only continue the analysis if some nearby 
#                                          #facets were actually identified
#                  
#        #      % - 2 - For each facet, check if the ray really does cross the facet rather than just passing it close-by:
#                  
#        #      % GENERAL METHOD:
#        #      % A. Take each edge of the facet in turn.
#        #      % B. Find the position of the opposing vertex to that edge.
#        #      % C. Find the position of the ray relative to that edge.
#        #      % D. Check if ray is on the same side of the edge as the opposing vertex.
#        #      % E. If this is true for all three edges, then the ray definitely passes through the facet.
#        #      %
#        #      % NOTES:
#        #      % A. If a ray crosses exactly on a vertex:
#        #      %    a. If the surrounding facets have normal components pointing in the same (or opposite) direction as the ray then the face IS crossed.
#        #      %    b. Otherwise, add the ray to the correctionlist.
#              
#              facetCROSSLIST = []   #Prepare to record all facets which are crossed by the ray.
#                                    #This array is built on-the-fly, but since
#                                    #it ought to be relatively small (typically a list of <10) 
#                                    #should not incur too much of a speed penalty.
#              
#        #      %----------
#        #      % - 1 - Check for crossed vertices:
#        #      %----------
#              possibleCROSSLIST, correctionLIST,facetCROSSLIST = CheckForCrossedVertices(
#                                  meshXYZ,
#                                  possibleCROSSLIST,correctionLIST,facetCROSSLIST,
#                                  gridCOx,gridCOy,loopX,loopY)
#                                        
#        #      %----------
#        #      % - 2 - Check for crossed facets:
#        #      %----------
#              facetCROSSLIST= CheckForCrossedFacets(meshXYZ,
#                                              possibleCROSSLIST,facetCROSSLIST,
#                                              gridCOx,gridCOy,loopX,loopY)
#              
#        #        %----------
#        #        % - 3 - Find the z coordinate of the locations where the ray crosses each facet or vertex:
#        #        %----------
#              gridCOzCROSS=FindCrossingLocation(meshXYZ,facetCROSSLIST,
#                                                gridCOx,gridCOy,loopX,loopY)  
#                
#                
#              #%Remove values of gridCOzCROSS which are outside of the mesh limits (including a 1e-12 margin for error).
#              gridCOzCROSS = gridCOzCROSS[ np.logical_and(
#                  np.greater_equal(gridCOzCROSS,(meshZmin-1e-12)*np.ones(gridCOzCROSS.shape)),
#                  np.less_equal(gridCOzCROSS,(meshZmax+1e-12)*np.ones(gridCOzCROSS.shape))) ]
#          
#              #Round gridCOzCROSS to remove any rounding errors, and take only the unique values:
#              gridCOzCROSS = np.round(gridCOzCROSS*1e12)/1e12
#              gridCOzCROSS = np.unique(gridCOzCROSS)
#    
#              if gridCOzCROSS.size%2 == 0: 
#                  for loopASSIGN in range(gridCOzCROSS.size/2):
#                      voxelsINSIDE = np.logical_and(
#                                    np.greater(gridCOz,gridCOzCROSS[2*loopASSIGN]), 
#                                    np.less(gridCOz,gridCOzCROSS[2*loopASSIGN+1]))
#                                    
#                      image[loopX,loopY,voxelsINSIDE] = 1
#              elif len(gridCOzCROSS)>0:
#                  correctionLIST.append([loopX,loopY])
#        
#    
#    # USE INTERPOLATION TO FILL IN THE RAYS WHICH COULD NOT BE VOXELISED
#    #For rays where the voxelisation did not give a clear result, the ray is
#    #computed by interpolating from the surrounding rays.    
#
#    image=InterpolateRemainingVoxels(correctionLIST,sampleDimensions,image)
#    
#    return image  
#
#
#
#def CheckForCrossedVertices(meshXYZ,possibleCROSSLIST,correctionLIST,facetCROSSLIST,
#                            gridCOx,gridCOy,loopX,loopY):
##      % Find which mesh facets contain a vertex which is crossed by the ray:
#  vertexCROSSLIST = possibleCROSSLIST[np.logical_or(np.logical_or(
#                   np.logical_and(meshXYZ[possibleCROSSLIST,0,0]==gridCOx[loopX], 
#                           meshXYZ[possibleCROSSLIST,1,0]==gridCOy[loopY]),
#                   np.logical_and(meshXYZ[possibleCROSSLIST,0,1]==gridCOx[loopX], 
#                           meshXYZ[possibleCROSSLIST,1,1]==gridCOy[loopY])),
#                   np.logical_and(meshXYZ[possibleCROSSLIST,0,2]==gridCOx[loopX], 
#                           meshXYZ[possibleCROSSLIST,1,2]==gridCOy[loopY]))
#                          ]
#  
#  if len(vertexCROSSLIST)>0:  #Only continue the analysis if potential 
#                              #vertices were actually identified
#
#      checkindex = np.zeros(vertexCROSSLIST.size)
#
#      while min(checkindex) == 0:
#      
#          vertexindex             = np.nonzero(checkindex==0)[0][0]
#          checkindex[vertexindex] = 1
#        
#          tempfaces,tempvertices = CONVERT_meshformat(meshXYZ[vertexCROSSLIST,:,:])
#          adjacentindex  = np.concatenate((
#              np.reshape(np.asarray([ np.sum(a[0] == tempfaces[vertexindex,0]) for a in tempfaces ],dtype=np.bool),(len(tempfaces),1)),
#              np.reshape(np.asarray([ np.sum(a[0] == tempfaces[vertexindex,0]) for a in tempfaces ],dtype=np.bool),(len(tempfaces),1))),
#              axis=1)
#          adjacentindex  = np.max(adjacentindex,axis=1)
#          checkindex[adjacentindex]  = 1;
#        
#          coN = COMPUTE_mesh_normals(meshXYZ[vertexCROSSLIST[adjacentindex],:,:])
#          #vtkPolyDataNormals
#          if max(coN[:,2])<0 or min(coN[:,2])>0:
#              facetCROSSLIST.append(vertexCROSSLIST[vertexindex])
#          else:
#              possibleCROSSLIST = []
#              correctionLIST.append([loopX,loopY])
#              checkindex[:]     = 1
#  
#  return possibleCROSSLIST, correctionLIST,facetCROSSLIST
#
#def CheckForCrossedFacets(meshXYZ,possibleCROSSLIST,facetCROSSLIST,gridCOx,gridCOy,loopX,loopY):
#  
#  if len(possibleCROSSLIST)>0:  #Only continue the analysis if some nearby 
#                                #facets were actually identified
#      
#      for loopCHECKFACET in np.transpose(possibleCROSSLIST):
#  
#      #Check if ray crosses the facet. Taking each edge of the facet in turn, 
#      #check if the ray is on the same side as the opposing vertex.
#    
#          Y1predicted = meshXYZ[loopCHECKFACET,1,1] - ((
#              meshXYZ[loopCHECKFACET,1,1]-meshXYZ[loopCHECKFACET,1,2])*(
#              meshXYZ[loopCHECKFACET,0,1]-meshXYZ[loopCHECKFACET,0,0])/(
#              meshXYZ[loopCHECKFACET,0,1]-meshXYZ[loopCHECKFACET,0,2]))
#          YRpredicted = meshXYZ[loopCHECKFACET,1,1] - ((
#              meshXYZ[loopCHECKFACET,1,1]-meshXYZ[loopCHECKFACET,1,2])*(
#              meshXYZ[loopCHECKFACET,0,1]-gridCOx[loopX])/(
#              meshXYZ[loopCHECKFACET,0,1]-meshXYZ[loopCHECKFACET,0,2]))
#        
#          if ((Y1predicted > meshXYZ[loopCHECKFACET,1,0] and
#              YRpredicted > gridCOy[loopY]) or 
#              (Y1predicted < meshXYZ[loopCHECKFACET,1,0] and
#              YRpredicted < gridCOy[loopY])):
#            #The ray is on the same side of the 2-3 edge as the 1st vertex.
#
#              Y2predicted = meshXYZ[loopCHECKFACET,1,2] - ((
#                  meshXYZ[loopCHECKFACET,1,2]-meshXYZ[loopCHECKFACET,1,0])*(
#                  meshXYZ[loopCHECKFACET,0,2]-meshXYZ[loopCHECKFACET,0,1])/(
#                  meshXYZ[loopCHECKFACET,0,2]-meshXYZ[loopCHECKFACET,0,0]))
#              YRpredicted = meshXYZ[loopCHECKFACET,1,2] - ((
#                  meshXYZ[loopCHECKFACET,1,2]-meshXYZ[loopCHECKFACET,1,0])*(
#                  meshXYZ[loopCHECKFACET,0,2]-gridCOx[loopX])/(
#                  meshXYZ[loopCHECKFACET,0,2]-meshXYZ[loopCHECKFACET,0,0]))
#              
#              if ((Y2predicted > meshXYZ[loopCHECKFACET,1,1] and 
#                  YRpredicted > gridCOy[loopY]) or
#                  (Y2predicted < meshXYZ[loopCHECKFACET,1,1] and
#                  YRpredicted < gridCOy[loopY])):
#                  #The ray is on the same side of the 3-1 edge as the 2nd vertex.
#    
#                  Y3predicted = meshXYZ[loopCHECKFACET,1,0] - ((
#                      meshXYZ[loopCHECKFACET,1,0]-meshXYZ[loopCHECKFACET,1,1])*(
#                      meshXYZ[loopCHECKFACET,0,0]-meshXYZ[loopCHECKFACET,0,2])/(
#                      meshXYZ[loopCHECKFACET,0,0]-meshXYZ[loopCHECKFACET,0,1]))
#                  YRpredicted = meshXYZ[loopCHECKFACET,1,0] - ((
#                      meshXYZ[loopCHECKFACET,1,0]-meshXYZ[loopCHECKFACET,1,1])*(
#                      meshXYZ[loopCHECKFACET,0,0]-gridCOx[loopX])/(
#                      meshXYZ[loopCHECKFACET,0,0]-meshXYZ[loopCHECKFACET,0,1]))
#                
#                  if ((Y3predicted > meshXYZ[loopCHECKFACET,1,2] and
#                      YRpredicted > gridCOy[loopY]) or 
#                      (Y3predicted < meshXYZ[loopCHECKFACET,1,2] and
#                      YRpredicted < gridCOy[loopY])):
#                    #The ray is on the same side of the 1-2 edge as the 3rd vertex.
#    
#                    #The ray passes through the facet since it is on the correct 
#                    #side of all 3 edges
#                      facetCROSSLIST.append(loopCHECKFACET)
#                      
#  return facetCROSSLIST    
#
#def FindCrossingLocation(meshXYZ,facetCROSSLIST,gridCOx,gridCOy,loopX,loopY):              
#          
#  facetCROSSLIST=np.asarray(facetCROSSLIST)
#  gridCOzCROSS = np.zeros(facetCROSSLIST.shape)
#  for loopFINDZ in facetCROSSLIST:
#
##          % METHOD:
##          % 1. Define the equation describing the plane of the facet.  For a
##          % more detailed outline of the maths, see:
##          % http://local.wasp.uwa.edu.au/~pbourke/geometry/planeeq/
##          %    Ax + By + Cz + D = 0
##          %    where  A = y1 (z2 - z3) + y2 (z3 - z1) + y3 (z1 - z2)
##          %           B = z1 (x2 - x3) + z2 (x3 - x1) + z3 (x1 - x2)
##          %           C = x1 (y2 - y3) + x2 (y3 - y1) + x3 (y1 - y2)
##          %           D = - x1 (y2 z3 - y3 z2) - x2 (y3 z1 - y1 z3) - x3 (y1 z2 - y2 z1)
##          % 2. For the x and y coordinates of the ray, solve these equations to find the z coordinate in this plane.
#
#      planecoA=(meshXYZ[loopFINDZ,1,0]*(meshXYZ[loopFINDZ,2,1]-meshXYZ[loopFINDZ,2,2])+ 
#            meshXYZ[loopFINDZ,1,1]*(meshXYZ[loopFINDZ,2,2]-meshXYZ[loopFINDZ,2,0])+
#            meshXYZ[loopFINDZ,1,2]*(meshXYZ[loopFINDZ,2,0]-meshXYZ[loopFINDZ,2,1]))
#      
#      planecoB = (meshXYZ[loopFINDZ,2,0]*(meshXYZ[loopFINDZ,0,1]-meshXYZ[loopFINDZ,0,2])+
#            meshXYZ[loopFINDZ,2,1]*(meshXYZ[loopFINDZ,0,2]-meshXYZ[loopFINDZ,0,0])+
#            meshXYZ[loopFINDZ,2,2]*(meshXYZ[loopFINDZ,0,0]-meshXYZ[loopFINDZ,0,1])) 
#      
#      planecoC = (meshXYZ[loopFINDZ,0,0]*(meshXYZ[loopFINDZ,1,1]-meshXYZ[loopFINDZ,1,2])+
#            meshXYZ[loopFINDZ,0,1]*(meshXYZ[loopFINDZ,1,2]-meshXYZ[loopFINDZ,1,0]) + 
#            meshXYZ[loopFINDZ,0,2]*(meshXYZ[loopFINDZ,1,0]-meshXYZ[loopFINDZ,1,1]))
#      
#      planecoD = (- meshXYZ[loopFINDZ,0,0]*(
#                meshXYZ[loopFINDZ,1,1]*meshXYZ[loopFINDZ,2,2]-
#                meshXYZ[loopFINDZ,1,2]*meshXYZ[loopFINDZ,2,1])-
#            meshXYZ[loopFINDZ,0,1]*(
#                meshXYZ[loopFINDZ,1,2]*meshXYZ[loopFINDZ,2,0]-
#                meshXYZ[loopFINDZ,1,0]*meshXYZ[loopFINDZ,2,2]) -
#            meshXYZ[loopFINDZ,0,2]*(
#                meshXYZ[loopFINDZ,1,0]*meshXYZ[loopFINDZ,2,1]-
#                meshXYZ[loopFINDZ,1,1]*meshXYZ[loopFINDZ,2,0]))
#
#      if abs(planecoC) < 1e-14 :
#          planecoC=0
#      
#      gridCOzCROSS[facetCROSSLIST==loopFINDZ] = (- planecoD - 
#            planecoA*gridCOx[loopX] - planecoB*gridCOy[loopY]) / planecoC
#    
#  return gridCOzCROSS 
#
#

#
#
#
##------------------------------------------------------------------------------
#def  CONVERT_meshformat(meshXYZ):
##%CONVERT_meshformat  Convert mesh data from array to faces,vertices format or vice versa
##%==========================================================================
##% AUTHOR        Adam H. Aitkenhead
##% CONTACT       adam.aitkenhead@christie.nhs.uk
##% INSTITUTION   The Christie NHS Foundation Trust
##%
##% USAGE         [faces,vertices] = CONVERT_meshformat(meshXYZ)
##%         or... [meshXYZ]        = CONVERT_meshformat(faces,vertices)
##%
##% IN/OUTPUTS    meshXYZ  - Nx3x3 array - An array defining the vertex
##%                          positions for each of the N facets, with: 
##%                            1 row for each facet
##%                            3 cols for the x,y,z coordinates
##%                            3 pages for the three vertices
##%
##%               vertices - Nx3 array   - A list of the x,y,z coordinates of
##%                          each vertex in the mesh.
##%
##%               faces    - Nx3 array   - A list of the vertices used in
##%                          each facet of the mesh, identified using the row
##%                          number in the array vertices.
##%==========================================================================
##
#
#    vertices = np.concatenate((meshXYZ[:,:,0],meshXYZ[:,:,1],meshXYZ[:,:,2]),axis=0) 
#    vertices = np.asarray(list(set([tuple(vertices[i,:])
#                                    for i in range(vertices.shape[0]) ])))
#
#    faces = np.zeros((meshXYZ.shape[0],2))
#
#    for loopF in range( 0,meshXYZ.shape[0]):
#        for loopV in range(0,2):
#
#            vertref = np.nonzero(vertices[:,0]==meshXYZ[loopF,0,loopV])[0]
#            vertref = vertref[vertices[vertref,1]==meshXYZ[loopF,1,loopV]]
#            vertref = vertref[vertices[vertref,2]==meshXYZ[loopF,2,loopV]]
#      
#            faces[loopF,loopV] = vertref
#
#    return faces,vertices
#
#
##------------------------------------------------------------------------------
#def COMPUTE_mesh_normals(meshdataIN):
##% COMPUTE_mesh_normals  Calculate the normals for each facet of a triangular mesh
##%==========================================================================
##% AUTHOR        Adam H. Aitkenhead
#
#    coordVERTICES = meshdataIN;
#
#    
#    #%======================
#    #% Initialise array to hold the normal vectors
#    #%======================
#    
#    facetCOUNT   = coordVERTICES.shape[0]
#    coordNORMALS = np.zeros((facetCOUNT,3))
#    
#   
#    
#    #%======================
#    #% Compute the normal vector for each facet
#    #%======================
#    
#    for loopFACE in range(0,facetCOUNT):
#      
#      #Find the coordinates for each vertex.
#        cornerA = coordVERTICES[loopFACE,range(3),0]
#        cornerB = coordVERTICES[loopFACE,range(3),1]
#        cornerC = coordVERTICES[loopFACE,range(3),2]
#      
#      #Compute the vectors AB and AC
#        AB = cornerB-cornerA
#        AC = cornerC-cornerA
#        
#      #Determine the cross product AB x AC
#        ABxAC = np.cross(AB,AC) 
#        
#      #Normalise to give a unit vector
#        ABxAC = ABxAC / np.linalg.norm(ABxAC)
#        coordNORMALS[loopFACE,range(3)] = ABxAC
#      
#    
#
#
#    return coordNORMALS
#




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
def VisualizeVolumeRendering(image):

    mlab.pipeline.volume(mlab.pipeline.scalar_field(image))

#    from tvtk.api import tvtk
#    data = image
#    i = tvtk.ImageData(spacing=(1, 1, 1), origin=(0, 0, 0))
#    i.point_data.scalars = data.ravel()
#    i.point_data.scalars.name = 'scalars'
#    i.dimensions = data.shape


#--------------------------------------------------------------------
def VisualizeCutPlanes(image):
    mlab.pipeline.image_plane_widget(mlab.pipeline.scalar_field(image),
                                plane_orientation='x_axes',
                                slice_index=10,
                            )
    mlab.pipeline.image_plane_widget(mlab.pipeline.scalar_field(image),
                                plane_orientation='y_axes',
                                slice_index=10,
                            )
    mlab.outline()


#--------------------------------------------------------------------
def ComputeVoronoiPoints(nPoint,anisotropy,imageBounds,randomSeed=0):    
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
    nPoint=200
    fiberRadius=0.05
    anisotropy=3
    randomSeed=0
    image=CreateVirtualVoronoiFoam(voxelNumbers,imageBounds,nPoint,fiberRadius,anisotropy,randomSeed)
    SaveImage(255*(image.astype(np.uint8)),'TestVoronoi.tif')

#--------------------------------------------------------------------
def TestVirtualGDL():
    image = CreateVirtualGDL(voxelNumbers=(500,500,200),fiberContent=0.1,fiberRadius=9,
                             fiberLength=500,binderContent=0.1,anisotropy=5,randomSeed=0) 
    SaveImage(100*(image.astype(np.uint8)),'TestBigGDL.tif')
    
   
#--------------------------------------------------------------------    
def TestVirtualCCL():
    image = CreateVirtualCatalystLayer((200,200,200),4,5000,50,200,2,30)
    SaveImage(100*(image.astype(np.uint8)),'TestCCL.tif')
    
    
#--------------------------------------------------------------------    
def TestVoxelizePython():
    mesh=CreateCylinder((0,0,0),(1,0,0),15,100)
    voxelNumbers=(100,100,100)
    bounds=mesh.GetBounds()
    gridX=np.linspace(bounds[0],bounds[1],voxelNumbers[0])
    gridY=np.linspace(bounds[3],bounds[2],voxelNumbers[1])
    gridZ=np.linspace(bounds[5],bounds[4],voxelNumbers[2])
    image=Voxelize(mesh,gridX,gridY,gridZ)
    SaveImage(100*(image.astype(np.uint8)),'TestVoxelizePython.tif')
    
#--------------------------------------------------------------------        
def TestVisualization():
    image = CreateVirtualGDL(voxelNumbers=(400,400,100),fiberContent=0.2,fiberRadius=9,
                             fiberLength=400,binderContent=0.05,anisotropy=5,randomSeed=0)

    VisualizeVolumeRendering(image)
    VisualizeCutPlanes(image)
    
    
    
    
#--------------------------------------------------------------------
if __name__ == "__main__":
    TestVisualization()