# -*- coding: utf-8 -*-

import numpy as np
#import vtk
#from vtk.util import numpy_support
import math
#from scipy import ndimage
import random
import SimpleITK as sitk
import time
import vtk

from VirtualMaterials.Simulation  import FullMorphology
from VirtualMaterials.VirtualImages  import BasicShapes
from VirtualMaterials.VirtualImages  import Voxelization
import VirtualMaterials as vmat


#--------------------------------------------------------------------
#      Virtual Materials
#--------------------------------------------------------------------


#--------------------------------------------------------------------
def CreateVirtualGDL(voxelNumbers=(200,200,200),
                     fiberContent=0.5,fiberRadius=10,fiberLength=100,
                     binderContent=0.3,anisotropy=1,randomSeed=1,
                     distinctFibers=False):
    """ Creates a virtual GDL with straigth fibers                    
    :param: voxelNumbers : virtual image size (default:(200,200,200))
    :param: fiberContent : volume fraction of fibers (default:0.5)
    :param: fiberRadius : fiber radius (unit=voxel) (default:10 )
    :param: fiberLength : fiber length (unit=voxel) (default:100)
    :param: binderContent : volume fraction of binder (default:0.3)
    :param: anisotropy : orientation of fibers (default:1)
    :param: randomSeed :    (default:11)
    :param: distinctFibers : True (each fiber has a different code) or False (default: False) 
    
    :return: virtual image (numpy array)
    
    :example:
    import VirtualMaterials as vmat
    virtualGDL=vmat.VirtualImages.VirtualMaterialsGeneration.CreateVirtualGDL(
                    voxelNumbers=(200,200,200),
                    fiberContent=0.5,fiberRadius=10,fiberLength=100,
                    binderContent=0.3,anisotropy=1,randomSeed=1,
                    distinctFibers=False)
    
    """
    #Algorithm :add fibers until a given fiber content is achieved. 
    #   Add binder until a given porosity is achieved    
    
    
    print('Create Virtual GDL')
    beginTime=time.time()
    
    binderCode = 1
    fiberCode = 2
    
    random.seed(randomSeed)

    assert(fiberContent+binderContent<=1)
    
    image=np.zeros(voxelNumbers,dtype=np.uint16)
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
        
        mesh = BasicShapes.CreateCylinder(center,axis,fiberRadius,fiberLength)
        objImage = Voxelization.Voxelize(mesh,gridX,gridY,gridZ,raydirection='z')
        
        if distinctFibers:
            fiberCode = fiberCode+1
        else:
            fiberCode=2
            
        image[objImage>0] = fiberCode    
            
        fiberTotalVolume = float(np.count_nonzero(image))/np.size(image)        
        print fiberTotalVolume,
        
    #Add binder
    
    print('Adding binder until binderContent %f is reached' %binderContent)
    
    def GetBinder(allFibers,binderThickness):
        myItkImage = sitk.GetImageFromArray(allFibers.astype(np.uint8))
        myItkImage = sitk.BinaryDilate(myItkImage, int(binderThickness), sitk.sitkBall, 0.0, 1.0,  False)   
        myItkImage = sitk.BinaryErode(myItkImage, int(binderThickness), sitk.sitkBall, 0.0, 1.0,  False)   
        binder=sitk.GetArrayFromImage(myItkImage).astype(np.bool)     
        binder = np.logical_and(binder,np.logical_not(allFibers))
        return binder
    
    #recherche dichotomique de binderThickness correspondant au binderContent demande
    binderTotalVolume = [0.0]
    binderThickness=[1]    
    
    allFibers = image>0     
    
    while binderTotalVolume[-1]<binderContent :
        
        binder = GetBinder(allFibers,binderThickness[-1])
        binderTotalVolume.append(float(np.count_nonzero(binder))/np.size(image))
        
        print binderTotalVolume[-1],      
        binderThickness.append(binderThickness[-1]+1)
        
        if binderTotalVolume[-1]<binderTotalVolume[-2]:
            print('Unable to reach binderContent %f. Best value : %f' %(binderContent,binderTotalVolume[-2]))
            break 
        
        
#guessBT =(upperBT+lowerBT)/2. 
#binder = GetBinder(image,binderThickness)
#binderTotalVolume = float(np.count_nonzero(binder))/np.size(image)
#if f(guessBT)==0:
#    return guessBT
#    if f(upperBT)>0:
#        if f(guessBT)>0:
#            upperBT, lowerBT = guessBT, lowerBT # dichotomie a droite
#            else:
#                upperBT, lowerBT = upperBT, guessBT # dichotomie a gauche
#                else:
#                    if f(guessBT)>0:
#                        upperBT, lowerBT = upperBT, guessBT # dichotomie  gauche
#                    else:
#                        upperBT, lowerBT = guessBT, lowerBT
    
    memoryType = vmat.Utilities.Utilities.BestMemoryType(fiberCode)
    image = image.astype(memoryType)
    
    image[binder] = binderCode    
    
    endTime=time.time()
    print("Time spent : {} s".format(endTime-beginTime))
    
    return image
    
    
    
#--------------------------------------------------------------------
def CreateVirtualCatalystLayer(voxelNumbers=(200,200,200),
                               carbonGrainRadius=4,nCarbonGrain=5000,
                               voidRadius=50,nVoid=200,
                               nafionThickness=2,nafionCoveragePercentage=30,
                               randomSeed=1):  
    
    """ Creates a Virtual Catalyst Layer using random balls and Nafion film
    
    :param: voxelNumbers : virtual image size
    :param: carbonGrainRadius
    :param: nCarbonGrain
    :param: voidRadius
    :param: nVoid
    :param: nafionThickness
    :param: nafionCoveragePercentage
    :param: randomSeed   
    
    :return: virtual image (numpy array)
    
    :example:
    import VirtualMaterials as vmat
    virtualCL=vmat.VirtualImages.VirtualMaterialsGeneration.CreateVirtualCatalystLayer(
    voxelNumbers=(200,200,200),
                       carbonGrainRadius=4,nCarbonGrain=5000,
                       voidRadius=50,nVoid=200,
                       nafionThickness=2,nafionCoveragePercentage=30,
                       randomSeed=1)
    """
    
    
    
    
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
        objImage=BasicShapes.CreateVoxelizedBallFast(center,voidRadius,voxelNumbers,bounds)            
#        mesh = CreateBall(center,voidRadius)
#        objImage=Voxelization.Voxelize(mesh,gridX,gridY,gridZ)
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
#            objImage=Voxelization.Voxelize(mesh,gridX,gridY,gridZ)
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
#            objImage=Voxelization.Voxelize(mesh,gridX,gridY,gridZ)
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
def CreateVirtualInterfaceGDLMPL(voxelNumbers=(400,400,300),
                                 penetrationLength=15):
    """ Creates a Virtual Interface GDL MPL using a virtual GDL and
    MPL injected at the bottom of the GDL with full morphology algorithm
    
    :param: voxelNumbers : virtual image size
    :param: penetrationLength : diameter of full morphology MPL balls, in voxel unit

    :return: virtual image (numpy array)
    
    :example:
    import VirtualMaterials as vmat
    virtualCL=vmat.VirtualImages.VirtualMaterialsGeneration.CreateVirtualInterfaceGDLMPL(
                    voxelNumbers=(400,400,300),
                    penetrationLength=15)
    """
    imageBoundaryThickness = 50
    
    
    imageSize = voxelNumbers
    boundary = imageBoundaryThickness 
    
    #Create GDL Image
    GDLVoxelNumbers = (imageSize[0]+2*boundary,
                       imageSize[1]+2*boundary,
                       imageSize[2]/2)
    
    gdl = CreateVirtualGDL(voxelNumbers=GDLVoxelNumbers,
                           fiberContent=0.2,fiberRadius=9,fiberLength=GDLVoxelNumbers[0],
                           binderContent=0.05,anisotropy=4,randomSeed=0)

    #Create interface image with GDL on top
    
    interfaceVoxelNumber = (GDLVoxelNumbers[0]-2*boundary,
                            GDLVoxelNumbers[1]-2*boundary,
                            2*GDLVoxelNumbers[2])
    interface = np.zeros(interfaceVoxelNumber,dtype=np.uint8)

    interface[:,:,:GDLVoxelNumbers[2]]=gdl[boundary:GDLVoxelNumbers[0]-boundary,
                                           boundary:GDLVoxelNumbers[1]-boundary,
                                           :GDLVoxelNumbers[2]]

    #Add MPL

    invadedVoxels = FullMorphology.FullMorphology(interface,inletFace=5,voxelLength=2,
                                    pressureList=[1.0/penetrationLength],
                                    pressureCode=[110],gamma=1)

    mpl = invadedVoxels==110
    
    mplCode = 3
    
    interface[mpl] = mplCode
        

    return interface




#--------------------------------------------------------------------
def CreateVirtualVoronoiFoam(voxelNumbers=(200,200,200),imageBounds=(0.0,1.0,0.0,1.0,0.0,1.0),
                             nPoint=200,fiberRadius=0.05,
                             anisotropy=1,randomSeed=0):
    
    """ Creates a Virtual Voronoi foam 
    
    :param: voxelNumbers : virtual image size
    :param: imageBounds : physical dimensions of virtual sample
    :param: nPoint : number of voronoi seed points
    :param: fiberRadius : radius of cylinders around voronoi mesh edges (unit=voxel)
    :param: anisotropy : controls a deformation of the voronoi mesh in the z direction 
    :param: randomSeed
    
    :return: virtual image (numpy array)
    
    :example:
    import VirtualMaterials as vmat
    virtualCL=vmat.VirtualImages.VirtualMaterialsGeneration.CreateVirtualVoronoiFoam(
            voxelNumbers=(200,200,200),
            imageBounds=(0.0,1.0,0.0,1.0,0.0,1.0),
            nPoint=200,fiberRadius=0.05,
            anisotropy=1,randomSeed=0)
    """
    
    
    print('Create Voronoi')
    beginTime=time.time()
    
    #Prepare the structure image
    image=np.zeros(voxelNumbers,dtype=np.bool)
    
    gridX=np.linspace(imageBounds[0],imageBounds[1],voxelNumbers[0]+1)
    gridY=np.linspace(imageBounds[2],imageBounds[3],voxelNumbers[1]+1)
    gridZ=np.linspace(imageBounds[4],imageBounds[5],voxelNumbers[2]+1)
    

    vertices,fibres = BasicShapes.ComputeVoronoiPoints(nPoint,anisotropy,imageBounds,randomSeed)
    

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
        mesh = BasicShapes.CreateCylinder(center,axis,fiberRadius,height)
        objImage=Voxelization.Voxelize(mesh,gridX,gridY,gridZ,raydirection='xz')
        image=np.logical_or(image,objImage)
      
      
    #spherical capings of cylinders      
    for iVertice in range(nVertice):
        center = vertices[iVertice]
        
        mesh = BasicShapes.CreateBall(tuple(center),fiberRadius)
        objImage=Voxelization.Voxelize(mesh,gridX,gridY,gridZ,raydirection='xz')
        image=np.logical_or(image,objImage)

    #image = InsertSubimageInImage(image,voxelNumbers,gridRelativePosition)
    
    endTime=time.time()
    print("Time spent : {} s".format(endTime-beginTime))
    
    return image

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
                  
        mesh = BasicShapes.CreateCylinder(center,holeAxis,holeRadius,holeHeight)
        objImage=Voxelization.Voxelize(mesh,gridX,gridY,gridZ)
        gdlImage=np.logical_and(gdlImage,np.logical_not(objImage))
        
    return gdlImage







#--------------------------------------------------------------------
def CreateVirtualAgglomerate(voxelNumbers,grainRadius,nGrain,voidRadius,nVoid,
                             nafionThickness=1,nafionCoveragePercentage=0,randomSeed=1):
    """ Not implemented """                             
    print('Not implemented')
    return 1


#--------------------------------------------------------------------
def CreateVirtualLayerWithCracks(voxelNumbers,voidRadius,nVoid,crackLength,
                                 nCrack,randomSeed=1):
    """ Not implemented """                             
    print('Not implemented')
    #couche uniforme avec des vides aléatoires
    
    
    #ajouter cracks    
    
    
    return 1



#--------------------------------------------------------------------
def CreateVirtualInterfaceCatalystLayerMembrane(voxelNumbers,grainRadius,nGrain,voidRadius,nVoid,
                             nafionThickness=1,nafionCoveragePercentage=0,randomSeed=1):
    """ Not implemented """                             
    print('Not implemented')
    #Create Catalyst Layer Image



    #Create a piece of Membrane



    #Add the two images


    return 1



#--------------------------------------------------------------------
def CreateVirtualCeramicElectrode():
    """ Not implemented """                             
    print('Not implemented')
    #Create Higth temperature Fuel Cell Electrode
    
    #Add polyhedra
    
    return 1





#--------------------------------------------------------------------
def CreateVoxelizedTubeFilter(polydataMesh,voxelNumbers,fiberRadius):
    """ Creates an image putting cylinders around the edges of a mesh
    (see paraview tube filter to get an idea of what it's doing).
    :param: polydataMesh : mesh, used as the skeleton of the image (vtk polydata object)
    :param: voxelNumbers: virtual image size
    :param: fiberRadius : cylinder raidus (in voxel unit)
    
    :return: image (numpy array)
    
    :example:
    
    """
    
    
    print('Create FibersOnMesh')
    beginTime=time.time()
    
    #Prepare the structure image
    image=np.zeros(voxelNumbers,dtype=np.bool)
    
    imageBounds = polydataMesh.GetBounds()
    margin=fiberRadius    
    
    gridX=np.linspace(imageBounds[0]-margin,imageBounds[1]+margin,voxelNumbers[0]+1)
    gridY=np.linspace(imageBounds[2]-margin,imageBounds[3]+margin,voxelNumbers[1]+1)
    gridZ=np.linspace(imageBounds[4]-margin,imageBounds[5]+margin,voxelNumbers[2]+1)
    
    #Extract lines and points
    
    edgeExtractor = vtk.vtkExtractEdges()    
    
    if vtk.vtkVersion.GetVTKMajorVersion()==6:
        edgeExtractor.SetInputData(polydataMesh)
    else:
        edgeExtractor.SetInput(polydataMesh)
    edgeExtractor.Update()
    
    
    points = edgeExtractor.GetOutput().GetPoints()
    nVertice = int(points.GetNumberOfPoints())
    vertices=[(float(points.GetPoint(iPoint)[0]),
               float(points.GetPoint(iPoint)[1]),
               float(points.GetPoint(iPoint)[2]))
                   for iPoint in range(nVertice)]
    
    output = edgeExtractor.GetOutput()
    nFibre = int(output.GetNumberOfCells())
    fibres = [ [int(output.GetCell(iFibre).GetPointId(0)),
                int(output.GetCell(iFibre).GetPointId(1))]
                for iFibre in range(nFibre)]

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
        mesh = BasicShapes.CreateCylinder(center,axis,fiberRadius,height)
        objImage=Voxelization.Voxelize(mesh,gridX,gridY,gridZ,raydirection='xz')
        image=np.logical_or(image,objImage)
      
      
    #spherical capings of cylinders      
    for iVertice in range(nVertice):
        print iVertice,
        center = vertices[iVertice]
        
        mesh = BasicShapes.CreateBall(tuple(center),fiberRadius)
        objImage=Voxelization.Voxelize(mesh,gridX,gridY,gridZ,raydirection='xz')
        image=np.logical_or(image,objImage)

    #image = InsertSubimageInImage(image,voxelNumbers,gridRelativePosition)
    
    endTime=time.time()
    print("Time spent : {} s".format(endTime-beginTime))
    
    return image
    
 
 
#--------------------------------------------------------------------   
def CreateTestImage_Layers():
    """ CreateTestImage_Layers """  

    voxelNumbers=(100,100,100)
    image = np.zeros(voxelNumbers,dtype=np.uint8)
    
    zInds=np.arange(21,40,dtype=np.int)      
    image[:,:,zInds]=100
    
    zInds=np.arange(61,80,dtype=np.int)      
    image[:,:,zInds]=200

    return image
    
#--------------------------------------------------------------------   
def CreateTestImage_TetrahedronMedialAxis():
    """ CreateTestImage_TetrahedronMedialAxis """ 

    #Image parameters
    voxelNumbers = (100,100,100)
    
    image = 255*np.ones(voxelNumbers,dtype=np.uint8)
    bounds=(0.0, float(voxelNumbers[0]), 
        0.0, float(voxelNumbers[1]), 
        0.0, float(voxelNumbers[2]))
    
    gridX=np.linspace(bounds[0],bounds[1],voxelNumbers[0]+1)
    gridY=np.linspace(bounds[2],bounds[3],voxelNumbers[1]+1)
    gridZ=np.linspace(bounds[4],bounds[5],voxelNumbers[2]+1)           
           
           
    #Create elements meshes : 3 cylinders, 1 sphere      
    boxCenter = np.asarray( (50,50,50) )
    sphereRadius=20                 
    sphere = BasicShapes.CreateBall(boxCenter,sphereRadius)
    
    cylinderRadius = 10
    
    p = np.asarray( (100,0,0) )
    center = (p+boxCenter)/2.0
    axis = p-center
    heigth = 2*np.linalg.norm(axis)
    cylinder0 = BasicShapes.CreateCylinder(center,axis,cylinderRadius,heigth)
    
    p = np.asarray( (0,100,0) )
    center = (p+boxCenter)/2.0
    axis = p-center
    heigth = 2*np.linalg.norm(axis)
    cylinder1 = BasicShapes.CreateCylinder(center,axis,cylinderRadius,heigth)
    
    p = np.asarray( (0,0,100) )
    center = (p+boxCenter)/2.0
    axis = p-center
    heigth = 2*np.linalg.norm(axis)
    cylinder2 = BasicShapes.CreateCylinder(center,axis,cylinderRadius,heigth)
    
    p = np.asarray( (100,100,100) )
    center = (p+boxCenter)/2.0
    axis = p-center
    heigth = 2*np.linalg.norm(axis)
    cylinder3 = BasicShapes.CreateCylinder(center,axis,cylinderRadius,heigth)
    
    
    #Fill image with voxelized elements
    imSphere = Voxelization.Voxelize(sphere,gridX,gridY,gridZ,raydirection='z')
    
    imCylinder0 = Voxelization.Voxelize(cylinder0,gridX,gridY,gridZ,raydirection='z')

    imCylinder1 = Voxelization.Voxelize(cylinder1,gridX,gridY,gridZ,raydirection='z')

    imCylinder2 = Voxelization.Voxelize(cylinder2,gridX,gridY,gridZ,raydirection='z')

    imCylinder3 = Voxelization.Voxelize(cylinder3,gridX,gridY,gridZ,raydirection='z')


    image[imCylinder0] = 0
    image[imCylinder1] = 0
    image[imCylinder2] = 50
    image[imCylinder3] = 100
    image[imSphere] = 0
    
    
    return image
 
    
#--------------------------------------------------------------------   
def CreateTestImage_TwoBalls():
    """ CreateTestImage_TwoBalls """ 
    #Image parameters
    voxelNumbers = (30,30,30)
    
    image = 255*np.ones(voxelNumbers,dtype=np.uint8)
    bounds=(0.0, float(voxelNumbers[0]), 
        0.0, float(voxelNumbers[1]), 
        0.0, float(voxelNumbers[2]))
    
    gridX=np.linspace(bounds[0],bounds[1],voxelNumbers[0]+1)
    gridY=np.linspace(bounds[2],bounds[3],voxelNumbers[1]+1)
    gridZ=np.linspace(bounds[4],bounds[5],voxelNumbers[2]+1)           
           
           
    #Create elements meshes : 2 spheres 
    sphereRadius=10       
           
    center1 = np.asarray( (11,11,11) )
    sphere1 = BasicShapes.CreateBall(center1,sphereRadius)
    
    center2 = np.asarray( (19,19,19) )
    sphere2 = BasicShapes.CreateBall(center2,sphereRadius)   
    
    
    #Fill image with voxelized elements
    imSphere1 = Voxelization.Voxelize(sphere1,gridX,gridY,gridZ,raydirection='z')
    imSphere2 = Voxelization.Voxelize(sphere2,gridX,gridY,gridZ,raydirection='z')

    image[imSphere1] = 0
    image[imSphere2] = 0
    
    return image
    
    
#--------------------------------------------------------------------   
def CreateTestImage_ThreeBalls():
    """ CreateTestImage_ThreeBalls """   
                          
    #Image parameters
    voxelNumbers = (30,30,30)
    
    image = 255*np.ones(voxelNumbers,dtype=np.uint8)
    bounds=(0.0, float(voxelNumbers[0]), 
        0.0, float(voxelNumbers[1]), 
        0.0, float(voxelNumbers[2]))
    
    gridX=np.linspace(bounds[0],bounds[1],voxelNumbers[0]+1)
    gridY=np.linspace(bounds[2],bounds[3],voxelNumbers[1]+1)
    gridZ=np.linspace(bounds[4],bounds[5],voxelNumbers[2]+1)           
           
           
    #Create elements meshes : 2 spheres 
    sphereRadius=8       
           
    center1 = np.asarray( (19,11,11) )
    sphere1 = BasicShapes.CreateBall(center1,sphereRadius)
    
    center2 = np.asarray( (11,19,11) )
    sphere2 = BasicShapes.CreateBall(center2,sphereRadius)   

    center3 = np.asarray( (11,11,19) )
    sphere3 = BasicShapes.CreateBall(center3,sphereRadius)
    
    #Fill image with voxelized elements
    imSphere1 = Voxelization.Voxelize(sphere1,gridX,gridY,gridZ,raydirection='z')
    imSphere2 = Voxelization.Voxelize(sphere2,gridX,gridY,gridZ,raydirection='z')
    imSphere3 = Voxelization.Voxelize(sphere3,gridX,gridY,gridZ,raydirection='z')
    
    image[imSphere1] = 0
    image[imSphere2] = 0
    image[imSphere3] = 0
    
    return image
    
#--------------------------------------------------------------------   
def CreateTestImage_DiffuseSphericalInclusions(sphereRadius=5):
    """ CreateTestImage_DiffuseSphericalInclusions    
    :param: sphereRadius=8
    """                      
    #Image parameters
    voxelNumbers = (100,100,100)
    
    image = 0*np.ones(voxelNumbers,dtype=np.uint8)
    bounds=(0.0, float(voxelNumbers[0]), 
        0.0, float(voxelNumbers[1]), 
        0.0, float(voxelNumbers[2]))
    
    gridX=np.linspace(bounds[0],bounds[1],voxelNumbers[0]+1)
    gridY=np.linspace(bounds[2],bounds[3],voxelNumbers[1]+1)
    gridZ=np.linspace(bounds[4],bounds[5],voxelNumbers[2]+1)           
           
           
    #Create elements meshes : 2 spheres 
           
    center1 = np.asarray( (80,20,20) )
    sphere1 = BasicShapes.CreateBall(center1,sphereRadius)
    
    center2 = np.asarray( (20,80,20) )
    sphere2 = BasicShapes.CreateBall(center2,sphereRadius)   

    center3 = np.asarray( (20,20,80) )
    sphere3 = BasicShapes.CreateBall(center3,sphereRadius)
    
    #Fill image with voxelized elements
    imSphere1 = Voxelization.Voxelize(sphere1,gridX,gridY,gridZ,raydirection='z')
    imSphere2 = Voxelization.Voxelize(sphere2,gridX,gridY,gridZ,raydirection='z')
    imSphere3 = Voxelization.Voxelize(sphere3,gridX,gridY,gridZ,raydirection='z')
    
    image[imSphere1] = 255
    image[imSphere2] = 255
    image[imSphere3] = 255
    
    return image    
    
#--------------------------------------------------------------------   
def CreateTestImage_CylinderArray(cylinderRadius=10):
    """ CreateTestImage_CylinderArray    
    :param: cylinderRadius=8
    """                    
    
    #Image parameters
    voxelNumbers = (200,150,200)
    
    image = 0*np.ones(voxelNumbers,dtype=np.uint8)
    bounds=(0.0, float(voxelNumbers[0]), 
        0.0, float(voxelNumbers[1]), 
        0.0, float(voxelNumbers[2]))
    
    gridX=np.linspace(bounds[0],bounds[1],voxelNumbers[0]+1)
    gridY=np.linspace(bounds[2],bounds[3],voxelNumbers[1]+1)
    gridZ=np.linspace(bounds[4],bounds[5],voxelNumbers[2]+1)           
           
           
    #Create elements meshes : 2 spheres 
    axis = (0,0,1)       
    height = 1.1*voxelNumbers[2]
    resolution = int(math.ceil(6*cylinderRadius))
    
    center1 = np.asarray( (50,50,height/2.0) )
    cylinder1 = BasicShapes.CreateCylinder(center1,axis,cylinderRadius,height,resolution=resolution)
    
    center2 = np.asarray( (100,50,height/2.0) )
    cylinder2 = BasicShapes.CreateCylinder(center2,axis,cylinderRadius,height,resolution=resolution)   

    center3 = np.asarray( (150,50,height/2.0) )
    cylinder3 = BasicShapes.CreateCylinder(center3,axis,cylinderRadius,height,resolution=resolution)
    
    center4 = np.asarray( (50,100,height/2.0) )
    cylinder4 = BasicShapes.CreateCylinder(center4,axis,cylinderRadius,height,resolution=resolution)
    
    center5 = np.asarray( (100,100,height/2.0) )
    cylinder5 = BasicShapes.CreateCylinder(center5,axis,cylinderRadius,height,resolution=resolution)
    
    center6 = np.asarray( (150,100,height/2.0) )
    cylinder6 = BasicShapes.CreateCylinder(center6,axis,cylinderRadius,height,resolution=resolution)
    
    
    #Fill image with voxelized elements
    imCylinder1 = Voxelization.Voxelize(cylinder1,gridX,gridY,gridZ,raydirection='x')
    imCylinder2 = Voxelization.Voxelize(cylinder2,gridX,gridY,gridZ,raydirection='x')
    imCylinder3 = Voxelization.Voxelize(cylinder3,gridX,gridY,gridZ,raydirection='x')
    imCylinder4 = Voxelization.Voxelize(cylinder4,gridX,gridY,gridZ,raydirection='x')
    imCylinder5 = Voxelization.Voxelize(cylinder5,gridX,gridY,gridZ,raydirection='x')
    imCylinder6 = Voxelization.Voxelize(cylinder6,gridX,gridY,gridZ,raydirection='x')


    image[imCylinder1] = 255
    image[imCylinder2] = 255
    image[imCylinder3] = 255
    image[imCylinder4] = 255
    image[imCylinder5] = 255
    image[imCylinder6] = 255
    
    return image    
        
    
    