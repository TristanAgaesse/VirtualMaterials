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
from scipy.spatial import Voronoi
import time

from VirtualMaterials.Utilities  import tifffile as tff
from VirtualMaterials.Simulation  import FullMorphology
from VirtualMaterials.VirtualImages  import BasicShapes
from VirtualMaterials.VirtualImages  import Voxelization



#--------------------------------------------------------------------
#      Virtual Materials
#--------------------------------------------------------------------


#--------------------------------------------------------------------
def CreateVirtualGDL(voxelNumbers=(200,200,200),fiberContent=0.5,fiberRadius=10,fiberLength=100,
                             binderContent=0.3,anisotropy=1,randomSeed=1):
    
    #Algorithm :add fibers until a given fiber content is achieved. 
    #   Add binder until a given porosity is achieved    
    
    
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
        
        mesh = BasicShapes.CreateCylinder(center,axis,fiberRadius,fiberLength)
        objImage = Voxelization.Voxelize(mesh,gridX,gridY,gridZ,raydirection='z')
        image = np.logical_or(image,objImage)
        
        fiberTotalVolume = float(np.count_nonzero(image))/np.size(image)        
        print(fiberTotalVolume)
        
    #Add binder
    
    print('Adding binder until binderContent is reached')
    
    def GetBinder(image,binderThickness):
        myItkImage = sitk.GetImageFromArray(image.astype(np.uint8))
        myItkImage = sitk.BinaryDilate(myItkImage, int(binderThickness), sitk.sitkBall, 0.0, 1.0,  False)   
        myItkImage = sitk.BinaryErode(myItkImage, int(binderThickness), sitk.sitkBall, 0.0, 1.0,  False)   
        binder=sitk.GetArrayFromImage(myItkImage).astype(np.bool)     
        binder = np.logical_and(binder,np.logical_not(image))
        return binder
    
    #recherche dichotomique de binderThickness correspondant au binderContent demande
    binderTotalVolume = 0.0
    binderThickness=1    
    
    while binderTotalVolume<binderContent :
        print(binderTotalVolume)
        
        binder = GetBinder(image,binderThickness)
        binderTotalVolume = float(np.count_nonzero(binder))/np.size(image)
        
        binderThickness += 1
        
       
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
def CreateVirtualInterfaceGDLMPL(penetrationLength=15):

    imageSize=(400,400,300)
    
    boundary=50
    
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
        objImage=Voxelization.Voxelize(mesh,gridX,gridY,gridZ)
        image=np.logical_or(image,objImage)
      
      
    #spherical capings of cylinders      
    for iVertice in range(nVertice):
        center = vertices[iVertice]
        
        mesh = BasicShapes.CreateBall(tuple(center),fiberRadius)
        objImage=Voxelization.Voxelize(mesh,gridX,gridY,gridZ)
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
                  
        mesh = BasicShapes.CreateCylinder(center,holeAxis,holeRadius,holeHeight)
        objImage=Voxelization.Voxelize(mesh,gridX,gridY,gridZ)
        gdlImage=np.logical_and(gdlImage,np.logical_not(objImage))
        
    return gdlImage




 
    