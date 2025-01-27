# -*- coding: utf-8 -*-

import numpy as np

import VirtualMaterials.VirtualImages.VirtualMaterialsGeneration  as virtMatGen
from VirtualMaterials.VirtualImages  import BasicShapes
from VirtualMaterials.VirtualImages  import Voxelization
from VirtualMaterials.Utilities  import Visualization
import VirtualMaterials.Utilities.Utilities as Utilities    
   
#--------------------------------------------------------------------
#      Tests
#--------------------------------------------------------------------
     
#--------------------------------------------------------------------    
def TestVoxelize(raydirection):
    #mesh=BasicShapes.CreateCylinder((0,0,0),(1,0,0),15,100)
    mesh=BasicShapes.CreateSpline()
    voxelNumbers=(100,100,100)
    bounds=mesh.GetBounds()
    gridX=np.linspace(bounds[0],bounds[1],voxelNumbers[0])
    gridY=np.linspace(bounds[3],bounds[2],voxelNumbers[1])
    gridZ=np.linspace(bounds[5],bounds[4],voxelNumbers[2])
    image=Voxelization.Voxelize(mesh,gridX,gridY,gridZ,raydirection=raydirection)
    Utilities.WriteTiff(100*(image.astype(np.uint8)),'TestVoxelizePython.tif')     
    return image    
    
#--------------------------------------------------------------------
def TestVirtualVoronoi():    
    voxelNumbers=(200,200,200)
    imageBounds = (0.0,1.0,0.0,1.0,0.0,1.0)
    nPoint=200
    fiberRadius=0.05
    anisotropy=3
    randomSeed=0
    image=virtMatGen.CreateVirtualVoronoiFoam(voxelNumbers=voxelNumbers,imageBounds=imageBounds,
                                              nPoint=nPoint,fiberRadius=fiberRadius,
                                              anisotropy=anisotropy,randomSeed=randomSeed)
    Utilities.WriteTiff(255*(image.astype(np.uint8)),'TestVoronoi.tif')
    return image 
    
#--------------------------------------------------------------------
def TestVirtualGDL():
    voxelNumbers=(500,500,200)
    fiberContent=0.1
    fiberRadius=9
    fiberLength=500
    binderContent=0.1
    anisotropy=5
    randomSeed=0
    image = virtMatGen.CreateVirtualGDL(voxelNumbers=voxelNumbers,
                         fiberContent=fiberContent,fiberRadius=fiberRadius,fiberLength=fiberLength,
                         binderContent=binderContent,
                         anisotropy=anisotropy,randomSeed=randomSeed) 
    Utilities.WriteTiff(100*(image.astype(np.uint8)),'TestBigGDL.tif')
    return image 
    
#--------------------------------------------------------------------    
def TestVirtualCCL():
    
    voxelNumbers=(200,200,200)
    carbonGrainRadius=4
    nCarbonGrain=5000
    voidRadius=50
    nVoid=200
    nafionThickness=2
    nafionCoveragePercentage=30
    randomSeed=1    
    image = virtMatGen.CreateVirtualCatalystLayer(voxelNumbers=voxelNumbers,
            carbonGrainRadius=carbonGrainRadius,nCarbonGrain=nCarbonGrain,
            voidRadius=voidRadius,nVoid=nVoid,
            nafionThickness=nafionThickness,nafionCoveragePercentage=nafionCoveragePercentage,
            randomSeed=randomSeed)
            
    Utilities.WriteTiff(100*(image.astype(np.uint8)),'TestCCL.tif')
    return image 
    
#--------------------------------------------------------------------        
def TestVisualization():
    image = virtMatGen.CreateVirtualGDL(voxelNumbers=(400,400,100),fiberContent=0.2,fiberRadius=9,
                             fiberLength=400,binderContent=0.05,anisotropy=5,randomSeed=0)

    Visualization.VisualizeVolumeRendering(image)
#    Visualization.VisualizeCutPlanes(image)
#    Visualization.VisualizeIsoSurface(image)
    
    
#--------------------------------------------------------------------
def TestInterfaceGDLMPL(): 
    interface=virtMatGen.CreateVirtualInterfaceGDLMPL(penetrationLength=30)   
    Utilities.WriteTiff(50*(interface.astype(np.uint8)),'TestInterfaceGDLMPL_penetration30.tif')    
    return interface 
