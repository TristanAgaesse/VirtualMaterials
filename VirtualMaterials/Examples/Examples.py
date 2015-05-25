# -*- coding: utf-8 -*-
"""
Created on Mon May 25 18:28:22 2015

@author: greentoto
"""

import numpy as np

import VirtualMaterials.VirtualImages.VirtualMaterialsGeneration  as virtMatGen
from VirtualMaterials.VirtualImages  import BasicShapes
from VirtualMaterials.VirtualImages  import Voxelization
from VirtualMaterials.VirtualImages  import Visualization
import VirtualMaterials.Utilities.Utilities as Utilities    
   
#--------------------------------------------------------------------
#      Tests
#--------------------------------------------------------------------
     
#--------------------------------------------------------------------    
def TestVoxelize():
    mesh=BasicShapes.CreateCylinder((0,0,0),(1,0,0),15,100)
    voxelNumbers=(100,100,100)
    bounds=mesh.GetBounds()
    gridX=np.linspace(bounds[0],bounds[1],voxelNumbers[0])
    gridY=np.linspace(bounds[3],bounds[2],voxelNumbers[1])
    gridZ=np.linspace(bounds[5],bounds[4],voxelNumbers[2])
    image=Voxelization.Voxelize(mesh,gridX,gridY,gridZ)
    Utilities.SaveImageTiff(100*(image.astype(np.uint8)),'TestVoxelizePython.tif')     
    
#--------------------------------------------------------------------
def TestVirtualVoronoi():    
    voxelNumbers=(200,200,200)
    imageBounds = (0.0,1.0,0.0,1.0,0.0,1.0)
    nPoint=200
    fiberRadius=0.05
    anisotropy=3
    randomSeed=0
    image=virtMatGen.CreateVirtualVoronoiFoam(voxelNumbers,imageBounds,nPoint,fiberRadius,anisotropy,randomSeed)
    Utilities.SaveImageTiff(255*(image.astype(np.uint8)),'TestVoronoi.tif')

#--------------------------------------------------------------------
def TestVirtualGDL():
    image = virtMatGen.CreateVirtualGDL(voxelNumbers=(500,500,200),fiberContent=0.1,fiberRadius=9,
                             fiberLength=500,binderContent=0.1,anisotropy=5,randomSeed=0) 
    Utilities.SaveImageTiff(100*(image.astype(np.uint8)),'TestBigGDL.tif')
    
   
#--------------------------------------------------------------------    
def TestVirtualCCL():
    image = virtMatGen.CreateVirtualCatalystLayer((200,200,200),4,5000,50,200,2,30)
    Utilities.SaveImageTiff(100*(image.astype(np.uint8)),'TestCCL.tif')
    
    

    
    
#--------------------------------------------------------------------        
def TestVisualization():
    image = virtMatGen.CreateVirtualGDL(voxelNumbers=(400,400,100),fiberContent=0.2,fiberRadius=9,
                             fiberLength=400,binderContent=0.05,anisotropy=5,randomSeed=0)

    Visualization.VisualizeVolumeRendering(image)
    Visualization.VisualizeCutPlanes(image)
    Visualization.VisualizeIsoSurface(image)
    
    
#--------------------------------------------------------------------
def TestInterfaceGDLMPL(): 

    interface=virtMatGen.CreateVirtualInterfaceGDLMPL(penetrationLength=30)   
    
    Utilities.SaveImageTiff(50*(interface.astype(np.uint8)),'TestInterfaceGDLMPL_penetration30.tif')    
    
