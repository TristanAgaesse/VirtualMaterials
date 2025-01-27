# -*- coding: utf-8 -*-

import numpy as np
from scipy import ndimage
import SimpleITK as sitk
import time
import VirtualMaterials as vmat

def CapillaryPressureCurve(image,porePhaseCode=0,inletFace=0,
                           voxelLength=1,nPoints=10,surfaceTension=72e-3):
    """Computes the capillary pressure curve of the pore phase using a 
    FullMorphology simulation of water injection.
    :param image: numpy ndarray
    :param porePhaseCode : code of the void voxels (default:0)
    :param inletFace: 0 for Xmin, 1 for Xmax, 2 for Ymin, 3 for Ymax, 4 for Zmin, 5 for Zmax (default:0)
    :param nPoints: number of points on the curve (default:10)
    :param surfaceTension: surface tension (default:72e-3)
    :return: output (dictionary)
    output['Saturation list']: saturation for Pc(S) curve
    output['Capillary pressure list (in Pa)']: capillary pressure for Pc(S) curve
    output['Ball radius list (in voxel)'] 
    output['Contact angle']: 180 (fixed)
    output['Image with water']: image of water distributions
    
    :Example:
    import VirtualMaterials as vmat 
    output = vmat.Simulation.FullMorphology.CapillaryPressureCurve(
               image,porePhaseCode=0,inletFace=0,voxelLength=1,nPoints=10)
    output['Saturation list']
    output['Capillary pressure list (in Pa)']
    output['Ball radius list (in voxel)']
    
    """
    
    simuImage = 255*(np.logical_not(image==porePhaseCode).astype(np.uint8)) 
    
    distanceMap = vmat.ImageAnalysis.Morphology.DistanceMap(simuImage==0)
    maxRadius = distanceMap.max()
    
    minRadius = 2   
    #I take a minimum radius of minRadius voxels. Indeed, for very small radii   
    #the result depends too much on the small details of the microstructure     
    
    gamma=surfaceTension 
    
    # Define the pressure points where to compute the water distributions
    
    radiusList = np.unique(np.linspace(minRadius,maxRadius,nPoints).astype(np.int)) 
    radiusList = radiusList.tolist()
    pressureList = [2*gamma/float(voxelLength*radius) for radius in radiusList]
    
#    minPressure = 2*gamma/float(voxelLength*maxRadius)
#    maxPressure = 2*gamma/float(voxelLength*minRadius)
#    pressureList = np.linspace(minPressure,maxPressure,nPoints) 
#    radiusList = [int(2*gamma/float(voxelLength*pressure)) for pressure in pressureList]
#    radiusList=np.unique(np.asarray(radiusList).astype(np.int))
#    pressureList = [2*gamma/float(voxelLength*radius) for radius in radiusList] #Update pressure list after removing non unique radii
#    
    nPoints = len(radiusList)
    pressureCode = [100+i for i in range(nPoints)]    
    
    # Simulation
    imageWithWater = FullMorphology(simuImage,inletFace=inletFace,
                                    voxelLength=voxelLength,pressureList=pressureList,
                                    pressureCode=pressureCode,gamma=gamma,
                                    distanceMap=distanceMap)
    
    # Convert the water distributions to Pc(S) curve
    waterVolumeFraction = vmat.ImageAnalysis.QuantifyGeometry.VolumeFraction(imageWithWater)    
    poreVolumeFraction = vmat.ImageAnalysis.QuantifyGeometry.VolumeFraction(image)[porePhaseCode] 
    
    cumulativeSaturation = 0
    saturationList = []
    for i in range(nPoints):
        cumulativeSaturation += waterVolumeFraction[pressureCode[nPoints-1-i]]/float(poreVolumeFraction)
        saturationList.append(cumulativeSaturation)
    
    output = dict()
    output['Contact angle'] = 180
    output['Image with water'] = imageWithWater
    output['Saturation list'] = saturationList
    output['Capillary pressure list (in Pa)'] = pressureList[::-1]
    output['Ball radius list (in voxel)'] = radiusList[::-1]
    
    return output
    
    
    
#----------------------------------------------------------------------------------------------
def FullMorphology(inputImage,inletFace=0,voxelLength=1,pressureList=[10],pressureCode=[110],gamma=72e-3,distanceMap=None):
    """FullMorphology simulation of water injection in a porous medium.
    :param inputImage: numpy ndarray 
    :param inletFace: 0 for Xmin, 1 for Xmax, 2 for Ymin, 3 for Ymax, 4 for Zmin, 5 for Zmax
    :param pressureList: list of one or several pressures
    :param pressureCode: list of codes to label water at the pressures in pressureList
    :param gamma: surface tension
    :param distanceMap: (facultative, default=None): precomputed distanceMap
    :return: image: inputImage + water labelled as pressureCode
    
    :Example:
    import VirtualMaterials as vmat 
    imageWithWater = vmat.Simulation.FullMorphology.FullMorphology(
                    inputImage,inletFace=1,voxelLength=1,
                    pressureList=[10],pressureCode=[110],gamma=72e-3)
    """
    
    
        
    beginTime=time.time()
            
    myImg = inputImage.copy()

    inletVoxels = np.zeros(myImg.shape,dtype=bool)
    
    
    if inletFace==0:
        inletVoxels[0,:,:] = True
    elif inletFace==1:
        inletVoxels[-1,:,:] = True
    elif inletFace==2:
        inletVoxels[:,0,:] = True
    elif inletFace==3:
        inletVoxels[:,-1,:] = True
    elif inletFace==4:
        inletVoxels[:,:,0] = True
    elif inletFace==5:
        inletVoxels[:,:,-1] = True        
        
    if distanceMap is None:
        distanceMap = vmat.ImageAnalysis.Morphology.DistanceMap(myImg==0)
    else:
        assert(distanceMap.shape == myImg.shape)
        distanceMap=distanceMap


    pressureCode=np.asarray(pressureCode).astype(np.uint8)
    pressureList = np.asarray(pressureList)
    ascendingOrder=np.argsort(pressureList)
    
    for i in range(len(pressureList)-1,-1,-1):
        print('begin '+str(i))
        pressure = pressureList[ascendingOrder[i]]
        water = __FullMorphologyHydrophobicStep__(distanceMap,2*gamma/(pressure*voxelLength),inletVoxels)
        myImg[water] = pressureCode[ascendingOrder[i]]
        del water  
    
      
      
    

    endTime=time.time()
    print("Time spent : {} s".format(endTime-beginTime))

    return myImg.astype(np.uint8)

#----------------------------------------------------------------------------------------------

def __FullMorphologyHydrophobicStep__(distanceMap,capillaryLength,inletVoxels):
    
    #Find the centers of the water balls
    indicesCenters=(distanceMap>float(capillaryLength+0.5)).reshape(distanceMap.shape)
    invadedVoxels=np.zeros(distanceMap.shape,dtype=bool)
    invadedVoxels[indicesCenters]=True
    del indicesCenters
    
              
    #Keep only water connected with the inlet          
    structuringElement = np.ones((3,3,3))
    invadedVoxels=ndimage.measurements.label(invadedVoxels,structuringElement)[0]
    
    labelsToKeep=np.unique(invadedVoxels[inletVoxels])    
    labelsToKeep=labelsToKeep[labelsToKeep>0]
    
    foo=np.zeros(invadedVoxels.shape,dtype=bool)
    
    for iLabel in labelsToKeep :
        foo[invadedVoxels==iLabel] = True
    
    del labelsToKeep
    
    invadedVoxels=invadedVoxels.astype(np.bool)
    invadedVoxels[np.logical_not(foo)]=False
    invadedVoxels[foo]=True
    
    #Dilate with a ball of size capillaryLength
    
    #Scipy implementation of dilation
    #ball=morphology.ball(capillaryLength)
    #invadedVoxels=ndimage.morphology.binary_dilation(invadedVoxels, structure=ball)     
    
    #SimpleITK implementation of dilation
    itkInvadedVoxels = sitk.GetImageFromArray(invadedVoxels.astype(np.uint8))
    itkInvadedVoxels = sitk.BinaryDilate(itkInvadedVoxels, int(capillaryLength), sitk.sitkBall, 0.0, 1.0,  False)   
    invadedVoxels=sitk.GetArrayFromImage(itkInvadedVoxels)  
    invadedVoxels=invadedVoxels.astype(np.bool) 
    
    
    return invadedVoxels
    
    