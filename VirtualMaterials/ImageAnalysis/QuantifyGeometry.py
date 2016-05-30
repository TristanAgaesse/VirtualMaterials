# -*- coding: utf-8 -*-
import SimpleITK as sitk
import numpy as np
import VirtualMaterials as vmat
from scipy import ndimage
import math
#----------------------------------------------------------------------------------------------    
def VolumeFraction(image): 
    """VolumeFraction : gives the volume fraction of each label in an image
    :param image : numpy image
    :return: volumeFraction : volumeFraction[numLabel] = volume fraction of label numLabel in the image
    
    :Example:    
    import VirtualMaterials as vmat
    volumeFraction = vmat.ImageAnalysis.QuantifyGeometry.VolumeFraction(image)
    """

    phaseList= np.unique(image)
    maxPhase = max(phaseList)
    volumeFraction = np.zeros(maxPhase+1)

    imSize=float(image.size)
    for iPhase in phaseList:
        volumeFraction[iPhase] = np.count_nonzero(image==iPhase)/imSize

    return volumeFraction

#----------------------------------------------------------------------------------------------    
def Tortuosity(image,label): 
    """Tortuosity : NOT IMPLEMENTED tortuosity of a label in the image
    :param image : numpy image
    :param label : label
    :return: tortuosity
    
    :Example:    
    import VirtualMaterials as vmat
    label = 0
    tortuosity = vmat.ImageAnalysis.QuantifyGeometry.Tortuosity(image,label)
    """
    
    #ITK MinimalPathExtraction
    print('Tortuosity : NOT IMPLEMENTED')
    tortuosity=-1
    return tortuosity


#----------------------------------------------------------------------------------------------    
def Constrictivity(image,label): 
    """Constrictivity : NOT IMPLEMENTED Constrictivity of a label in the image
    :param image : numpy image
    :param label : label
    :return: constrictivity
    
    :Example:    
    import VirtualMaterials as vmat
    label = 0
    constrictivity = vmat.ImageAnalysis.QuantifyGeometry.Constrictivity(image,label)
    """
    
    # Mean size according to full morphology
    print('Constrictivity : NOT IMPLEMENTED')
    constrictivity=-1
    return constrictivity
    
    
    
#----------------------------------------------------------------------------------------------    
def ChordLength(image,label,direction=(1,0,0),mode='meanLength'): 
    """ChordLength : 
    :param image : numpy image
    :param label : label
    :param direction : chord direction (defaultValue=(1,0,0))
    :param mode : 'meanLength' or 'allLength'
    :return: chordLength
    
    :Example:    
    import VirtualMaterials as vmat
    label = 0
    chordLength = vmat.ImageAnalysis.QuantifyGeometry.ChordLength(image,label,direction=(1,0,0),mode='meanLength')
    """
    
    # Mask image
    
    maskImage = np.zeros(image.shape,dtype=np.bool)
    
    direction = np.asarray(direction)
    direction = direction/np.linalg.norm(direction)
    prototypeLength = min(image.shape/direction)
    start = np.array([0,0,0])    
    end = np.floor(start+prototypeLength*direction)
    shift=np.array([min(start[0],end[0]),min(start[1],end[1]),min(start[2],end[2])])            
    start = start-shift
    end = end-shift 
    assert(np.all(np.greater_equal(start,np.array([0,0,0]) )))
    assert(np.all(np.greater_equal(end,np.array([0,0,0]) )))
    assert(np.all(np.less_equal(start,image.shape )))
    assert(np.all(np.less_equal(end,image.shape )))
    chordPrototype = vmat.VirtualImages.BasicShapes.CreateLine(start, end)
    X,Y,Z = chordPrototype[:,0]-1,chordPrototype[:,1]-1,chordPrototype[:,2]-1
    
    #Put chordPrototype in mask image, along with all its copies    
    shiftXList = [3*i for i in range(image.shape[0]/3+1)]
    shiftYList = [3*i for i in range(image.shape[1]/3+1)]
    shiftZlist = [3*i for i in range(image.shape[2]/3+1)]
    perpDir=np.argmax(direction)
    if perpDir==0:
        shiftXList=[0]
    elif perpDir==1:
        shiftYList=[0]
    elif perpDir==2:
        shiftZlist=[0]
    
    for shiftX in shiftXList:
        for shiftY in shiftYList:
            for shiftZ in shiftZlist:
                Xshifted=np.mod(np.asarray(X+shiftX),image.shape[0]*np.ones(X.size,dtype=np.int))
                Yshifted=np.mod(np.asarray(Y+shiftY),image.shape[1]*np.ones(Y.size,dtype=np.int))
                Zshifted=np.mod(np.asarray(Z+shiftZ),image.shape[2]*np.ones(Z.size,dtype=np.int))
                maskImage[Xshifted,Yshifted,Zshifted]=True
    
    # length of chords
    labeledmask = vmat.ImageAnalysis.FeatureExtraction.ConnectedComponentsLabeling(maskImage)
    maskVolume=ndimage.measurements.labeled_comprehension(
                  labeledmask, labeledmask,range(1,labeledmask.max()+1),
                  np.size,np.int,0)
    
    prototypeVolume = float(chordPrototype.shape[0])
    
    foo=np.abs((maskVolume-prototypeVolume)/prototypeVolume)
    properMask= np.nonzero( np.less(foo,0.05))[0]+1
    maskImage = np.zeros(image.shape,dtype=np.bool)
    
    labeledmask_unique_sorted, labeledmask_idx = np.unique(labeledmask, return_inverse=True)
    labeledmask_in_properMask_bool = np.in1d(labeledmask_unique_sorted, properMask, assume_unique=True)    
    maskImage=labeledmask_in_properMask_bool[labeledmask_idx]
    maskImage=np.reshape( maskImage,image.shape)
    
    lengthToVolumeRatio = prototypeLength/float(prototypeVolume)
    
    # Multiply image and mask image
    chordImage = np.logical_and(image==label,maskImage)
    
    # label chords
    labeledChords = vmat.ImageAnalysis.FeatureExtraction.ConnectedComponentsLabeling(chordImage)
    
    # length of chords
    chordVolume=ndimage.measurements.labeled_comprehension(
                  labeledChords, labeledChords,range(1,labeledChords.max()+1),
                  np.size,np.int,0)
    
    chordLength = chordVolume*lengthToVolumeRatio      
    
    if mode=='allLength':
        length = chordLength
    else:
        length = np.mean(chordLength)
        
    
    return length
    
    
    
    
#----------------------------------------------------------------------------------------------    
def PoreSizeDistribution_Continuous(image,nPoint=10): 
    """PoreSizeDistribution_Continuous : NOT IMPLEMENTED 
    PoreSizeDistribution_Continuous of a label in the image
    :param image : numpy image (boolean)
    :param nPoint : label (default=10)
    :return: radiusList,cPSD
    
    :Example:    
    import VirtualMaterials as vmat
    label = 0
    radiusList,cPSD = vmat.ImageAnalysis.QuantifyGeometry.PoreSizeDistribution_Continuous(image,label)
    """
    
    # Compute distance map
    distanceMap = vmat.ImageAnalysis.Morphology.DistanceMap(image) 
    
    #Compute radius covering map :
    # for size from 1 voxel to max(DistanceMap), dilate distanceMap>size with
    # ball of radius size
    radiusCovMap=np.zeros(image.shape,dtype=np.uint8)
    maxRadius = int(distanceMap.max())
    nPoint = min((nPoint,maxRadius))
    radiusList = np.linspace(1,maxRadius,nPoint).astype(np.uint8)
    for i in range(nPoint):
        radius = radiusList[-i]
        foo=vmat.ImageAnalysis.Morphology.Dilation(distanceMap>=float(radius),radius)
        radiusCovMap[np.logical_and(foo,radiusCovMap==0)] = radius
    
    # Post treatement : coonvert the radius covering map to a PSD
    volumeFraction = vmat.ImageAnalysis.QuantifyGeometry.VolumeFraction(radiusCovMap)
    
    sphereVolume = [4/3.0*math.pi*R**3 for R in radiusList.tolist()]
    cPSD = [volumeFraction[radiusList[i]]/sphereVolume[i] 
                                    for i in range(1,maxRadius)]
    cPSD = np.asarray(cPSD)
    
    
    
    return radiusList,cPSD
    
    
    
    
    