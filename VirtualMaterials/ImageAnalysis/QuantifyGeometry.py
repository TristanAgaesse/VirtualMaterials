# -*- coding: utf-8 -*-
import SimpleITK as sitk
import numpy as np
import VirtualMaterials as vmat
from scipy import ndimage

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
    volumeFraction = np.zeros((1,maxPhase+1))

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
    tortuosity = -1
    """
    
    #ITK MinimalPathExtraction
    tortuosity=-1
    return tortuosity

#----------------------------------------------------------------------------------------------    
def ChordLength(image,label,direction=(1,0,0),mode='mean'): 
    """ChordLength : NOT IMPLEMENTED tortuosity of a label in the image
    :param image : numpy image
    :param label : label
    :param direction : chord direction (defaultValue=(1,0,0))
    :param mode : 'mean' or 'allLength'
    :return: chordLength
    
    :Example:    
    import VirtualMaterials as vmat
    label = 0
    ChordLength = vmat.ImageAnalysis.QuantifyGeometry.ChordLength(image,label,direction=(1,0,0),mode='mean')
    """
    
    # Mask image
    
    maskImage = np.zeros(image.shape,dtype=np.bool)

    direction = np.asarray(direction)
    direction = direction/np.linalg.norm(direction)
    prototypeLength = np.dot(direction,image.shape)
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
    
    #Put chordPrototype in mask image, along with all its copies    
    
    prototypeVolume = np.count_nonzero(chordPrototype)
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
    