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
def ChordLength(image,label,direction=(1,0,0)): 
    """ChordLength : 
    :param image : numpy image
    :param label : code of the phase you want to study
    :param direction : chord direction (defaultValue=(1,0,0))
    :return: chordLength
    
    :Example:    
    import VirtualMaterials as vmat
    import numpy as np
    import matplotlib.pyplot as plt
    label = 0
    chordLength = vmat.ImageAnalysis.QuantifyGeometry.ChordLength(image,label,direction=(1,0,0))
    meanChordLength = np.mean(chordLength)
    # plot chord length distribution
    hist, bin_edges = np.histogram(chordLength,bins=10,density=True)
    bin_center=(bin_edges[0:-1]+bin_edges[1:])/2.0
    plt.plot(bin_center,hist)
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
    
    # Remove pathological chords, i.e. the ones were there is an uncertainty 
    #regarding their length
    # Here we use the following criterium: the chord must have the same length 
    #as a reference chord.
    # A better criteria would be : don't take into account the chords that 
    # intersect the frontiers of the domain    
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
    
    return chordLength
    
    
    
    
#----------------------------------------------------------------------------------------------    
def PoreSizeDistribution_Continuous(image,label=0,nPoint=10): 
    """PoreSizeDistribution_Continuous : NOT IMPLEMENTED 
    PoreSizeDistribution_Continuous of a label in the image
    :param image : numpy image 
    :param label : code of the phase you want to study (default=0)
    :param nPoint : number of sampling points to compute the size distribution (default=10)
    :return: radiusList,cPSD
    
    :Example:    
    import VirtualMaterials as vmat
    poreCode = 0
    radiusList,cPSD = vmat.ImageAnalysis.QuantifyGeometry.PoreSizeDistribution_Continuous(image,label=poreCode,nPoint=10)
    """
    
    imageBool = np.copy(image==label)
    # Compute distance map
    distanceMap = vmat.ImageAnalysis.Morphology.DistanceMap(imageBool) 
    
    #Compute radius covering map :
    # for size from 1 voxel to max(DistanceMap), dilate distanceMap>size with
    # ball of radius size
    radiusCovMap=np.zeros(imageBool.shape,dtype=np.uint8)
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
    
    
#--------------------------------------------------------------------------
def PoreSizeDistribution_Watershed(image,label=0,hMaxima=4): 
    """PoreSizeDistribution_Watershed :
    Pore and constriction size distribution using Watershed segmentation.
    Returns the size distribution of pore and constriction, using the following geometric definition :    
    poreEquivalentRadius : radius of the sphere of same volume (in voxels)
    constrictionInscribedRadius : max of distance map on constriction (in voxels)

    :param image : numpy image
    :param hMaxima : seed parameter of the watershed algorithm (default=4). More information in VirtualMaterials.Simulations.PoreNetworkExtraction
    :return: poreEquivalentRadius,constrictionInscribedRadius 
        
    :Example:    
    import VirtualMaterials as vmat
    import numpy as np
    import matplotlib.pyplot as plt
    poreCode = 0
    poreEquivalentRadius,constrictionInscribedRadius = vmat.ImageAnalysis.QuantifyGeometry.PoreAndLinkSizeDistribution_Watershed(
                                        simuImage,label=poreCode,hMaxima=4)
    #plot pore size distribution                                    
    pore_hist, pore_bin_edges = np.histogram(poreEquivalentRadius,bins=10,density=True)
    pore_bin_center=(pore_bin_edges[0:-1]+pore_bin_edges[1:])/2.0
    plt.plot(pore_bin_edges,pore_hist)                                    
    """
    
        
    ExtractionResult = vmat.Simulation.PoreNetworkExtraction.ExtractNetwork(
                image=image,phases={'someName':label},seedMethod='hMaxima',seedParam=hMaxima)
    
    constrictionInscribedRadius = ExtractionResult['internalLinkCapillaryRadius']
    
    poreVolume = ExtractionResult['poreVolumes'] 
    poreEquivalentRadius = np.power(3*poreVolume/float(math.pi*4),1/3.0)
    
    #poreInscribedRadius = ExtractionResult['poreInscribedSphereRadius']
    
    return poreEquivalentRadius,constrictionInscribedRadius
    
    
    