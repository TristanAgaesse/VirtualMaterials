# -*- coding: utf-8 -*-


import numpy as np
import hdf5storage
from scipy import ndimage
from skimage import morphology
from skimage import feature
import SimpleITK as sitk
import VirtualMaterials.ImageAnalysis as imageAnalysis
from VirtualMaterials.Utilities import Utilities as utilities    
from collections import defaultdict
import time
import re 
#from numba import jit
import os
#import mahotas
#from skimage import io as skimio


#----------------------------------------------------------------------------------------------
def ExtractNetwork(image=np.ones((3,3,3)),phases={'void':False},seedMethod='hMaxima',seedParam=4):
    """Extract a pore network from an image.
    :param image: numpy ndarray or imageFileName
    :param phases: dict, {'PhaseName':PhaseCode in image,...}
    :param seedMethod : "hMaxima" or "localMax"
    :param seedParam: value of parameter used for seed method
    :return: dict containing the extraction results
    
    :Example:
    
    import VirtualMaterials.Simulations.PoreNetworkExtraction as pnex
    ExtractionResult = pnex.ExtractNetwork(image=np.ones((3,3,3)),phases={'void':False},seedMethod='hMaxima',seedParam=4)
    pnex.SaveResults('foo_extractionResults.mat',ExtractionResult)
    """
    
    beginTime=time.time()
    
    structuringElement = np.ones((3,3,3))
    distanceType='ITKDanielson'
    
    image,phases,seedMethod,seedParam = __ReadInputs__(image,phases,seedMethod,seedParam)
    
    print('PoresWatershedSegmentation')
    
    pores,watershedLines,distanceMap,porePhase = PoresSegmentation(image,
                                          structuringElement=structuringElement,
                                          seedMethod=seedMethod,seedParam=seedParam,
                                          distanceType=distanceType,phases=phases)
    
    print('FindLinks')
    
    links,interfaceToPore=FindLinks(image,pores,watershedLines,structuringElement)
    
    print('AnalyseElementsGeometry')
    
    ExtractionResult = AnalyseElementsGeometry(image,pores,links,distanceMap,phases=phases)
    
    
    ExtractionResult.update({'interfaceToPore':interfaceToPore,'imagePores':pores,
                             'imageLiens':links,'myImage':image,
                             'porePhase':porePhase,
                             'seedMethod':seedMethod,'seedParam':seedParam})
    
    endTime=time.time()
    print("Time spent : {} s".format(endTime-beginTime))
    
    return ExtractionResult
    
    
    
#----------------------------------------------------------------------------------------------
def __ReadInputs__(image,phases,seedMethod,seedParam):

    if isinstance(image,basestring):
        imageFileName=image
        assert(os.path.isfile(imageFileName))
        image=utilities.ReadTiff(imageFileName)    
    else:
        assert(isinstance(image,np.ndarray))
    
    assert(image.min()>=0)
    memoryType = utilities.BestMemoryType(image.max())    
    image = image.astype(memoryType)
    
    assert(isinstance(phases,dict))
    assert(isinstance(int(seedParam),int))
    
    assert(seedMethod=="hMaxima" or seedMethod=="localMax"  )  
    
    
    return image,phases,seedMethod,seedParam
    

#----------------------------------------------------------------------------------------------
def SaveResults(outputFile,ExtractionResult):
    
    print('Saving results to disk')
    assert( isinstance(outputFile,basestring) )   
    assert(isinstance(ExtractionResult,dict) )
    
    hdf5storage.savemat(outputFile,mdict=ExtractionResult)
    
    
#----------------------------------------------------------------------------------------------
def PoresSegmentation(myImg,phases={'void':False},structuringElement=np.ones((3,3,3)),seedMethod='hMaxima',seedParam=4,distanceType='ITKDanielson'):
    #Phases=dict, phases['myphase']=codeForMyPhaseInImage
    
    pores=np.zeros(myImg.shape,dtype=np.uint8)
    watershedLines=np.zeros(myImg.shape,dtype=np.bool)
    
    if distanceType=='euclidean' or distanceType=='ITKDanielson':
        memoryType=np.float16    
    elif distanceType=='chamfer':
        memoryType=np.int8
    distanceMap=np.zeros(myImg.shape,dtype=memoryType)    
        
    labelShift=[0]   
    for phaseName in phases.keys():
        phaseCode=phases[phaseName]
        phaseImage= (myImg==np.uint8(phaseCode)).astype(np.bool) 
        
        poresPhase,watershedLinesPhase,distanceMapPhase = PoresWatershedSegmentationOnePhase(
                            phaseImage,structuringElement=structuringElement,
                            seedMethod=seedMethod,seedParam=seedParam,distanceType=distanceType)
        
        phaseImage=phaseImage.astype(np.bool)
        memoryType=utilities.BestMemoryType(poresPhase.max()+pores.max())
        pores = pores.astype(memoryType)
        pores[phaseImage]=(poresPhase[phaseImage]).astype(memoryType)+labelShift[-1]*(
                           poresPhase[phaseImage]>0).astype(memoryType)
        
        watershedLines[phaseImage] = watershedLinesPhase[phaseImage]
        distanceMap[phaseImage] = distanceMapPhase[phaseImage]
        del phaseImage, poresPhase,watershedLinesPhase,distanceMapPhase
        
        labelShift.append(pores.max())
        
        
    #phaseBoundaries = SobelEdgeDetection(myImg)     
    phaseBoundaries = imageAnalysis.FeatureExtraction.SobelEdgeDetection(myImg)   
    watershedLines=np.logical_or(watershedLines,phaseBoundaries)
    
    
    porePhase=np.zeros(pores.max(),dtype=np.uint8)    
    for i in range(len(labelShift)-1):
        porePhase[np.arange(labelShift[i],labelShift[i+1])]=int(phases[phases.keys()[i]])
    
    
    return pores,watershedLines,distanceMap,porePhase
    
    
    
    
    
    
#----------------------------------------------------------------------------------------------
def PoresWatershedSegmentationOnePhase(phaseImage,structuringElement=np.ones((3,3,3)),
                                       distanceType='ITKDanielson',watershedAlgo='ITK',
                                       seedMethod='hMaxima',seedParam=4):
    
    
    #Calcul de la carte de distance distanceMap
    
    if distanceType=='euclidean':
        memoryType=np.float16
        distanceMap = ndimage.distance_transform_edt(phaseImage
                                                    ).astype(memoryType)
    elif distanceType=='chamfer':
        memoryType=np.int8
        distanceMap = ndimage.distance_transform_cdt(phaseImage,
                                        metric='chessboard').astype(memoryType)
    elif distanceType=='ITKDanielson':
        memoryType=np.float16
        itkimage = sitk.GetImageFromArray(np.logical_not(phaseImage).astype(np.uint8))
        itkdistanceMap = sitk.DanielssonDistanceMap( itkimage )
        del itkimage
        distanceMap=sitk.GetArrayFromImage(itkdistanceMap).astype(memoryType) 
        del itkdistanceMap
    
    #Choix des marqueurs pour la segmentation (centres des pores) : H-maxima :
    #maxima de la carte de distance dont les pics sont étêtés d'une hauteur h. Utilise  
    #une recontruction morphologique pour construire la carte de distance étêtée.
    
    hMaximaLib='skimage'
    
    if seedMethod=='localMax':
        seedParam=int(seedParam)
        local_maxi= feature.peak_local_max(distanceMap.astype(np.float), 
                                           min_distance=seedParam, indices=False)
    
    elif seedMethod=='hMaxima' and hMaximaLib=='skimage' and seedParam>0:
        hContrast=np.asarray(seedParam).astype(memoryType)
        reconstructed=morphology.reconstruction(distanceMap-hContrast, distanceMap
                                            ).astype(memoryType)
        
        local_maxi=(distanceMap-reconstructed).astype(np.bool)
        del reconstructed
        
    elif seedMethod=='hMaxima' and hMaximaLib=='ITK' and seedParam>0:    
        hContrast=np.asarray(seedParam).astype(memoryType)
        itkSeedimage = sitk.GetImageFromArray((distanceMap-hContrast).astype(np.float32))
        itkMarkerImage = sitk.GetImageFromArray(distanceMap.astype(np.float32))
        itkReconstructed = sitk.ReconstructionByDilation(itkSeedimage, itkMarkerImage)
        del itkMarkerImage,itkSeedimage
        reconstructed = sitk.GetArrayFromImage(itkReconstructed).astype(memoryType)                                    
        del itkReconstructed

        local_maxi=((distanceMap-reconstructed)>hContrast/10).astype(np.bool)
        del reconstructed
        

        
    
    markers = ndimage.measurements.label(local_maxi , structure=structuringElement)[0]
    
    del local_maxi
    
    
    #Calcul des lignes de partage de niveau 
    
    if watershedAlgo=='Mahotas':
        _,watershedLines = mahotas.cwatershed(
                                (distanceMap.max()-distanceMap).astype(np.int8), 
                                markers, Bc=structuringElement , return_lines=True)
    elif watershedAlgo=='ITK':
        itkMarkers= sitk.GetImageFromArray(markers)
        itkDistance = sitk.GetImageFromArray(distanceMap.astype(np.float))
        itkDistance = sitk.InvertIntensity(itkDistance)
        
        wsITK = sitk.MorphologicalWatershedFromMarkers(itkDistance,itkMarkers,
                                                   markWatershedLine=True,
                                                   fullyConnected=True)
        del itkMarkers, itkDistance                                           
#        mask = itk.MaskImageFilter.IUC2IUC2IUC2.New(ws, fill)
#        overlay = itk.LabelOverlayImageFilter.IUC2IUC2IRGBUC2.New(reader,
#                                                                  mask,
#                                                                  UseBackground=True)
        ws=sitk.GetArrayFromImage(wsITK).astype(np.uint8) 
        del wsITK
        watershedLines= (ws==0).astype(np.bool)                                                           
        watershedLines[np.logical_not(phaseImage)] = False                                                         
    del markers
    
    
    #Labeler des pores séparés par les lignes de partage de niveau
    
    pores=ndimage.measurements.label(np.logical_and(phaseImage,np.logical_not(watershedLines)), 
                                     structure=structuringElement 
                                     )[0]
    pores[np.logical_not(phaseImage)] = 0
    memoryType=utilities.BestMemoryType(pores.max())
    pores=pores.astype(memoryType)
    
    
    return pores,watershedLines,distanceMap
    
    
    
    
    
#---------------------------------------------------------------------------------------------- 
def FindLinks(myImage,pores,watershedLines,structuringElement):
    #Trouve les liens. Algorithme :
    #1) Attribuer à chaque voxel de watershedLines un lien s'il a exactement deux 
    #pores voisins.
    #2) Les autres voxels de watershedlines sont attribués au lien le plus représenté
    #dans les voxels voisins



    #1) Trouver les pores voisins de chaque voxel de watershed line
    
    imageSize=pores.shape
    assert myImage.shape == watershedLines.shape == imageSize
    
        #Side bands to avoid unexpected boundary effects
    seSize=int(np.shape(structuringElement)[1]//2)
    sideBandes=np.ones((imageSize[0],imageSize[1],imageSize[2]), dtype=bool)
    sideBandes[seSize:-seSize,seSize:-seSize,seSize:-seSize]=False
    watershedLines[sideBandes]=0
    
        #Remove useless parts of watershedLines
    links=np.logical_and(watershedLines,np.logical_not(myImage))
    
        #Boucle sur les voxels de watershedlines pour trouver leurs pores voisins
    indices=links.ravel().nonzero()[0]
    nVox=indices.size
    structuringElement=structuringElement.astype(np.bool)
    linksToPores=[]    
    correctionList=[]


    def StudyNeighbourhood(pores,indices,structuringElement,imageSize):  
        X,Y,Z =np.unravel_index(indices,imageSize)
        oneColumn=np.ones(X.shape,dtype=np.int)        
        neighboor=[]
        center = structuringElement.shape[0]//2
        for iSE in range(structuringElement.size):        
            xIse,yIse,zIse = np.unravel_index(iSE,structuringElement.shape)
            if structuringElement[xIse,yIse,zIse]:
                shiftX,shiftY,shiftZ = xIse-center,yIse-center,zIse-center
                neighboor.append( pores[X+shiftX*oneColumn,
                                        Y+shiftY*oneColumn,
                                        Z+shiftZ*oneColumn] )
                #TODO:horzcat au lieu de append pour avoir une seule matrice
        U=[set([neighboor[j][i] for j in range(len(neighboor))])-{0}  for i in range(nVox)]
        #TODO pour gagner en temps de calcul: pour trouver les colonnes à deux elements,  
        #retrancher le max de chaque colonne puis count non zero (deux fois) 
        linksToPores=[[[min(U[i]),max(U[i])],indices[i]] for i in range(nVox) if len(U[i])==2]
        correctionList=[indices[i] for i in range(nVox) if len(U[i])!=2]
        return linksToPores,correctionList


    def FillDict(linksToPores):
        key=[[str(linksToPores[i][0][0])+'_'+str(linksToPores[i][0][1]),i] 
                                         for i in range(len(linksToPores)) 
                                         if not (linksToPores[i] is None)]        

        mydict = defaultdict(list)
        for i in range(len(key)):
            mydict[key[i][0]].append(linksToPores[key[i][1]][1])

        return mydict
            
             
    def LabelLinks(mydict,imageSize):
        mykeys= mydict.keys()
        nLink = len(mykeys)
        memoryType = utilities.BestMemoryType(nLink)
        labeledLinks=np.zeros(imageSize,dtype=memoryType)
        for iLink in range(nLink):
            ind=mydict[mykeys[iLink]]
            X,Y,Z=np.unravel_index(ind,imageSize)
            labeledLinks[X,Y,Z] = iLink+1
        return labeledLinks


    def BuildInterfaceToPore(mydict) :
        mykeys= mydict.keys()
           
        interfaceToPore=np.asarray([ [ int(re.search('\w+(?<=_)', mykeys[i]).group(0)[0:-1]),
                            int(re.search('(?<=_)\w+', mykeys[i]).group(0))] 
                            for i in range(len(mykeys))  ])
        return interfaceToPore                         
    
    
    
    linksToPores,correctionList = StudyNeighbourhood(pores,indices,
                                                   structuringElement,imageSize)

    mydict = FillDict(linksToPores)
    
    del linksToPores  
    
    labeledLinks = LabelLinks(mydict,imageSize)
    
    interfaceToPore = BuildInterfaceToPore(mydict)
               
    del mydict           
               
    return labeledLinks, interfaceToPore    
    



#----------------------------------------------------------------------------------------------
def AnalyseElementsGeometry(myImg,pores,links,distanceMap,phases={'void':False}):
    
    
    PNMGeometricData=dict()
    
    phasesCodes = np.setdiff1d(np.unique(myImg),0)
    
    # Pores : infos sur la forme, position des pores
    poreLabels = range(1,pores.max()+1)

    pores_CenterOfMass    = PoresGeometry_CenterOfMass(pores)    
    
    pores_Volume          = PoresGeometry_Volume(pores)  
    
    pores_InscribedSphereRadius = PoresGeometry_InscribedSphereRadius(pores,distanceMap,poreLabels)
    
    poreVoxelLookUpTable  = BuildVoxelLookUpTable(pores)
    pores_NeighborPhases  = PoresGeometry_NeighborPhases( myImg,pores,poreLabels,
                                                 poreVoxelLookUpTable,
                                                 phasesCodes)
    
    PNMGeometricData['poreCenterOfMass']          = pores_CenterOfMass
    PNMGeometricData['poreVolumes']               = pores_Volume
    PNMGeometricData['poreInscribedSphereRadius'] = pores_InscribedSphereRadius
    PNMGeometricData['poresNeighborPhases']       = pores_NeighborPhases
    
    del poreVoxelLookUpTable,pores_CenterOfMass,pores_Volume,pores_NeighborPhases,pores_InscribedSphereRadius
    
    
    # Liens internes : infos sur la forme et position des liens internes    
    
    
    linkLabels = range(1,links.max()+1)
    
    links_CenterOfMass          = LinksGeometry_CenterOfMass(links,linkLabels)    
    
    links_WidestLocation        = LinksGeometry_WidestLocation(links,distanceMap,linkLabels)
    
    links_InscribedSphereRadius = LinksGeometry_InscribedSphereRadius(
                                                links,distanceMap,linkLabels) 
    
    links_SurfaceArea           = LinksGeometry_SurfaceArea(links,linkLabels)
    
#    links_HydraulicDiameter     = LinksGeometry_HydraulicDiameter(
#                                                myImg,pores,links,linkLabels,
#                                                linkLabelEnds,linkOrderLabels)
    linkVoxelLookUpTable        = BuildVoxelLookUpTable(links)
    links_NeighborPhases        = LinksGeometry_NeighborPhases(
                                                myImg,links,linkLabels,
                                                linkVoxelLookUpTable, 
                                                phasesCodes )          

    PNMGeometricData['internalLinkCenterOfMass']       = links_CenterOfMass 
    PNMGeometricData['internalLinkWidestLocation']    = links_WidestLocation
    PNMGeometricData['internalLinkCapillaryRadius']   = links_InscribedSphereRadius
    PNMGeometricData['internalLinkGeometricSurface']  = links_SurfaceArea
    #PNMGeometricData['internalLinkHydraulicDiameter'] = links_HydraulicDiameter
    PNMGeometricData['internalLinkNeighborPhases']    = links_NeighborPhases
    
    del linkVoxelLookUpTable,links_CenterOfMass,links_WidestLocation,links_InscribedSphereRadius,links_SurfaceArea,links_NeighborPhases  
    
    # Infos sur la forme, position des liens frontiere
        
    for iBoundary in range(6):
        if iBoundary == 0:    
            boundarySlice=pores[0,:,:]
            boundaryDistances=distanceMap[0,:,:]
            
        elif iBoundary == 1:
            boundarySlice=pores[-1,:,:]
            boundaryDistances=distanceMap[-1,:,:]
            
        elif iBoundary == 2:
            boundarySlice=pores[:,0,:]
            boundaryDistances=distanceMap[:,0,:]
            
        elif iBoundary == 3:
            boundarySlice=pores[:,-1,:]
            boundaryDistances=distanceMap[:,-1,:]
            
        elif iBoundary == 4:
            boundarySlice=pores[:,:,0]
            boundaryDistances=distanceMap[:,:,0]
            
        elif iBoundary == 5:
            boundarySlice=pores[:,:,-1]
            boundaryDistances=distanceMap[:,:,-1]
        
        linkLabels = range(1,pores.max()+1)
                
        links_CenterOfMass          = LinksGeometry_CenterOfMass(boundarySlice,linkLabels)

        links_WidestLocation  = LinksGeometry_WidestLocation(boundarySlice,boundaryDistances,linkLabels)

        inscribedSphereRadius = LinksGeometry_InscribedSphereRadius(
                                    boundarySlice,boundaryDistances,linkLabels)
        
        surfaceArea           = LinksGeometry_SurfaceArea(boundarySlice,linkLabels)
                           
        linkVoxelLookUpTable  = BuildVoxelLookUpTable(boundarySlice)                   
        links_NeighborPhases  = LinksGeometry_NeighborPhases(
                                                boundarySlice,boundarySlice,linkLabels,
                                                linkVoxelLookUpTable, 
                                                phasesCodes )                   
                           
        PNMGeometricData['boundaryCenterOfMass'+str(iBoundary)]          = links_CenterOfMass
        PNMGeometricData['boundaryWidestLocation'+str(iBoundary)]   = links_WidestLocation
        PNMGeometricData['boundaryCapillaryRadius'+str(iBoundary)]  = inscribedSphereRadius
        PNMGeometricData['boundaryGeometricSurface'+str(iBoundary)] = surfaceArea
        PNMGeometricData['boundaryNeighborPhases'+str(iBoundary)]   = links_NeighborPhases
    
        del links_CenterOfMass,links_WidestLocation,inscribedSphereRadius,surfaceArea,links_NeighborPhases
        
    return PNMGeometricData





#----------------------------------------------------------------------------------------------
def PoresGeometry_CenterOfMass(pores): 
    
    pores_center=ndimage.measurements.center_of_mass(pores, labels=pores ,
                                                     index=range(1,pores.max()+1))
                                                     
    pores_center=np.transpose(np.squeeze(np.dstack(pores_center)))

    return pores_center


#----------------------------------------------------------------------------------------------
def PoresGeometry_Volume(pores):
    
    pores_volumes=ndimage.measurements.labeled_comprehension(
                                            pores, pores,range(1,pores.max()+1),
                                            np.size,np.int,0)

    return pores_volumes


#----------------------------------------------------------------------------------------------
def PoresGeometry_InscribedSphereRadius(pores,distanceMap,poreLabels):

    pores_InscribedSphereRadius = ndimage.measurements.labeled_comprehension(
                                            distanceMap, pores, poreLabels,
                                            np.max,np.float16,0).astype(np.float32)
    
    return pores_InscribedSphereRadius
         
#----------------------------------------------------------------------------------------------
#@jit
def PoresGeometry_NeighborPhases(myImg,pores,poreLabels,voxelLookUpTable,phasesCodes): 
    
    phasesCodes = np.asarray(phasesCodes)
    nPhase = phasesCodes.size
    
    poreLabels = np.asarray(poreLabels)
    nPoreLabel = poreLabels.size 
    surfaceComposition = np.zeros((nPoreLabel,nPhase))
    
    imageShape = myImg.shape  
                    
    dimension = pores.ndim                 
    structElement=  np.ones(3*np.ones(dimension),dtype=bool)           
                        
    for iPore in range(nPoreLabel):
        
        voxels = GetVoxelOfLabel(poreLabels[iPore],voxelLookUpTable)
        
        if len(voxels)>0 :
            Xmin,Xmax=max(voxels[0].min()-1,0),min(voxels[0].max()+1,imageShape[0]-1)
            Ymin,Ymax=max(voxels[1].min()-1,0),min(voxels[1].max()+1,imageShape[1]-1)
            Zmin,Zmax=max(voxels[2].min()-1,0),min(voxels[2].max()+1,imageShape[2]-1)       
            
            #poreImage = pores[Xmin:Xmax+1,Ymin:Ymax+1,Zmin:Zmax+1]       
            
            localMyImg = myImg[Xmin:Xmax+1,Ymin:Ymax+1,Zmin:Zmax+1]
            poreImage = np.zeros(localMyImg.shape,dtype=np.bool)
            poreImage[voxels[0]-Xmin,voxels[1]-Ymin,voxels[2]-Zmin]=True
            
            dilatedPore = imageAnalysis.Morphology.FastDilation(poreImage,structElement)
                  
            poreSurfaceNeighborhood = localMyImg[np.logical_and(dilatedPore,
                                                                localMyImg.astype(np.bool))]
    
            surfaceComposition[iPore,:] = ndimage.measurements.labeled_comprehension(
                                                poreSurfaceNeighborhood, poreSurfaceNeighborhood, 
                                                phasesCodes,
                                                np.size,np.int,0)
        else:
            surfaceComposition[iPore,:] = np.zeros(nPhase,dtype=np.int)
            
          
    return surfaceComposition

#----------------------------------------------------------------------------------------------
def LinksGeometry_CenterOfMass(links,linkLabels): 

    links_center_arg=ndimage.measurements.center_of_mass(links, labels=links,
                                                     index=linkLabels)
                                                     
    links_center=np.transpose(np.squeeze(np.dstack(links_center_arg)))

    return links_center

#----------------------------------------------------------------------------------------------
def LinksGeometry_WidestLocation(links,distanceMap,linkLabels):

    links_WidestLocation_arg=ndimage.measurements.maximum_position(
                                            distanceMap, links, linkLabels)
                                            
    links_WidestLocation=np.transpose(np.squeeze(np.dstack(links_WidestLocation_arg)))

    return links_WidestLocation


#----------------------------------------------------------------------------------------------
def LinksGeometry_InscribedSphereRadius(links,distanceMap,linkLabels):

    links_InscribedSphereRadius = ndimage.measurements.labeled_comprehension(
                                            distanceMap, links, linkLabels,
                                            np.max,np.float16,0).astype(np.float32)
    
    return links_InscribedSphereRadius
    
    
#----------------------------------------------------------------------------------------------
def LinksGeometry_SurfaceArea(links,linkLabels):
    
    links_SurfaceArea=ndimage.measurements.labeled_comprehension(
                                            links, links, 
                                            linkLabels,
                                            np.size,np.int,0).astype(np.int)
    
    return links_SurfaceArea
    
    
#---------------------------------------------------------------------------------------------- 
def LinksGeometry_HydraulicDiameter(myImg,pores,links,linkLabels,voxelLookUpTable):     
        
#    nLink = len(linkLabels)    
#    hydraulicDiameters = np.zeros(nLink)
#    
#    void= (myImg==0)
#    for iLink in linkLabels:
#        
#        dilatedLink = ndimage.binary_dilation(links==iLink,
#                                              structure=np.ones((3,3,3),dtype=bool))        
#        
#        linkNeighborhood = np.logical_and(dilatedLink,void)
#
#        volume = np.count_nonzero(linkNeighborhood)        
#        
#        surfaceArea =         1 #TODO : 
#        
#        hydraulicDiameters[iLink] = volume/surfaceArea
     
     
    return 1


#---------------------------------------------------------------------------------------------- 
#@jit 
def LinksGeometry_NeighborPhases(myImg,links,linkLabels,voxelLookUpTable,phasesCodes):      
    
    phasesCodes = np.asarray(phasesCodes)
    nPhase = phasesCodes.size
    
    imageShape = myImg.shape   
    
    dimension = links.ndim                 
    structElement=  np.ones(3*np.ones(dimension),dtype=np.bool) 
    
    linkLabels = np.asarray(linkLabels)
    nLinkLabel = linkLabels.size 
    surfaceComposition = np.zeros((nLinkLabel,nPhase))  
    
    for iLink in range(nLinkLabel):
        
        voxels = GetVoxelOfLabel(linkLabels[iLink],voxelLookUpTable)
        
        if len(voxels)>0:
            if dimension==3:
                Xmin,Xmax=max(voxels[0].min()-1,0),min(voxels[0].max()+1,imageShape[0]-1)
                Ymin,Ymax=max(voxels[1].min()-1,0),min(voxels[1].max()+1,imageShape[1]-1)
                Zmin,Zmax=max(voxels[2].min()-1,0),min(voxels[2].max()+1,imageShape[2]-1)       
                
                localMyImg = myImg[Xmin:Xmax+1,Ymin:Ymax+1,Zmin:Zmax+1]
                linkImage = np.zeros(localMyImg.shape,dtype=np.bool)
                linkImage[voxels[0]-Xmin,voxels[1]-Ymin,voxels[2]-Zmin]=True
                #linkImage = links[Xmin:Xmax+1,Ymin:Ymax+1,Zmin:Zmax+1]       
                
            elif dimension==2:
                Xmin,Xmax=max(voxels[0].min()-1,0),min(voxels[0].max()+1,imageShape[0]-1)
                Ymin,Ymax=max(voxels[1].min()-1,0),min(voxels[1].max()+1,imageShape[1]-1)
                
                localMyImg = myImg[Xmin:Xmax+1,Ymin:Ymax+1] 
                linkImage = np.zeros(localMyImg.shape,dtype=np.bool)
                linkImage[voxels[0]-Xmin,voxels[1]-Ymin]=True
                #linkImage = links[Xmin:Xmax+1,Ymin:Ymax+1]       
             
            dilatedLink = imageAnalysis.Morphology.FastDilation(linkImage,structElement)
       
            
            linkSurfaceNeighborhood = localMyImg[np.logical_and(dilatedLink,
                                                                localMyImg.astype(np.bool))]
    
            surfaceComposition[iLink,:] = ndimage.measurements.labeled_comprehension(
                                                linkSurfaceNeighborhood, linkSurfaceNeighborhood, 
                                                phasesCodes,
                                                np.size,np.int,0)
        
        else:
            surfaceComposition[iLink,:] = np.zeros(nPhase,dtype=np.int)
     
    
    return surfaceComposition



#----------------------------------------------------------------------------------------------     
def BuildVoxelLookUpTable(labeledImage):
    
    imageShape = labeledImage.shape
    
    labelImage = labeledImage.reshape((labeledImage.size))
    orderLabels = np.argsort(labelImage)
    sortedLabels = labelImage[orderLabels]
    del labelImage
    labelEnds=np.flatnonzero(np.roll(sortedLabels,-1)-sortedLabels)  
    #labelEnds=np.append(labelEnds,labelImage.size-1)

    #what happens if labelled Image is filled with zeros ?
    labelIndices = np.zeros(labeledImage.max()+1,dtype=np.int)
    labelIndices[sortedLabels[labelEnds]]=np.arange(labelEnds.size)
    
    voxelLookUpTable = labelEnds,orderLabels,labelIndices,imageShape
    
    return voxelLookUpTable
    
#----------------------------------------------------------------------------------------------    
#@jit
def GetVoxelOfLabel(numLabel,voxelLookUpTable):
    
    labelEnds,orderLabels,labelIndices,imageShape=voxelLookUpTable
    
    if numLabel>=0 and numLabel<labelIndices.size:
        iLabel = labelIndices[numLabel]
        if iLabel>0 or numLabel==0:
            vRange = np.arange(labelEnds[iLabel-1]+1,labelEnds[iLabel]+1)
            voxelIndices = orderLabels[vRange]
            voxelIndices = np.unravel_index(voxelIndices,imageShape)
        else:
            voxelIndices=[]
        
    else:
        voxelIndices=[]
    
    
    return voxelIndices
    
 


    
