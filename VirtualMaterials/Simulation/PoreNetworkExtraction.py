# -*- coding: utf-8 -*-


import numpy as np
import hdf5storage
from scipy import ndimage
from skimage import morphology
from skimage import feature
import SimpleITK as sitk
from VirtualMaterials.Utilities  import tifffile as tff
from collections import defaultdict
import time
import re 

#import mahotas
#from skimage import io as skimio


#----------------------------------------------------------------------------------------------
def ExtractNetwork(inputFileName,outputFileName,phases={'void':False},hContrast=4):


    beginTime=time.time()
    
    structuringElement = np.ones((3,3,3))
    distanceType='ITKDanielson'
    
    #Load image from disk
    myImg=tff.imread(inputFileName).astype(np.uint8)    
    
    
    #Perform pores segmentation
    print('PoresWatershedSegmentation')
    
    pores,watershedLines,distanceMap,porePhase = PoresSegmentation(myImg,
                                                structuringElement=structuringElement,
                                                hContrast=hContrast,
                                                distanceType=distanceType,phases=phases)
    
    
    print('FindLinks')
    
    links,interfaceToPore=FindLinks(myImg,pores,watershedLines,structuringElement)
    
    
    print('AnalyseElementsGeometry')
    
    PNMGeometricData = AnalyseElementsGeometry(myImg,pores,links,distanceMap)
    

    print('Saving results to disk')
    
    PNMGeometricData.update({'interfaceToPore':interfaceToPore,'imagePores':pores,
                             'porePhase':porePhase,'myImage':myImg.astype(np.bool)})
    hdf5storage.savemat(outputFileName,mdict=PNMGeometricData)
    
    
    endTime=time.time()
    print("Time spent : {} s".format(endTime-beginTime))
    
    
    
    
    
#----------------------------------------------------------------------------------------------
def PoresSegmentation(myImg,phases={'void':False},structuringElement=np.ones((3,3,3)),hContrast=4,distanceType='ITKDanielson'):
    #Phases=dict, phases['myphase']=codeForMyPhaseInImage
    
    pores=np.zeros(myImg.shape,dtype=np.uint32)
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
                            hContrast=hContrast,distanceType=distanceType)
        
        phaseImage=phaseImage.astype(np.bool)
        pores[phaseImage]=poresPhase[phaseImage]+labelShift[-1]*(poresPhase[phaseImage]>0
                                                                ).astype(np.uint32)
        
        watershedLines[phaseImage] = watershedLinesPhase[phaseImage]
        distanceMap[phaseImage] = distanceMapPhase[phaseImage]
        del phaseImage, poresPhase,watershedLinesPhase,distanceMapPhase
        
        labelShift.append(pores.max())
        
        
    phaseBoundaries = CannyEdgeDetection(myImg)     
    watershedLines=np.logical_or(watershedLines,phaseBoundaries)
    
    
    porePhase=np.zeros(pores.max(),dtype=np.uint8)    
    for i in range(len(labelShift)-1):
        porePhase[np.arange(labelShift[i],labelShift[i+1])]=int(phases[phases.keys()[i]])
    
    return pores,watershedLines,distanceMap,porePhase
    
    
    
    
    
    
#----------------------------------------------------------------------------------------------
def PoresWatershedSegmentationOnePhase(phaseImage,structuringElement=np.ones((3,3,3)),hContrast=4,
                                       distanceType='ITKDanielson',markerChoice='Hmaxima',
                                       watershedAlgo='ITK'):
    
    
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
        distanceMap=sitk.GetArrayFromImage(itkdistanceMap).astype(memoryType) 
    
    
    #Choix des marqueurs pour la segmentation (centres des pores) : H-maxima :
    #maxima de la carte de distance dont les pics sont étêtés d'une hauteur h. Utilise  
    #une recontruction morphologique pour construire la carte de distance étêtée.
    
    
    if markerChoice=='Hmaxima' and hContrast>0:
        hContrast=np.asarray(hContrast).astype(memoryType)
        reconstructed=morphology.reconstruction(distanceMap-hContrast, distanceMap
                                            ).astype(memoryType)
        
        local_maxi=(distanceMap-reconstructed).astype(np.bool)
        del reconstructed
    else:
        local_maxi= feature.peak_local_max(distanceMap.astype(memoryType), 
                                           min_distance=10, indices=False)
        
    
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
#        mask = itk.MaskImageFilter.IUC2IUC2IUC2.New(ws, fill)
#        overlay = itk.LabelOverlayImageFilter.IUC2IUC2IRGBUC2.New(reader,
#                                                                  mask,
#                                                                  UseBackground=True)
        ws=sitk.GetArrayFromImage(wsITK).astype(np.uint8) 
        watershedLines= (ws==0).astype(np.bool)                                                           
        watershedLines[np.logical_not(phaseImage)] = False                                                         
    del markers
    
    
    #Labeler des pores séparés par les lignes de partage de niveau
    
    pores=ndimage.measurements.label(np.logical_and(phaseImage,np.logical_not(watershedLines)), 
                                     structure=structuringElement 
                                     )[0]
    pores[np.logical_not(phaseImage)] = 0
    
    
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
    sideBandes[seSize:-1-seSize,seSize:-1-seSize,seSize:-1-seSize]=False
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
        for iSE in range(structuringElement.size):        
            xIse,yIse,zIse = np.unravel_index(iSE,structuringElement.shape)
            if structuringElement[xIse,yIse,zIse]:
                center = (structuringElement.shape[0]+1)/2
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
                         

    def LabelLinks(mydict,links,imageSize):
        mykeys= mydict.keys()     
        for iLink in range(len(mykeys)):
            ind=mydict[mykeys[iLink]]
            X,Y,Z=np.unravel_index(ind,imageSize)
            links[X,Y,Z] = iLink+1
        return links


    def BuildInterfaceToPore(mydict) :
        mykeys= mydict.keys()
           
        interfaceToPore=np.asarray([ [ int(re.search('\w+(?<=_)', mykeys[i]).group(0)[0:-1]),
                            int(re.search('(?<=_)\w+', mykeys[i]).group(0))] 
                            for i in range(len(mykeys))  ])
        return interfaceToPore                         
    
    
    
    linksToPores,correctionList = StudyNeighbourhood(pores,indices,
                                                   structuringElement,imageSize)

    mydict = FillDict(linksToPores)
    
    links = links.astype(np.int)
    
    links = LabelLinks(mydict,links,imageSize)
    
    interfaceToPore = BuildInterfaceToPore(mydict)
               
               
               
    #2) Trouver les lien le plus présent parmis les voisins des voxels restants


#  countCORRECTIONLIST = len(correctionLIST)
    
     
#    if countCORRECTIONLIST>0:
#        
#        #If necessary, add a one-pixel border around the x and y edges of the
#        #array.  This prevents an error if the code tries to interpolate a ray at
#        #the edge of the x,y grid.
#        cond0 = min([correctionLIST[i][0] for i in range(len(correctionLIST))])==0
#        cond1 = max([correctionLIST[i][0] for i in range(len(correctionLIST))]
#                                                    )==sampleDimensions[0]-1
#        cond2 = min([correctionLIST[i][1] for i in range(len(correctionLIST))])==0
#        cond3 = max([correctionLIST[i][1] for i in range(len(correctionLIST))]
#                                                    )==sampleDimensions[1]-1
#    
#        if cond0 or cond1 or cond2 or cond3:
#            image = np.hstack( (np.zeros((sampleDimensions[0],1,sampleDimensions[2])),
#                                image,np.zeros((sampleDimensions[0],1,sampleDimensions[2]))))
#            image = np.vstack( (np.zeros((1,sampleDimensions[1]+2,sampleDimensions[2])),
#                                image,np.zeros((1,sampleDimensions[1]+2,sampleDimensions[2]))))
#            correctionLIST = [ [correctionLIST[i][0]+1,correctionLIST[i][1]+1] 
#                                            for i in range(len(correctionLIST)) ]
#        
#        for loopC in range(countCORRECTIONLIST):
#            voxelsforcorrection = np.squeeze( np.max( [ 
#                image[correctionLIST[loopC][0]-1,correctionLIST[loopC][1]-1,:],
#                image[correctionLIST[loopC][0]-1,correctionLIST[loopC][1],:],
#                image[correctionLIST[loopC][0]-1,correctionLIST[loopC][1]+1,:],
#                image[correctionLIST[loopC][0],correctionLIST[loopC][1]-1,:],
#                image[correctionLIST[loopC][0],correctionLIST[loopC][1]+1,:],
#                image[correctionLIST[loopC][0]+1,correctionLIST[loopC][1]-1,:],
#                image[correctionLIST[loopC][0]+1,correctionLIST[loopC][1],:],
#                image[correctionLIST[loopC][0]+1,correctionLIST[loopC][1]+1,:],
#                ], axis=0 ) )
#            voxelsforcorrection = (voxelsforcorrection>=4)
#            image[correctionLIST[loopC][0],correctionLIST[loopC][1],voxelsforcorrection] = 1
 
    return links, interfaceToPore    
    



#----------------------------------------------------------------------------------------------
def AnalyseElementsGeometry(myImg,pores,links,distanceMap):
    
    
    PNMGeometricData=dict()
    
    
    # Pores : infos sur la forme, position des pores

    pores_Center          = PoresGeometry_Center(pores)    
    
    pores_Volume          = PoresGeometry_Volume(pores)  
        
    pores_NeighborPhases  = PoresGeometry_NeighborPhases(myImg,pores,links)
    
    PNMGeometricData['poreCenters']         = pores_Center
    PNMGeometricData['poreVolumes']         = pores_Volume
    PNMGeometricData['poresNeighborPhases'] = pores_NeighborPhases
    
    
    # Liens internes : infos sur la forme et position des liens internes    
    linkLabels = range(1,links.max()+1)
    
    links_Center                = LinksGeometry_Center(links,distanceMap,linkLabels)    
    
    links_InscribedSphereRadius = LinksGeometry_InscribedSphereRadius(links,distanceMap,linkLabels) 
    
    links_SurfaceArea           = LinksGeometry_SurfaceArea(links,linkLabels)
    
    links_HydraulicDiameter     = LinksGeometry_HydraulicDiameter(myImg,pores,links,linkLabels)

    links_NeighborPhases        = LinksGeometry_NeighborPhases(myImg,pores,links,linkLabels)          

    PNMGeometricData['internalLinkBarycenters']       = links_Center 
    PNMGeometricData['internalLinkCapillaryRadius']   = links_InscribedSphereRadius
    PNMGeometricData['internalLinkGeometricSurface']  = links_SurfaceArea
    PNMGeometricData['internalLinkHydraulicDiameter'] = links_HydraulicDiameter
    PNMGeometricData['internalLinkNeighborPhases']    = links_NeighborPhases
    
    
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
                
        links_center          = LinksGeometry_Center(boundarySlice,boundaryDistances,linkLabels)
        
        inscribedSphereRadius = LinksGeometry_InscribedSphereRadius(boundarySlice,boundaryDistances,linkLabels)
        
        surfaceArea           = LinksGeometry_SurfaceArea(boundarySlice,linkLabels)
                                
        PNMGeometricData['boundaryCenters'+str(iBoundary)]          = links_center
        PNMGeometricData['boundaryCapillaryRadius'+str(iBoundary)]  = inscribedSphereRadius
        PNMGeometricData['boundaryGeometricSurface'+str(iBoundary)] = surfaceArea
    
    
    return PNMGeometricData





#----------------------------------------------------------------------------------------------
def PoresGeometry_Center(pores): 
    
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
def PoresGeometry_NeighborPhases(myImg,pores,links): 

    nPore = pores.max()     
    phases = np.setdiff1d(np.unique(myImg),0)
    nPhase = phases.size
     
    surfaceComposition = np.zeros((nPore,nPhase))
    
    void= (myImg==0)
    for iPore in range(nPore):
        
        dilatedPore = ndimage.binary_dilation(pores==iPore,
                                              structure=np.ones((3,3,3),dtype=bool))        
        
        poreSurfaceNeighborhood = myImg[np.logical_and(dilatedPore,np.logical_not(void))]

        surfaceComposition[iPore,:] = ndimage.measurements.labeled_comprehension(
                                            poreSurfaceNeighborhood, poreSurfaceNeighborhood, 
                                            phases,
                                            np.size,np.int,0)
        
     
    return surfaceComposition
     
     
     
     

#----------------------------------------------------------------------------------------------
def LinksGeometry_Center(links,distanceMap,linkLabels): 

    links_center_arg=ndimage.measurements.maximum_position(
                                            distanceMap, links, linkLabels)
                                            
    links_center=np.transpose(np.squeeze(np.dstack(links_center_arg)))

    return links_center


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
def LinksGeometry_HydraulicDiameter(myImg,pores,links,linkLabels):     
            
    hydraulicDiameters = np.zeros(nLink)
    
    void= (myImg==0)
    for iLink in linkLabels:
        
        dilatedLink = ndimage.binary_dilation(links==iLink,
                                              structure=np.ones((3,3,3),dtype=bool))        
        
        linkNeighborhood = np.logical_and(dilatedLink,void)

        volume = np.count_nonzero(linkNeighborhood)        
        
        surfaceArea =         
        
        hydraulicDiameters[iLink] = volume/surfaceArea
     
     
     return 1
     
#----------------------------------------------------------------------------------------------     
def LinksGeometry_NeighborPhases(myImg,pores,links,linkLabels):      
     
    phases = np.setdiff1d(np.unique(myImg),0)
    nPhase = phases.size
     
    surfaceComposition = np.zeros((nLink,nPhase))
    
    void= (myImg==0)
    for iLink in linkLabels:
        
        dilatedLink = ndimage.binary_dilation(links==iLink,
                                              structure=np.ones((3,3,3),dtype=bool))        
        
        linkSurfaceNeighborhood = myImg[np.logical_and(dilatedLink,np.logical_not(void))]

        surfaceComposition[iLink,:] = ndimage.measurements.labeled_comprehension(
                                            linkSurfaceNeighborhood, linkSurfaceNeighborhood, 
                                            phases,
                                            np.size,np.int,0)
        
     return surfaceComposition



 
 
##-----------------------------------------------------------------------------
def CannyEdgeDetection(myImg,variance=2):    
    #Uses ITK canny edge detector to detect boundaries between phases in the 
    #material image

    #Load image into SimpleITK
    myItkImage = sitk.GetImageFromArray(myImg.astype(np.uint8))
    caster = sitk.CastImageFilter()
    caster.SetOutputPixelType(sitk.sitkFloat32)
    floatImage = caster.Execute( myItkImage )
    
    #Canny edge detection
    canny = sitk.CannyEdgeDetectionImageFilter()
    variance=float(variance)
    canny.SetVariance( [ variance,variance,variance] )
    #canny.SetLowerThreshold( 10 )
    #canny.SetUpperThreshold( 1000 ) 
    myItkImage = canny.Execute( floatImage )
    
    #Go back to a numpy array image
    caster = sitk.CastImageFilter()
    caster.SetOutputPixelType(sitk.sitkInt8)
    myItkImage = caster.Execute( myItkImage )
    phaseBoundaries = sitk.GetArrayFromImage(myItkImage).astype(np.bool)

    return phaseBoundaries

     
     
     
     
     
#----------------------------------------------------------------------------------------------
def Test():
    inputfile='testGDL.tif'
    outputfile='/home/270.12-Modeling_PEMFC_Li/theseTristan/PSI_drainage_python/watershedTest_GDL_h4_euclidean.mat'
    hContrast=2
    myphases={'void':0,'fiber':100,'binder':200}
    ExtractNetwork(inputfile,outputfile,hContrast,phases=myphases,distanceType='chamfer')

#----------------------------------------------------------------------------------------------

if __name__ == "__main__":
    Test()
    
