# -*- coding: utf-8 -*-
"""
Created on Mon Nov 10 09:39:57 2014

@author: Tristan Agaesse
"""
import numpy as np
import hdf5storage
import math
from scipy import ndimage
from skimage import io as skimio
from skimage import morphology
from skimage import feature
import mahotas


def ExtractNetwork(inputFileName,outputFileName,hContrast):

    structuringElement = np.ones((3,3,3))

    myImg=skimio.MultiImage(inputFileName)
    myImg=myImg.concatenate()
    myImg=myImg.swapaxes(0,2).astype(np.bool)
    
    print('PoresWatershedSegmentation')
    pores,watershedLines,distanceMap = PoresWatershedSegmentation(
                                            myImg,structuringElement,hContrast)
    
    print('FindLinks')
    links=FindLinks(myImg,pores,watershedLines,structuringElement.astype(np.bool))
    
    print('AnalysePoresSegmentationGeometry')
    PNMGeometricData, links = AnalysePoresSegmentationGeometry(
                                            myImg,structuringElement,
                                            pores,links,distanceMap)
    
    print('BuildConnectivityTables') 
    interfaceToPore = BuildConnectivityTables(pores,links);    

    PNMGeometricData.update({'interfaceToPore':interfaceToPore,'imagePores':pores})
    #PNMGeometricData.update({'pores':pores,'watershedLines':watershedLines})
    hdf5storage.savemat(outputFileName,mdict=PNMGeometricData)
    hdf5storage.savemat(inputFileName+"_reshaped.mat",{'myImage':myImg.astype(np.bool)})
    



#----------------------------------------------------------------------------------------------
def PoresWatershedSegmentation(myImg,structuringElement,hContrast) :
    
    #calcul de la carte de distanceMap
    distanceMap = ndimage.distance_transform_edt(np.logical_not(myImg)).astype(np.float16)
    
    #Choix des marqueurs pour la segmentation (centres des pores) : H-maxima :
    #maxima de la carte de distance dont les pics sont étêtés d'une hauteur h. Utilise une 
    #recontruction morphologique pour construire la carte de distance étêtée.
    hContrast=hContrast
    reconstructed=morphology.reconstruction(distanceMap-hContrast, distanceMap
                                            ).astype(np.float16)
    
    if hContrast>0.1:
        local_maxi=(distanceMap-reconstructed).astype(np.bool)
    else:
        local_maxi= feature.peak_local_max(distanceMap.astype(np.int8), 
                                           min_distance=10, indices=False)
        
    del reconstructed
    
    markers = ndimage.measurements.label(local_maxi , structure=structuringElement)[0]
    
    del local_maxi
    
    #Calcul des lignes de partage de niveau 
    _,watershedLines = mahotas.cwatershed(
                                (distanceMap.max()-distanceMap).astype(np.int8), 
                                markers, Bc=structuringElement , return_lines=True)

    del markers
    
    #Label des pores séparés par les lignes de partage de niveau
    pores=ndimage.measurements.label(np.logical_not(np.logical_or(myImg,watershedLines)), 
                                     structure=structuringElement 
                                     )[0]
    
    return pores,watershedLines,distanceMap
    

#----------------------------------------------------------------------------------------------
def AnalysePoresSegmentationGeometry(myImg,structuringElement,pores,links,distanceMap):
     

    structuringElement=structuringElement.astype(np.int)
    links_label=ndimage.label(links,structure=structuringElement)[0]
    
    
    # Infos sur la forme, position des liens internes
    links_center_arg=ndimage.measurements.maximum_position(
                                            distanceMap, links_label,
                                            range(1,links_label.max()+1))
                                            
    links_center=np.transpose(np.squeeze(np.dstack(links_center_arg)))   
    
    linkDiameterDistanceMap=ndimage.measurements.labeled_comprehension(
                                            distanceMap, links_label, 
                                            range(1,links_label.max()+1),
                                            max,np.float16,0)    
    
    
    # Dilatation des liens internes
    links=ndimage.morphology.grey_dilation(links_label,  size=(3,3,3))
    
    
    # Infos sur la forme, position des pores
    pores_center=ndimage.measurements.center_of_mass(pores, labels=pores ,
                                                     index=range(1,pores.max()+1))
                                                     
    pores_center=np.transpose(np.squeeze(np.dstack(pores_center)))
    pores_volumes=ndimage.measurements.labeled_comprehension(
                                            pores, pores,range(1,pores.max()+1),
                                            np.size,np.int32,0)
    
    PNMGeometricData=dict()
    #PNMGeometricData['imageLiensDilates']=links
    PNMGeometricData['internalLinkDiameters']=linkDiameterDistanceMap.astype(np.float32)
    PNMGeometricData['internalLinkBarycenters']=links_center
    PNMGeometricData['poreCenters']=pores_center
    PNMGeometricData['poreVolumes']=pores_volumes
    
    
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
        
        links_center_arg=ndimage.measurements.maximum_position(
                                            boundaryDistances, boundarySlice, 
                                            range(1,pores.max()+1))
                                            
        links_center=np.transpose(np.squeeze(np.dstack(links_center_arg))) 
        
        diameterDistanceMap=ndimage.measurements.labeled_comprehension(
                                        boundaryDistances, boundarySlice, 
                                        range(1,pores.max()+1),max,np.int16,0)
                                        
        PNMGeometricData['boundaryCenters'+str(iBoundary)]=links_center
        PNMGeometricData['boundaryDiameters'+str(iBoundary)]=diameterDistanceMap.astype(np.float32)
      
      
    return PNMGeometricData,links


#----------------------------------------------------------------------------------------------
def FindLinks(myImage,pores,watershedLines,structuringElement):
#FindLinks  Trouve les parties de watershed qui sont entre deux pores
#Input: myImage,pores,watershedLines,structuringElement
#Output : links
        
    imageSize=pores.shape
    assert myImage.shape == imageSize
    assert watershedLines.shape == imageSize
        
    #structuring element size
#    foo=structuringElement.shape
###    assert(isequal(foo,foo(1)*ones(1,3)) && rem(foo(1),2)==1,'Wrong size of structuring element');
#    seSize=int(math.floor(foo[1]/2))
#    sideBandes=np.ones((imageSize[0],imageSize[1],imageSize[2]), dtype=bool)
#    sideBandes[seSize:-1-seSize,seSize:-1-seSize,seSize:-1-seSize]=False
#    watershedLines[sideBandes]=0
#   
    links=np.logical_and(watershedLines,np.logical_not(myImage))
    
#    indices=links.ravel().nonzero()[0]
#   
#    for i in range(indices.size):
#        ind=indices[i]
#        (x,y,z)=np.unravel_index(ind,imageSize)
#        localPores=pores[x-seSize:x+seSize+1,y-seSize:y+seSize+1,z-seSize:z+seSize+1]
#        if np.setdiff1d(np.unique(localPores[structuringElement]),[0]).size > 2 :
#            links[x,y,z]=False
#
#    assert np.count_nonzero(links[np.logical_not(watershedLines.astype(np.bool))])==0

    return links

#----------------------------------------------------------------------------------------------
 
def BuildConnectivityTables(poresImage,internalLinkImage):
    
    nInterface=internalLinkImage.max()
#    nPores=poresImage.max()
    #nVoxelsInterface=internalLinkImage.size()
    #reshapedInterface=reshape(internalLinkImage,[1,nVoxelsInterface])
    #[sortedInterface,orderInterface]=sort(reshapedInterface)
    orderInterface=np.argsort(internalLinkImage, axis=None, kind='quicksort', order=None)
    sortedInterface=internalLinkImage.flatten()[orderInterface]
    
    labelEnds=np.flatnonzero(np.roll(sortedInterface,1)-sortedInterface)
    assert(labelEnds.size == nInterface+1)    

    interfaceToPore = [] 
    for j in range(1,nInterface+1):
        
        intersection = poresImage.flatten()[orderInterface[labelEnds[j-1]+1:
                                                           labelEnds[j]+1]] 
        assert intersection.size>0                                                   
        intersectedPores=np.unique(intersection[intersection>0])
        if intersectedPores.size>0:        
            sizeIntersectionPores=ndimage.measurements.labeled_comprehension(
                                        intersection, intersection, 
                                        intersectedPores,np.size,np.int32,0
                                        )
        else:
            sizeIntersectionPores=[]
        interfaceToPore.append([intersectedPores,sizeIntersectionPores])


    return interfaceToPore
    
    
    
#----------------------------------------------------------------------------------------------
def Test():
    inputfile='PSI_sampleDrainage_635.tif'
    outputfile='datatest_635_4'
    hContrast=4
    ExtractNetwork(inputfile,outputfile,hContrast)

#----------------------------------------------------------------------------------------------

if __name__ == "__main__":
    Test()
    