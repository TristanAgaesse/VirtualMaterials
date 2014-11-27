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
import mahotas
#from imread import imread_multi
#import tifffile as tff

def main():
    
    foldername ='/home/ta240184/Documents/MATLAB/PNM_Matlab_V0.1/Simulations/PSI_drainage_python/'
    inputFileName = foldername+"PSI_sampleDrainage.tif"
    outputFileName = foldername+"segmentation_analyse_results_h2.mat" 
    
    hContrast = 2
    structuringElement = np.ones((3,3,3))
    
    
    
    myImg=skimio.MultiImage(inputFileName)
    myImg=myImg.concatenate()
    myImg=myImg.swapaxes(0,2).astype(np.bool)
    #myImg = imread_multi(foldername+"Dry_extended.tif")
    #myImg = np.dstack(myImg).astype(bool)
    
    pores,watershedLines,distanceMap = PoresWatershedSegmentation(myImg,structuringElement,hContrast)
    
    PNMGeometricData = AnalysePoresSegmentationGeometry(myImg,structuringElement,pores,watershedLines,distanceMap)
    
    PNMGeometricData.update({'myImage':myImg, 'pores':pores,'watershedLines':watershedLines,'distanceMap':distanceMap})
    hdf5storage.savemat(outputFileName,mdict=PNMGeometricData)
    
    #hdf5storage.savemat(foldername+"img_pores_distMap.mat",mdict={'myImage':myImg, 'pores':pores,'watershedLines':watershedLines,'distanceMap':distanceMap})
    #hdf5storage.savemat(foldername+"segmentation_analyse_results.mat",mdict=PNMGeometricData)




#----------------------------------------------------------------------------------------------
def PoresWatershedSegmentation(myImg,structuringElement,hContrast) :
    
    #calcul de la carte de distanceMap
    distanceMap = ndimage.distance_transform_cdt(np.logical_not(myImg),metric='chessboard').astype(np.int8)
    
    #Choix des marqueurs pour la segmentation (centres des pores) : H-maxima :
    #maxima de la carte de distance dont les pics sont étêtés d'une hauteur h. Utilise une 
    #recontruction morphologique pour construire la carte de distance étêtée.
    hContrast=np.int8(hContrast)
    reconstructed=morphology.reconstruction(distanceMap-hContrast, distanceMap).astype(np.int8)
    local_maxi=(distanceMap-reconstructed).astype(np.bool)
    
    del reconstructed
    
    markers = ndimage.measurements.label(local_maxi , structure=structuringElement)[0]
    
    del local_maxi
    
    #Calcul des lignes de partage de niveau 
    _,watershedLines = mahotas.cwatershed(distanceMap.max()-distanceMap, markers, Bc=structuringElement , return_lines=True)

    del markers
    
    #Label des pores séparés par les lignes de partage de niveau
    pores=ndimage.measurements.label(np.logical_not(np.logical_or(myImg,watershedLines)), structure=structuringElement )[0]
    
    return pores,watershedLines,distanceMap
    
    #hdf5storage.savemat(foldername+"pores_lines_img_dist.mat",mdict={'myImage':myImg, 'pores':pores,'watershedLines':watershedLines,'distanceMap':distanceMap})
    #pylab.imshow(pores[:,:,5])


#----------------------------------------------------------------------------------------------
def AnalysePoresSegmentationGeometry(myImg,structuringElement,pores,watershedLines,distanceMap):
    
#    foldername="/home/greentoto/Bureau/PSI_Invasion_eau/SGL 24BA/drainage/results_python/"
#    infos=hdf5storage.loadmat(foldername+"pores_lines_img_dist.mat")
#    #links=io.loadmat(foldername+"liens_bool.mat")['links']
#    distanceMap=infos['distanceMap']
#    pores=infos['pores']    
    
    structuringElement=structuringElement.astype(np.bool)
    links=FindLinks(myImg,pores,watershedLines,structuringElement)
    
    structuringElement=structuringElement.astype(np.int)
    links_label=ndimage.label(links,structure=structuringElement)[0]
    
    
    # Infos sur la forme, position des liens internes
#    links_center=ndimage.measurements.center_of_mass(links, labels=links_label ,index=range(1,links_label.max()+1))
#    links_center=np.transpose(np.squeeze(np.dstack(links_center)))
    links_center_arg=ndimage.measurements.maximum_position(distanceMap, links_label, range(1,links_label.max()+1))
    links_center=np.transpose(np.squeeze(np.dstack(links_center_arg)))   
    
    linkDiameterDistanceMap=ndimage.measurements.labeled_comprehension(distanceMap, links_label, range(1,links_label.max()+1),max,np.int16,0)    
    
    
    # Dilatation des liens internes
    links=ndimage.morphology.grey_dilation(links_label,  size=(5,5,5))
    
    
    # Infos sur la forme, position des pores
    pores_center=ndimage.measurements.center_of_mass(pores, labels=pores ,index=range(1,pores.max()+1))
    pores_center=np.transpose(np.squeeze(np.dstack(pores_center)))
    pores_volumes=ndimage.measurements.labeled_comprehension(pores, pores, range(1,pores.max()+1),np.size,np.int32,0)
    
    PNMGeometricData=dict()
    PNMGeometricData['imageLiensDilates']=links
    PNMGeometricData['internalLinkDiameters']=linkDiameterDistanceMap
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
        
        links_center_arg=ndimage.measurements.maximum_position(boundaryDistances, boundarySlice, range(1,pores.max()+1))
        links_center=np.transpose(np.squeeze(np.dstack(links_center_arg))) 
        
        #centers=ndimage.measurements.center_of_mass(boundarySlice, labels=boundarySlice ,index=range(1,pores.max()+1))
        #centers=np.transpose(np.squeeze(np.dstack(centers)))
        diameterDistanceMap=ndimage.measurements.labeled_comprehension(boundaryDistances, boundarySlice, range(1,pores.max()+1),max,np.int16,0)
        PNMGeometricData['boundaryCenters'+str(iBoundary)]=centers
        PNMGeometricData['boundaryDiameters'+str(iBoundary)]=diameterDistanceMap
      
      
    return PNMGeometricData


#----------------------------------------------------------------------------------------------
def FindLinks(myImage,pores,watershedLines,structuringElement):
#FindLinks  Trouve les parties de watershed qui sont entre deux pores
#Input: myImage,pores,watershedLines,structuringElement
#Output : links
    
    imageSize=pores.shape
    assert myImage.shape == imageSize
    assert watershedLines.shape == imageSize
        
    #structuring element size
    foo=structuringElement.shape
#    assert(isequal(foo,foo(1)*ones(1,3)) && rem(foo(1),2)==1,'Wrong size of structuring element');
    seSize=int(math.floor(foo[1]/2))
    sideBandes=np.ones((imageSize[0],imageSize[1],imageSize[2]), dtype=bool)
    sideBandes[seSize:-1-seSize,seSize:-1-seSize,seSize:-1-seSize]=False
    watershedLines[sideBandes]=0
   
    links=np.logical_and(watershedLines,np.logical_not(myImage))
    
    indices=links.ravel().nonzero()[0]
   
    for i in range(indices.size):
        ind=indices[i]
        (x,y,z)=np.unravel_index(ind,imageSize)
        localPores=pores[x-seSize:x+seSize+1,y-seSize:y+seSize+1,z-seSize:z+seSize+1]
        if np.setdiff1d(np.unique(localPores[structuringElement]),[0]).size > 2 :
            links[x,y,z]=False

#    booleans=np.zeros(pores.size, dtype=bool)
#
#    booleans[indices[boolfoo]]=True
#    links=np.reshape(booleans,pores.shape)
    assert np.count_nonzero(links[np.logical_not(watershedLines.astype(np.bool))])==0
    #links=np.zeros((imageSize[0],imageSize[1],imageSize[2]),dtype=bool)

    #links[booleans]=True

    return links



if __name__ == "__main__":
    main()