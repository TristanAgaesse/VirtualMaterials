# -*- coding: utf-8 -*-
import SimpleITK as sitk
import numpy as np
from VirtualMaterials.Utilities import Utilities as utilities    
from collections import defaultdict
import re 


#-----------------------------------------------------------------------------
def SobelEdgeDetection(image):    
    """Uses ITK sobel edge detector to detect boundaries between phases
    :param image : numpy image    
    :return: numpy image of the sobel edges
    
    :Example:    
    import VirtualMaterials as vmat
    sobelEdges = vmat.ImageAnalysis.FeatureExtraction.SobelEdgeDetection(image)
    """
        
    myItkImage = sitk.GetImageFromArray(image.astype(np.uint32))
    caster = sitk.CastImageFilter()
    caster.SetOutputPixelType(sitk.sitkFloat32)
    floatImage = caster.Execute( myItkImage )

    itkEdges = sitk.SobelEdgeDetection(floatImage)
    sobelEdges = sitk.GetArrayFromImage(itkEdges).astype(np.bool)

    del myItkImage,floatImage,itkEdges  

    return sobelEdges


#-----------------------------------------------------------------------------
def CannyEdgeDetection(image,variance=2):       
    """Uses ITK Canny edge detector to detect boundaries between phases. Canny 
    edge detector is the following :
      - Apply Gaussian filter to smooth the image in order to remove the noise
      - Find the intensity gradients of the image
      - Apply non-maximum suppression to get rid of spurious response to edge detection
      - Apply double threshold to determine potential edges
      - Track edge by hysteresis: Finalize the detection of edges by suppressing all the other edges that are weak and not connected to strong edges.
    
    :param image : numpy image  
    :param : variance : width of the gaussian filter  
    :return: numpy image of the sobel edges
    
    :Example:    
    import VirtualMaterials as vmat
    sobelEdges = vmat.ImageAnalysis.FeatureExtraction.CannyEdgeDetection(image,variance=2)
    """

    
    #Load image into SimpleITK
    myItkImage = sitk.GetImageFromArray(image.astype(np.uint8))
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
    cannyEdges = sitk.GetArrayFromImage(myItkImage).astype(np.bool)

    del myItkImage, caster, canny, floatImage  

    return cannyEdges


    
#---------------------------------------------------------------------------------------------- 
def FindContacts(image):
    """Trouve les contacts entre les labels d'une image. Algorithme :
    # - Attribuer à chaque voxel de contours un contact s'il a exactement deux 
    # labels voisins différents.

    :param: image : numpy image avec plusieurs labels
    :return: labeledContacts, contactToLabel
    
    :Example:    
    import VirtualMaterials as vmat
    labeledContacts, contactToLabel = vmat.ImageAnalysis.FeatureExtraction.FindContacts(image)
    """

    
    contours = SobelEdgeDetection(image)


    #structuringElement = np.ones((3,3,3))
    structuringElement = np.asarray(
        [[[0,0,0],[0,1,0],[0,0,0]],
         [[1,1,1],[1,1,1],[1,1,1]],
         [[0,0,0],[0,1,0],[0,0,0]]])
    
    
    imageSize=image.shape
    assert contours.shape == imageSize
    
        #Side bands to avoid unexpected boundary effects
    seSize=int(np.shape(structuringElement)[1]//2)
    sideBandes=np.ones((imageSize[0],imageSize[1],imageSize[2]), dtype=bool)
    sideBandes[seSize:-seSize,seSize:-seSize,seSize:-seSize]=False
    contours[sideBandes]=0
    
    
        #Boucle sur les voxels des contours pour trouver leurs pores voisins
    indices=contours.ravel().nonzero()[0]
    nVox=indices.size
    structuringElement=structuringElement.astype(np.bool)
    linksToPores=[]    
    correctionList=[]


    def StudyNeighbourhood(image,indices,structuringElement,imageSize):  
        X,Y,Z =np.unravel_index(indices,imageSize)
        oneColumn=np.ones(X.shape,dtype=np.int)        
        neighboor=[]
        center = structuringElement.shape[0]//2
        for iSE in range(structuringElement.size):        
            xIse,yIse,zIse = np.unravel_index(iSE,structuringElement.shape)
            if structuringElement[xIse,yIse,zIse]:
                shiftX,shiftY,shiftZ = xIse-center,yIse-center,zIse-center
                neighboor.append( image[X+shiftX*oneColumn,
                                        Y+shiftY*oneColumn,
                                        Z+shiftZ*oneColumn] )
                #TODO:horzcat au lieu de append pour avoir une seule matrice
        U=[set([neighboor[j][i] for j in range(len(neighboor))])  for i in range(nVox)]
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
            
             
    def LabelContacts(mydict,imageSize):
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
    
    
    
    linksToPores,correctionList = StudyNeighbourhood(image,indices,
                                                   structuringElement,imageSize)

    mydict = FillDict(linksToPores)
    
    del linksToPores  
    
    labeledContacts = LabelContacts(mydict,imageSize)
    
    contactToLabel = BuildInterfaceToPore(mydict)
               
    del mydict           
               
    return labeledContacts, contactToLabel   
    

