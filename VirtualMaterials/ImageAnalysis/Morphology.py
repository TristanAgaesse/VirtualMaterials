# -*- coding: utf-8 -*-
import SimpleITK as sitk
import numpy as np


#----------------------------------------------------------------------------------------------    
def FastDilation(image,structuringElement):    
    """Custom implementation of morphology dilation
    :param image : numpy image    
    :param: structuringElement
    :return: numpy image 
    
    :Example:    
    import VirtualMaterials as vmat
    structuringElement = np.ones((3,3,3),dtype=np.bool)
    sobelEdges = vmat.ImageAnalysis.Morphology.FastDilation(image,structuringElement)
    """
    structuringElement= np.asarray(structuringElement)
    center = structuringElement.shape[0]//2
    image=image.astype(np.bool)    
    
    biggerImage = np.zeros(np.asarray(image.shape)+2*center,dtype=np.bool)
    dilatedImage = np.zeros(biggerImage.shape,dtype=np.bool) 
    
    dim=image.ndim    
    
    #smallImageIndices=(center:-center-1,center:-center-1,center:-center-1)
#    smallImageIndices = [np.arange(center,center+image.shape[iDim]) 
#                            for iDim in range(dim)]
    
    smallImageIndices = [slice(center,center+image.shape[iDim]) 
                            for iDim in range(dim)]
    
    biggerImage[smallImageIndices] = image
    
    #X,Y,Z = np.nonzero(biggerImage)
    nnzIndices=np.nonzero(biggerImage)
    oneColumn=np.ones(nnzIndices[0].size,dtype=np.int)        
    
    for iSE in range(structuringElement.size):        
        #xIse,yIse,zIse = np.unravel_index(iSE,structuringElement.shape)
        iSEposition = np.unravel_index(iSE,structuringElement.shape)
        #if structuringElement[xIse,yIse,zIse]:
#            shiftX,shiftY,shiftZ = xIse-center,yIse-center,zIse-center
#            dilatedImage[X+shiftX*oneColumn,Y+shiftY*oneColumn,Z+shiftZ*oneColumn]=True
        if structuringElement[iSEposition]:    
            positionTrue=[]
            for iDim in range(dim):
                shift=iSEposition[iDim]-center
                positionTrue.append( nnzIndices[iDim]+shift*oneColumn )
                  
            dilatedImage[positionTrue]=True
                  
    dilatedImage = dilatedImage[smallImageIndices]
    
    
    return dilatedImage
    



    
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
    


    