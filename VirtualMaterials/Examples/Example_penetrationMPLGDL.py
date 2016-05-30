# -*- coding: utf-8 -*-
import VirtualMaterials as vmat
import numpy as np
import hdf5storage
import matplotlib.pyplot as plt

"""Etude des interfaces entre MPL et GDL 
On génère des structures virtuelles d'assemblage MPL et GDL. La MPL pénètre 
dans la GDL avec un algorithme de FullMorphology.
On calcule leurs propriétés. On extrait l'effet de la pénétration.
"""


def CreateGDLwithMPL(penetrationLength=10,randomSeed=1,
                                 anisotropy=10):
                              
    #voxelNumbers=(100,100,50)
    voxelNumbers=(200,200,200)                      
    image = vmat.VirtualImages.VirtualMaterialsGeneration.CreateVirtualInterfaceGDLMPL(
          voxelNumbers=voxelNumbers,
          penetrationLength=penetrationLength,randomSeed=randomSeed,
                                 anisotropy=anisotropy)
          
    #Elever la MPL en trop pour garder une fraction de MPL constante      
    phaseFraction = vmat.ImageAnalysis.QuantifyGeometry.VolumeFraction(image)      
    mplCode = 3
    mplFraction = phaseFraction[mplCode]
    mplFractionRequired = 0.5
    
    nVoxelsTooMuch = int( (mplFraction-mplFractionRequired)/float(1-mplFractionRequired)*voxelNumbers[2] )    
    image = image[:,:,0:-1-nVoxelsTooMuch]          
          
    phaseFraction = vmat.ImageAnalysis.QuantifyGeometry.VolumeFraction(image)
    mplFraction = phaseFraction[mplCode]     
    imThickness=image.shape[2]
    print('MPL fraction : %f, imageThickness: %d' %(mplFraction,imThickness) )     
    return image



def SimuEffectiveProperties(numpyInputImage,beta):
   
    errtol = 1e-6
    maxit = 100
    
    tensor=[]
  
    d = 'x'
    numpyOutputImage,effectiveProp,_,_ = vmat.Simulation.EJHeat.EJ_heat(
    numpyInputImage,beta,d,errtol,maxit)
    tensor.append(effectiveProp)
  
    d = 'y'
    numpyOutputImage,effectiveProp,_,_ = vmat.Simulation.EJHeat.EJ_heat(
    numpyInputImage,beta,d,errtol,maxit)
    tensor.append(effectiveProp)
  
    d = 'z'
    numpyOutputImage,effectiveProp,_,_ = vmat.Simulation.EJHeat.EJ_heat(
    numpyInputImage,beta,d,errtol,maxit)
    tensor.append(effectiveProp)

    tensor=np.asarray(tensor)
    return tensor

def GetBeta(image,caseName):
    
    beta=np.zeros((2**16,),dtype=np.float)
    
    void=0
    binder = 1
    fibers = 2
    mpl = 3
    
    if caseName=='Diffusion' :    
        beta[void] = 1
        beta[binder] = 0.01 
        beta[fibers] = 0
        beta[mpl] = 0.01
        
    elif caseName=='Thermic' :    
        beta[void] = 0.001
        beta[binder] = 0.01 
        beta[fibers] = 1
        beta[mpl] = 0.01
        
        
    elif caseName=='Electric' :    
        beta[void] = 0
        beta[binder] = 0.01 
        beta[fibers] = 1
        beta[mpl] = 0.01   
        
    return beta

def PostTraitement(effProperties):
    
    penetrationLengthList = effProperties['penetrationLengthList']
    diff_IP = effProperties['diff_IP']
    diff_TP = effProperties['diff_TP']
    thermic_IP = effProperties['thermic_IP']
    thermic_TP = effProperties['thermic_TP']
    elec_IP = effProperties['elec_IP']
    elec_TP = effProperties['elec_TP']
    
    plList=penetrationLengthList    
    #Plots    
    plt.figure(0)    
    plt.plot(plList,diff_IP,label='Diffusion in-plane')
    plt.plot(plList,diff_TP,label='Diffusion through-plane') 
    plt.plot(plList,thermic_IP,label='Thermic in-plane')
    plt.plot(plList,thermic_TP,label='Thermic through-plane')
    plt.plot(plList,elec_IP,label='Electric in-plane')
    plt.plot(plList,elec_TP,label='Electric through-plane')
    
    plt.xlabel('PenetrationLength')
    plt.ylabel('Effective properties (adimensioned)')
    plt.title('Effective properties of MPL+GDL structures')
    plt.legend(bbox_to_anchor=(1, 1),bbox_transform=plt.gcf().transFigure)
    plt.show()  

def main(penetrationLengthList=[10],randomSeed=randomSeed,anisotropy=anisotropy):
    folder = '/home/270.12-Modeling_PEMFC_Li/theseTristan/ImageBasedSimulations/PenetrationMPL-GDL/'
    
    diff_IP = []
    diff_TP = []
    thermic_IP = []
    thermic_TP = []
    elec_IP = []
    elec_TP = []
    
    for penetrationLength in penetrationLengthList:
        print("PenetrationLength %d" %penetrationLength)
        image = CreateGDLwithMPL(penetrationLength=penetrationLength,
                            randomSeed=randomSeed,anisotropy=anisotropy)
        
        
        filename = folder+"GDL_200-200-200_randomSeed%d_anisotropy%d_penetration%d.tiff" %(penetrationLength,randomSeed, anisotropy)   
        vmat.Utilities.Utilities.WriteTiff(image,filename)
        
        #--------- Diffusion --------------------
        caseName='Diffusion'
        print('Cas %s' %caseName)
        beta = GetBeta(image,caseName)
        tensor = SimuEffectiveProperties(image,beta)
        inPlane = (tensor[0,0]+tensor[1,1])/2
        throughPlane= tensor[2,2]
        diff_IP.append(inPlane)
        diff_TP.append(throughPlane)
        
        #--------- Thermique --------------------
        caseName='Thermic'
        print('Cas %s' %caseName)
        beta = GetBeta(image,caseName)
        tensor = SimuEffectiveProperties(image,beta)
        inPlane = (tensor[0,0]+tensor[1,1])/2
        throughPlane= tensor[2,2]
        thermic_IP.append(inPlane)
        thermic_TP.append(throughPlane)
        
        #--------- Electrique--------------------
        caseName='Electric'
        print('Cas %s' %caseName)
        beta = GetBeta(image,caseName)
        tensor = SimuEffectiveProperties(image,beta)
        inPlane = (tensor[0,0]+tensor[1,1])/2
        throughPlane= tensor[2,2]
        elec_IP.append(inPlane)
        elec_TP.append(throughPlane)
        
        
    filename=folder+'effectiveProps_200-200-200_randomSeed%d_anisotropy%d_penetration%d.mat' %(randomSeed, anisotropy) 
    effProperties = {'penetrationLengthList':penetrationLengthList,
            'diff_IP':diff_IP,'diff_TP':diff_TP,
            'thermic_IP':thermic_IP,'thermic_TP':thermic_TP,
            'elec_IP':elec_IP,'elec_TP':elec_TP}
    hdf5storage.savemat(filename,effProperties)    
        
    #------PostTraitement------------------------    
    PostTraitement(effProperties)
      
        
        
#function which is called when this file is executed by python        
if __name__== "__main__":
    penetrationLengthList=[10,12,14,16,18,20,22,24,70]
    print(penetrationLengthList)
    main(penetrationLengthList=penetrationLengthList,randomSeed=1,anisotropy=10)
    