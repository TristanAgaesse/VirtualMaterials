# -*- coding: utf-8 -*-
#%Test de la formule de la constrictivité sur un réseau de pore régulier
#%formé de pores ayant tous la même taille. 
#% Paramètres du réseau :
#%   a : pas du réseau
#%   D : diamètre des pores
#%   k*D : diamètre des liens
#%   u=D/a, paramètre adimensionné.

import matplotlib.pyplot as plt
import numpy as np

#-----------------Gaz phase : formula-----------------

#Geometric parameters
epsilonGaz = lambda u, k: u**2*k**2+u**3*(1-k**2)    #volume fraction
tortuosityGaz = 1
constrictivityGaz = lambda u, k: k**2     # constrictivité

# Constrictivity equation
mGaz=lambda u, k: 2.03*(epsilonGaz(u,k))**1.57*(constrictivityGaz(u,k))**0.57/tortuosityGaz**2
# Formule pour la conductivité effective : calcul à la main possible sur la
# géométrie très simple
deffGaz= lambda u, k:  1/(((1/u)*(1-1/k**2)+(1/u**2)*(1/k**2)))

#-----------------Solide phase : formula-----------------

# geometric parameters
epsilonSolide = lambda u, k: 1-epsilonGaz(u,k)    #volume fraction
tortuositySolide = 1
constrictivitySolide = lambda u, k: (1-u)*(1+u*(2*k-1))/(1-(k*u)**2)#((1-u)/(1-k*u))**2  #    # constrictivité

# Constrictivity equation
mSolide=lambda u, k: 2.03*(epsilonSolide(u,k))**1.57*(constrictivitySolide(u,k))**0.57/tortuositySolide**2
# Formule pour la conductivité effective : calcul à la main possible sur la
# géométrie très simple
deffSolide= lambda u, k:  1/((1-u)/(1-(k*u)**2)+k/(-2*k+u*(2*k-1)+1/u)+(1-k)/(-u+1/u))


#-----------------  Gaz phase plots   -----------------

#Comparing m and deffGaz, which are two expressions for effective diffusivity
plt.figure(0)
nPoints = 9 
uArray = np.linspace(0.1,0.9,nPoints) 
kArray = 0.5*np.ones(nPoints) 

deffGazArray=np.zeros(nPoints)
for i in range(9):
    deffGazArray[i]=deffGaz(uArray[i],kArray[i]) 

mGazArray=np.zeros(nPoints)
for i in range(9):
    mGazArray[i]=mGaz(uArray[i],kArray[i])

plt.plot(uArray,deffGazArray,label='Real value (analytic formula)')
plt.plot(uArray,mGazArray,label='Constrictivity equation') 
plt.xlabel('u=d/a')
plt.ylabel('Diffusion coefficient (adimensioned)')
plt.title('Constrictivity equation on a simple structured pore network - gaz phase')
plt.legend(bbox_to_anchor=(1, 1),bbox_transform=plt.gcf().transFigure)
plt.show()


#Error analysis
plt.figure(1)
ratio=[]
for j in range(1,10):
    kArray = j/10.0*np.ones(nPoints) 
    deffGazArray=np.zeros(nPoints)
    mGazArray=np.zeros(nPoints)
    for i in range(9):
        deffGazArray[i]=deffGaz(uArray[i],kArray[i]) 
    for i in range(9):
        mGazArray[i]=mGaz(uArray[i],kArray[i])
    ratio.append(mGazArray/deffGazArray)

for j in range(1,10): 
    plt.plot(uArray,ratio[j-1],label='Ratio for k=0.%d' %j)
    
plt.xlabel('u=d/a')
plt.ylabel('Ratio : constrictivity equation over real value')
plt.title('Constrictivity equation on a simple structured pore network - gaz phase')
plt.legend(bbox_to_anchor=(1, 1),bbox_transform=plt.gcf().transFigure)
plt.show()


#Correlation between deffGaz and porosity
plt.figure(2)
nPoints = 20 
uArray = np.linspace(0.1,0.9,nPoints) 
kArray = np.linspace(0.1,0.9,nPoints) 

deffSolideArray=np.zeros(nPoints*nPoints)
epsilonSolideArray = np.zeros(nPoints*nPoints)
for i in range(nPoints):
    for j in range(nPoints):
        deffSolideArray[i*nPoints+j]=deffSolide(uArray[i],kArray[j])
        epsilonSolideArray[i*nPoints+j]=epsilonSolide(uArray[i],kArray[j])    

plt.scatter(epsilonSolideArray,deffSolideArray)    
plt.xlabel('Porosity')
plt.ylabel('Diffusion coefficient (normalized)')
plt.title('Correlation between diffusion and porosity - solide phase')
#plt.legend(bbox_to_anchor=(1, 1),bbox_transform=plt.gcf().transFigure)
plt.show()        


#--------------------Solide phase plots---------------------

#Comparing m and deffSolide, which are two expressions for effective diffusivity
plt.figure(3)
nPoints = 9 
uArray = np.linspace(0.1,0.9,nPoints) 
kArray = 0.5*np.ones(nPoints) 

deffSolideArray=np.zeros(nPoints)
for i in range(9):
    deffSolideArray[i]=deffSolide(uArray[i],kArray[i]) 

mSolideArray=np.zeros(nPoints)
for i in range(9):
    mSolideArray[i]=mSolide(uArray[i],kArray[i])

plt.plot(uArray,deffSolideArray,label='Real value (analytic formula)')
plt.plot(uArray,mSolideArray,label='Constrictivity equation') 
plt.xlabel('u=d/a')
plt.ylabel('Diffusion coefficient (adimensioned)')
plt.title('Constrictivity equation on a simple structured pore network - solide phase')
plt.legend(bbox_to_anchor=(1, 1),bbox_transform=plt.gcf().transFigure)
plt.show()


#Error analysis
plt.figure(4)
ratio=[]
for j in range(1,10):
    kArray = j/10.0*np.ones(nPoints) 
    deffSolideArray=np.zeros(nPoints)
    mSolideArray=np.zeros(nPoints)
    for i in range(9):
        deffSolideArray[i]=deffSolide(uArray[i],kArray[i]) 
    for i in range(9):
        mSolideArray[i]=mSolide(uArray[i],kArray[i])
    ratio.append(mSolideArray/deffSolideArray)

for j in range(1,10): 
    plt.plot(uArray,ratio[j-1],label='Ratio for k=0.%d' %j)
    
plt.xlabel('u=d/a')
plt.ylabel('Ratio : constrictivity equation over real value')
plt.title('Constrictivity equation on a simple structured pore network - solide phase')
plt.legend(bbox_to_anchor=(1, 1),bbox_transform=plt.gcf().transFigure)
plt.show()


#Correlation between deffGaz and porosity
plt.figure(5)
nPoints = 20 
uArray = np.linspace(0.1,0.9,nPoints) 
kArray = np.linspace(0.1,0.9,nPoints) 

deffSolideArray=np.zeros(nPoints*nPoints)
epsilonSolideArray = np.zeros(nPoints*nPoints)
for i in range(nPoints):
    for j in range(nPoints):
        deffSolideArray[i*nPoints+j]=deffSolide(uArray[i],kArray[j])
        epsilonSolideArray[i*nPoints+j]=epsilonSolide(uArray[i],kArray[j])    

plt.scatter(epsilonSolideArray,deffSolideArray)    
plt.xlabel('Porosity')
plt.ylabel('Diffusion coefficient (normalized)')
plt.title('Correlation between diffusion and porosity - solide phase')
#plt.legend(bbox_to_anchor=(1, 1),bbox_transform=plt.gcf().transFigure)
plt.show()        



#--------------------Correlation solide gaz---------------------

def pareto_frontier(Xs, Ys, maxX = True, maxY = True):
    '''
    Method to take two equally-sized lists and return just the elements which lie 
    on the Pareto frontier, sorted into order.
    Default behaviour is to find the maximum for both X and Y, but the option is
    available to specify maxX = False or maxY = False to find the minimum for either
    or both of the parameters.
    '''      
    # Sort the list in either ascending or descending order of X
    myList = sorted([[Xs[i], Ys[i]] for i in range(len(Xs))], reverse=maxX)
    # Start the Pareto frontier with the first value in the sorted list
    p_front = [myList[0]]    
    # Loop through the sorted list
    for pair in myList[1:]:
        if maxY: 
            if pair[1] >= p_front[-1][1]: # Look for higher values of Y…
                p_front.append(pair) # … and add them to the Pareto frontier
        else:
            if pair[1] <= p_front[-1][1]: # Look for lower values of Y…
                p_front.append(pair) # … and add them to the Pareto frontier
    # Turn resulting pairs back into a list of Xs and Ys
    p_frontX = [pair[0] for pair in p_front]
    p_frontY = [pair[1] for pair in p_front]
    return p_frontX, p_frontY

#Correlation between deffGaz and deffSolide
plt.figure(6)
nPoints = 100 
uArray = np.linspace(0.05,0.95,nPoints) 
kArray = np.linspace(0.05,0.95,nPoints) 

deffSolideArray=np.zeros(nPoints*nPoints)
deffGazArray = np.zeros(nPoints*nPoints)
uArray_SolidGazCorrel = np.zeros(nPoints*nPoints)
kArray_SolidGazCorrel = np.zeros(nPoints*nPoints)
for i in range(nPoints):
    for j in range(nPoints):
        deffSolideArray[i*nPoints+j]=deffSolide(uArray[i],kArray[j])
        deffGazArray[i*nPoints+j]=deffGaz(uArray[i],kArray[j])
        uArray_SolidGazCorrel[i*nPoints+j] = uArray[i]
        kArray_SolidGazCorrel[i*nPoints+j] = kArray[j]

p_front = pareto_frontier(deffGazArray, deffSolideArray, maxX = True, maxY = True)

plt.scatter(deffGazArray,deffSolideArray) 
plt.plot(p_front[0],p_front[1],'r')   
plt.xlabel('Diffusion coefficient of pore phase (normalized)')
plt.ylabel('Conduction coefficient of solid phase (normalized)')
plt.title('Correlation between pore diffusion and solid conduction')
#plt.legend(bbox_to_anchor=(1, 1),bbox_transform=plt.gcf().transFigure)
plt.show()     




#Parallel coordinates plot to see Pareto optimal structures
def ismember(B,A):
    B_unique_sorted, B_idx = np.unique(B, return_index=True)
    B_in_A_bool = np.in1d(B_unique_sorted, A, assume_unique=True)
    commonVals=B_unique_sorted[B_in_A_bool]
    indicesOfCommonVals = B_idx[B_in_A_bool]
    return commonVals,indicesOfCommonVals


isOptimal=ismember(deffSolideArray,p_front[1])
optimalU = uArray_SolidGazCorrel[isOptimal[1]]
optimalK = kArray_SolidGazCorrel[isOptimal[1]]
plt.figure(7)
plt.scatter(optimalU,optimalK)   
plt.xlabel('u')
plt.ylabel('k')
plt.title('Optimal structures parameters')
#plt.scatter(constrictivityGaz(optimalU,optimalK),constrictivitySolide(optimalU,optimalK))
#plt.scatter(epsilonGaz(optimalU,optimalK),epsilonSolide(optimalU,optimalK))

import pandas
from pandas.tools.plotting import parallel_coordinates

mydict={}
mydict['Pore Diffusion']=deffGazArray
mydict['Solid Conductivity']=deffSolideArray
mydict['u']=uArray_SolidGazCorrel
mydict['k']=kArray_SolidGazCorrel
mydict['Gas constrictivity']=constrictivityGaz(uArray_SolidGazCorrel,kArray_SolidGazCorrel)
mydict['Solid constrictivity']=constrictivitySolide(uArray_SolidGazCorrel,kArray_SolidGazCorrel)
paretoOptimal=np.zeros(nPoints*nPoints,dtype=np.bool)
paretoOptimal[isOptimal[1]]=True #['Yes' else 'No' for i in nPoints if isOptimal[1]==1 ]
mydict['Pareto optimal']=paretoOptimal

columnName = ['Pore Diffusion','Gas constrictivity','Solid Conductivity','Solid constrictivity','u','k','Pareto optimal']  #mydict.keys()
myDataFrame = pandas.DataFrame(mydict,columns=columnName)
#plt.figure(8)
#parallel_coordinates(myDataFrame, 'Pareto optimal')


# Cas de réseaux désordonnés formés de cellules telles que celles calculées précédemment
# Moyennes harmoniques et arithmétiques : bornes de Voigt et Reuss

nK=nPoints
arithmSolid = np.zeros(nK)
harmSolid = np.zeros(nK)
arithmGaz = np.zeros(nK)
harmGaz = np.zeros(nK)
for ik in range(0,nK):
    kFilter = kArray_SolidGazCorrel==kArray[ik]
    arithmSolid[ik] = np.mean(deffSolideArray[kFilter])
    harmSolid[ik] = (np.mean(deffSolideArray[kFilter]**(-1)))**(-1)
    arithmGaz[ik] = np.mean(deffGazArray[kFilter])
    harmGaz[ik] = (np.mean(deffGazArray[kFilter]**(-1)))**(-1)

plt.figure()
plt.plot(kArray,arithmSolid,'r',label='Solid - arithmetic bound')   
plt.plot(kArray,harmSolid,'b',label='Solid - harmonic bound') 
plt.plot(kArray,arithmGaz,'k',label='Gas - arithmetic bound') 
plt.plot(kArray,harmGaz,'g',label='Gas - harmonic bound') 
plt.xlabel('k')
plt.ylabel('Effective property')
plt.title('Voigt and Reuss bounds for effective properties of disordered cubic network')
plt.legend(bbox_to_anchor=(1, 1),bbox_transform=plt.gcf().transFigure)
plt.show()     

# Cas de réseaux désordonnés formés de cellules telles que celles calculées précédemment
# Moyennes harmoniques et arithmétiques : bornes de Voigt et Reuss

nK=nPoints
arithmSolid = np.zeros(nK)
harmSolid = np.zeros(nK)
arithmGaz = np.zeros(nK)
harmGaz = np.zeros(nK)
wShape = 1.5 #https://en.wikipedia.org/wiki/Weibull_distribution
wScale = 0.4 #https://en.wikipedia.org/wiki/Weibull_distribution
wU=uArray_SolidGazCorrel/wScale
weibull = wShape/wScale*np.power(wU,wShape-1)*np.exp(-np.power(wU,wShape))    
for ik in range(0,nK):
    kFilter = kArray_SolidGazCorrel==kArray[ik]
    #uFilter = np.logical_and(uArray_SolidGazCorrel>0.01,uArray_SolidGazCorrel<0.8)    
    #paramFilter = np.logical_and(kFilter,uFilter)
    paramFilter=kFilter
    weigth=np.mean(weibull[paramFilter])
    arithmSolid[ik] = np.mean(weibull[paramFilter]*deffSolideArray[paramFilter])/weigth
    harmSolid[ik] = (np.mean(weibull[paramFilter]*(deffSolideArray[paramFilter]**(-1)))/weigth)**(-1)
    arithmGaz[ik] = np.mean(weibull[paramFilter]*deffGazArray[paramFilter])/weigth
    harmGaz[ik] = (np.mean(weibull[paramFilter]*(deffGazArray[paramFilter]**(-1)))/weigth)**(-1)

plt.figure()
plt.plot(kArray,arithmSolid,'r',label='Solid - arithmetic bound')   
plt.plot(kArray,harmSolid,'b',label='Solid - harmonic bound') 
plt.plot(kArray,arithmGaz,'k',label='Gas - arithmetic bound') 
plt.plot(kArray,harmGaz,'g',label='Gas - harmonic bound') 
plt.xlabel('k')
plt.ylabel('Effective property')
plt.title('Voigt and Reuss bounds for effective properties of disordered cubic network with custom pore size distribution')
plt.legend(bbox_to_anchor=(1, 1),bbox_transform=plt.gcf().transFigure)
plt.show()  

#Weibull distribution
plt.figure()
xArray=np.linspace(0,1,30)
wU=xArray/wScale
weibull = wShape/wScale*np.power(wU,wShape-1)*np.exp(-np.power(wU,wShape))
weigth=np.mean(weibull)
weibull=weibull/weigth 
plt.plot(xArray,weibull)
plt.xlabel('Pore size parameter u')
plt.ylabel('Density of probability')
plt.title('Pore size distribution')
plt.show()
