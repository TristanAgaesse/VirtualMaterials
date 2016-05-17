# -*- coding: utf-8 -*-
"""
Created on Tue Dec  1 18:43:22 2015

@author: greentoto
"""
import numpy as np
import unittest

from VirtualMaterials.VirtualImages import VirtualMaterialsGeneration
from VirtualMaterials.Simulation  import EJHeat

class test_EJHeat(unittest.TestCase):
    

    def test_LayeredMedia(self):

        layers=np.zeros((100,100,100),dtype=np.uint8)
        
        zInds=np.arange(30,60,dtype=np.int)
        
        layers[:,:,zInds]=100
        
        beta=np.zeros((250,),dtype=np.float)
        beta[0] = 1
        beta[100] = 2 
        
        errtol = 1e-6
        maxit = 100
        
        d = 'x'
        temperatureX,B_lX,_,_ = EJHeat.EJ_heat(layers,beta,d,errtol,maxit)
        
        d = 'y'
        temperatureY,B_lY,_,_ = EJHeat.EJ_heat(layers,beta,d,errtol,maxit)
        
        d = 'z'
        temperatureZ,B_lZ,_,_ = EJHeat.EJ_heat(layers,beta,d,errtol,maxit)

        a = beta[0]*0.7+beta[100]*0.3
        b = 1/(0.7*1/beta[0]+0.3*1/beta[100])
        self.assertAlmostEqual(B_lX[0],a)
        self.assertAlmostEqual(B_lY[1],a)
        self.assertAlmostEqual(B_lZ[2],b)
        
        
    def test_CylinderArray(self): 
        #Test error between Ej-Heat and Rayleigh formula for a cylinder array
        
#        cylinderRadiusList=[2,5,10,20,30,40]
#        k1List=[0.001,0.01,0.1,0.9,10,100,1000]
        cylinderRadiusList = [10]
        k1List= [0.01]
        
        simu = [[] for i in range(len(cylinderRadiusList))]
        analytical = [[] for i in range(len(cylinderRadiusList))]
        relError = [[] for i in range(len(cylinderRadiusList))]
        matrix= 0 
        cylinders= 255
        
        for i in range(len(cylinderRadiusList)):
            
            cylinderRadius = cylinderRadiusList[i]
            image=VirtualMaterialsGeneration.CreateTestImage_CylinderArray(
                        cylinderRadius = cylinderRadius,voxelNumberParam=0.5 )
            
        
            simu[i]=[]
            analytical[i]=[]
            relError[i]=[] 
            for j in range(len(k1List)):
                km=1
                k1=k1List[j]
                
                beta = np.zeros((256,),dtype=np.float)
                beta[matrix]=km
                beta[cylinders]=k1
                direction='x'
                temperature,B_l,RR,I = EJHeat.EJ_heat(image,beta,direction,1e-6,100)
                 
                phi = np.count_nonzero(image==cylinders)/float(image.size)
                C1 = (k1+km)/float(k1-km)
                C2 = (k1-km)/float(k1+km)
                
                formula = 1 + 2*phi/(C1-phi+C2*(0.30584*phi**4+0.013363*phi**8))
                
                simu[i].append(B_l[0])
                analytical[i].append(formula)
                relError[i].append((abs(B_l[0]-formula)/formula))
        
            self.assertLess(relError[0][0],0.1)
#        print(simu)
#        print(analytical)
#        print(relError)
#        
#        k=k1List
#        rel=relError
        
#        #Plot relative error
#        plt.figure(0)
#        plt.plot(k,rel[0],'c--',label='Phi = %f' %phi[0])
#        plt.plot(k,rel[1],'m--',label='Phi = %f' %phi[1])
#        plt.plot(k,rel[2],'k--',label='Phi = %f' %phi[2])
#        plt.plot(k,rel[3],'r--',label='Phi = %f' %phi[3])
#        plt.plot(k,rel[4],'b--',label='Phi = %f' %phi[4])
#        plt.plot(k,rel[5],'g--',label='Phi = %f' %phi[5])
#        
#        plt.xscale('log')
#        plt.xlabel('k1/km')
#        plt.yscale('log')
#        plt.ylabel('Relative error between Ej-Heat and Rayleigh formula')
#        plt.legend(loc='upper center')
#        plt.title('Ej-Heat test case : cylinder array')
#        plt.show()
#        
#        #Plot conductivity values
#        plt.figure(1)
#        plt.plot(k,simu[0],'c--',label='Phi = %f' %phi[0])
#        plt.plot(k,simu[1],'m--',label='Phi = %f' %phi[1])
#        plt.plot(k,simu[2],'k--',label='Phi = %f' %phi[2])
#        plt.plot(k,simu[3],'r--',label='Phi = %f' %phi[3])
#        plt.plot(k,simu[4],'b--',label='Phi = %f' %phi[4])
#        plt.plot(k,simu[5],'g--',label='Phi = %f' %phi[5])
#        
#        plt.xscale('log')
#        plt.xlabel('k1/km')
#        plt.yscale('log')
#        plt.ylabel('Ej-Heat simulation')
#        plt.legend(loc='upper center')
#        plt.title('Ej-Heat test case : cylinder array')
#        plt.show()                                                                                      
        
        
        
        
#----------------------------------------------------------------------------------------------
if __name__ == '__main__':
    unittest.main()   