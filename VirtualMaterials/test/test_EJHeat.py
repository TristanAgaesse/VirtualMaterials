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
    

    def test_EJHeat(self):

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
        
#----------------------------------------------------------------------------------------------
if __name__ == '__main__':
    unittest.main()   