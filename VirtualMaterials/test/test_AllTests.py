# -*- coding: utf-8 -*-
"""
Created on Tue Aug 18 16:05:02 2015

@author: ta240184
"""

import unittest


def allTests():
    
    from VirtualMaterials.test.test_PoreNetworkExtraction import test_PoreNetworkExtraction
    from VirtualMaterials.test.test_FullMorphology import test_FullMorphology
    
    
    suite = unittest.TestSuite()
    suite.addTest(unittest.makeSuite(test_PoreNetworkExtraction))
    suite.addTest(unittest.makeSuite(test_FullMorphology))
    
    return suite
    


    
if __name__ == '__main__':
    unittest.TextTestRunner(verbosity=2).run(allTests()) 