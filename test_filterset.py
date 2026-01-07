import unittest
import torch 
import numpy as np 
from FilterSet import Filter_Set

class TestFilterSet(unittest.TestCase):
    
    def test_shapes_and_values(self):
        
        M, N = 16, 16
        designed_filter = Filter_Set(M, N, 2, 2)
        
        my_filterset = designed_filter.generate_wavelets(
            wavelets='morlet', precision='single'
        )
        
        psi = my_filterset['psi']
        phi = my_filterset['phi']
        
        #check shape
        self.assertEqual(psi.shape, (2,2, M,N), 'incorrect psi')
        self.assertEqual(phi.shape, (M,N), "phi shape is incorrect")
        
        
        
if __name__ == "__main__":
    unittest.main()