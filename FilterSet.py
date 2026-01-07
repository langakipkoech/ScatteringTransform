import numpy as np
import torch
import torch.fft 


class FilterSet:
    
    def __init__(self, M, N, J=None, L=4):
        
        if J is None:
            J = int(np.log2(min(M,N))) - 1
        self.M = M 
        self.N = N
        self.J = J
        self.L = L
        
    def generate_wavelets(
        self, if_save = False, save_dir = None, wavelets="morlet", precision="single", l_oversampling = 1, frequency_factor = 1
    ):
        
        #morlet wavelet
        if precision == "single":
            dtype = torch.float32
            dtype_np = np.float32
            
        if precision == "double":
            dtype = torch.float64
            dtype_np = np.float64
            
        for j in range(self.J):
            for l in range(self.L):
                