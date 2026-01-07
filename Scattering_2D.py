import numpy as np 
import torch
import torch.fft as fft 
from FilterSet import Filter_Set

class ScatteringTransform:
    '''
    Docstring for ScatteringTransform
    2D scattering transform
    '''
    def __init__(self, M, N, J=None, L=4, order=2, precision="single"):
        
        '''
        Docstring for __init__
        
        :param self: Description
        :param M: spatial dimension
        :param N: Spatial dimension
        :param J: Number of Scales
        :param L: number of orientations
        :param order: Max scattering order
        :param precision: 'single' or 'double'
        '''
        
        self.M = M
        self.N = N 
        self.J = J if J is not None else int(np.log2(min(M,N))) - 1
        self.order = order
        self.precision = precision 
        
        #generate filters, wavelet and averaging
        filter_gen = Filter_Set(M, N, J, L)
        self.filters = filter_gen.generate_wavelets(precision=precision)
        
        self.dtype = torch.float32 if precision == 'single' else torch.float64
        self.cdtype = torch.complex64 if precision == 'single' else torch.complex128
        
        
    def forward(self, x):
        '''
        Docstring for forward
        
        :param self: Description
        :param x: Input signal
        '''
        #handle batched input
        if x.dim() == 2:
            x = x.unsqueeze(0)
            squeeze_output = True 
        else: 
            squeeze_output = False 
            
        batch_size = x.shape[0]
        
        #convert to correct dtype
        x = x.to(self.dtype)
        
        #compute FFT of input
        x_fft = fft.fft2(x)
        
        #storage for scattering coefficients
        S0 = []
        S1 = []
        S2 = []
        
        #zeroth order
        
        
    def _lowpass(self, x_fft):
        '''
        Docstring for _lowpass
        
        :param self: Description
        :param x_fft: Input signal in frequency domain
        '''
        phi = self.filters['phi'].to(x_fft.device)
        phi_complex = phi.to(self.cdtype)
        
        #multipy input signal fourier domain
        y_fft = x_fft * phi_complex
        
        #iinverse fft and take real part
        y = fft.ifft2(y_fft).real 
        
        return y
        
            
            