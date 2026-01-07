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
        S0_coefficients = self._lowpass(x_fft)
        S0.append(S0_coefficients)
        
        
        #u1
        U1_list = []
        #compute first order coefficients
        for j1 in range(self.J):
            for l1 in range(self.L):
                #convolve with wavelet psi[j1, l1]
                U1 = self._wavelet_transform(x_fft, j1, l1)
                
                #apply modulus to the coefficients
                U1_mod = torch.abs(U1)
                U1_mod_fft = fft.fft2(U1_mod)
                
                #pass through low pass
                S1_coeff = self._lowpass(U1_mod_fft)
                
                S1.append(S1_coeff)
                
                #store for second order
                U1_list.append((U1_mod_fft, j1, l1))
                
        for u1_mod, j1, l1 in U1_list:
            for j2 in range(j1 + 1, self.J):
                for l2 in range(self.L):
                    #convolve with wavelet
                    U2 = self._wavelet_transform(U1_mod_fft, j2, l2)
                    
                    #apply modulus
                    U2_mod = torch.abs(U2)
                    U2_mod_fft = fft.fft2(U2_mod)
                    
                    #lowpass filter
                    S2_coeff = self._lowpass(U2_mod_fft)
                    S2.append(S2_coeff)
                    
        #stack coefficients
        result = {
            'S0': torch.stack(S0, dim=1),
            'S1': torch.stack(S1, dim=1),
            'S2': torch.stack(S2, dim=1)
        }
        
        return result
                
                
        
        
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
        
            
    def _wavelet_transform(self, x_fft, j, l):
        '''
        Docstring for _wavelet_transform
        
        :param self: Description
        :param x_fft: innput signal
        :param j: scale
        :param l: orientation
        '''
        #select this specific filter
        psi = self.filters['psi'][j, l].to(x_fft.device)
        psi_complex = psi.to(self.cdtype)
        
        #multiply signal in fourier domain
        y_fft = x_fft * psi_complex
        #inverse fft
        y = fft.ifft2(y_fft)
        
        return y