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
            
        psi = torch.zeros((self.J, self.L, self.M, self.N), dtype=dtype)    
        for j in range(self.J):
            for l in range(self.L):
                #radial frequency factor
                k0 = frequency_factor * 3.0 / 4.0 * np.pi / 2 ** j 
                theta0 = (int(self.L - self.L/2) -1) * np.pi / self.L 
                
    #morlet 2d           
    def morlet_2d(self, M, N, sigma, theta, xi, slant = 0.5, offset = 0, fft_shift = False):
        '''
        Docstring for morlet_2d
        
        :param self: Description
        :param M: spatial size
        :param N: spatial size
        :param sigma: bandwidth parameter for gaussian envelope
        :param theta: angle [0, pi]
        :param xi: central frequency
        :param slant: parameter which guides ellipsoidal shape of the morlet
        :param offset: offset which parameter starts
        :param fft_shift: shift signal
        
        returns:
        morlet_fft : ndarray size (M,N)
        '''
        
        wav = self.gabor_2d(M, N, sigma, theta, xi, slant, offset, fft_shift)
        wv_modulus = self.gabor_2d(M, N, sigma, theta, 0, slant, offset, fft_shift)
        
        K = wav.sum() / wv_modulus.sum()
        morlet = wav - K * wv_modulus 
        return morlet
        
        
    #gabor 2d
    def gabor_2d(self, M, N, sigma, theta, xi, slant=1.0, offset=0, fft_shift=False):
        '''
        Docstring for gabor_2d
        
        :param self: Description
        :param M: spatial size
        :param N: spatial size
        :param sigma: bandwidth parameter
        :param theta: angle between 0, pi
        :param xi: central frequency
        :param slant: parameter which guides ellipsoidal shape
        :param offset: offset by which signal starts
        :param fft_shift: shift signal in numpy style
        
        returns : morlet fft ndarray size (M,N)
        '''
        R = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]], np.float64)
        R_inv = np.array([[np.cos(theta), np.sin(theta)], [-np.sin(theta), np.cos(theta)]], np.float64)
        #term that streches the gaussian
        D = np.array([[1, 0], [0, slant * slant]])
    
        curv = np.matmul(R, np.matmul(D, R_inv)) / ( 2 * sigma * sigma)
        
        gab = np.zeros((M,N), np.complex128)
        xx = np.empty((2,2, M,N))
        yy = np.empty((2, 2, M, N))
        
        for ii, ex in enumerate([-1, 0]):
            for jj, ey in enumerate([-1, 0]):
                xx[ii, jj], yy[ii, jj] = np.mgrid[
                    offset + ex * M: offset +  M + ex * M,
                    offset + ey * N: offset + N + ey * N
                ]
                
        arg = -(curv[0,0] * xx * xx + (curv[0,1] + curv[1,0])* xx * yy + curv[1,1] * yy * yy) +\
            1.j * (xx * xi * np.cos(theta) + yy * xi * np.sin(theta))
        
        #shape m x n
        gab = np.exp(arg).sum((0,1))
        
        norm_factor = 2 * np.pi * sigma * sigma / slant
        
        gab = gab / norm_factor
        
        if fft_shift:
            gab = np.fft.fftshift(gab, axes=(0,1))
            
        return gab
    
    
                
                