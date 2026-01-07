import numpy as np
import torch
import torch.fft 


class Filter_Set:
    
    def __init__(self, M, N, J=None, L=4):
        
        if J is None:
            J = int(np.log2(min(M,N))) - 1
        self.M = M 
        self.N = N
        self.J = J
        self.L = L
        
    def generate_wavelets(
        self, if_save=False, save_dir=None, wavelets="morlet", 
        precision="single", l_oversampling=1, frequency_factor=1
    ):
        
        # Set data types
        if precision == "single":
            dtype = torch.float32
            cdtype = torch.complex64
            dtype_np = np.float32
            cdtype_np = np.complex64
            
        elif precision == "double":
            dtype = torch.float64
            cdtype = torch.complex128
            dtype_np = np.float64
            cdtype_np = np.complex128
        else:
            raise ValueError(f"precision must be 'single' or 'double', got {precision}")
            
        # Initialize wavelet array (complex-valued in Fourier domain)
        psi = torch.zeros((self.J, self.L, self.M, self.N), dtype=cdtype)
        
        for j in range(self.J):
            for l in range(self.L):
                # Radial frequency factor
                k0 = frequency_factor * 3.0 / 4.0 * np.pi / 2 ** j
                
                # FIXED: Orientation angle should depend on l
                theta0 = l * np.pi / self.L
                
                if wavelets == "morlet":
                    wavelet_spatial = self.morlet_2d(
                        M=self.M, N=self.N, 
                        sigma=0.8 * 2 ** j / frequency_factor, 
                        theta=theta0, 
                        xi=k0, 
                        slant=4.0 / self.L * l_oversampling
                    )
                    wavelet_fourier = np.fft.fft2(wavelet_spatial)
                    
                # Remove DC component
                wavelet_fourier[0, 0] = 0
                
                # FIXED: Store complex values, not just real part
                psi[j, l] = torch.from_numpy(wavelet_fourier.astype(cdtype_np))
        
        # Generate lowpass filter (scaling function)
        if wavelets == "morlet":
            # FIXED: Use self.J explicitly instead of loop variable j
            phi_spatial = self.gabor_2d(
                M=self.M, N=self.N, 
                sigma=0.8 * 2 ** self.J / frequency_factor, 
                theta=0, 
                xi=0
            )
            # FIXED: Store complex values
            phi = torch.from_numpy(phi_spatial.astype(cdtype_np)) * (self.M * self.N) ** 0.5
        else:
            raise ValueError(f"Wavelet type '{wavelets}' not implemented")
            
        filters_set = {'psi': psi, 'phi': phi}
        
        # Optional: save filters if requested
        if if_save and save_dir is not None:
            import os
            os.makedirs(save_dir, exist_ok=True)
            torch.save(filters_set, os.path.join(save_dir, 'filters.pt'))
        
        return filters_set
    
    def morlet_2d(self, M, N, sigma, theta, xi, slant=0.5, offset=0, fft_shift=False):
        '''
        Generate 2D Morlet wavelet in spatial domain.
        
        :param M: spatial size (height)
        :param N: spatial size (width)
        :param sigma: bandwidth parameter for gaussian envelope
        :param theta: angle [0, pi]
        :param xi: central frequency
        :param slant: parameter which guides ellipsoidal shape of the morlet
        :param offset: offset which parameter starts
        :param fft_shift: shift signal
        
        returns:
        morlet : ndarray size (M,N), complex-valued
        '''
        
        wav = self.gabor_2d(M, N, sigma, theta, xi, slant, offset, fft_shift)
        wv_modulus = self.gabor_2d(M, N, sigma, theta, 0, slant, offset, fft_shift)
        
        K = wav.sum() / wv_modulus.sum()
        morlet = wav - K * wv_modulus 
        return morlet
        
    def gabor_2d(self, M, N, sigma, theta, xi, slant=1.0, offset=0, fft_shift=False):
        '''
        Generate 2D Gabor function in spatial domain.
        
        :param M: spatial size (height)
        :param N: spatial size (width)
        :param sigma: bandwidth parameter
        :param theta: angle between 0, pi
        :param xi: central frequency
        :param slant: parameter which guides ellipsoidal shape
        :param offset: offset by which signal starts
        :param fft_shift: shift signal in numpy style
        
        returns: gabor function, ndarray size (M,N), complex-valued
        '''
        R = np.array([
            [np.cos(theta), -np.sin(theta)], 
            [np.sin(theta), np.cos(theta)]
        ], np.float64)
        
        R_inv = np.array([
            [np.cos(theta), np.sin(theta)], 
            [-np.sin(theta), np.cos(theta)]
        ], np.float64)
        
        # Term that stretches the gaussian
        D = np.array([[1, 0], [0, slant * slant]])
    
        curv = np.matmul(R, np.matmul(D, R_inv)) / (2 * sigma * sigma)
        
        gab = np.zeros((M, N), np.complex128)
        xx = np.empty((2, 2, M, N))
        yy = np.empty((2, 2, M, N))
        
        # Handle periodicity by summing over neighboring cells
        for ii, ex in enumerate([-1, 0]):
            for jj, ey in enumerate([-1, 0]):
                xx[ii, jj], yy[ii, jj] = np.mgrid[
                    offset + ex * M: offset + M + ex * M,
                    offset + ey * N: offset + N + ey * N
                ]
        
        # Compute Gabor function
        arg = -(curv[0, 0] * xx * xx + (curv[0, 1] + curv[1, 0]) * xx * yy + curv[1, 1] * yy * yy) + \
            1.j * (xx * xi * np.cos(theta) + yy * xi * np.sin(theta))
        
        # Sum over periodic replications
        gab = np.exp(arg).sum((0, 1))
        
        # Normalize
        norm_factor = 2 * np.pi * sigma * sigma / slant
        gab = gab / norm_factor
        
        if fft_shift:
            gab = np.fft.fftshift(gab, axes=(0, 1))
            
        return gab