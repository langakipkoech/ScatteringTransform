import numpy as np 
import torch
import torch.fft as fft 
from FilterSet import Filter_Set

class ScatteringTransform:
    '''
    2D scattering transform implementation.
    '''
    def __init__(self, M, N, J=None, L=4, order=2, precision="single"):
        '''
        Initialize ScatteringTransform.
        
        :param M: spatial dimension (height)
        :param N: spatial dimension (width)
        :param J: Number of scales (default: log2(min(M,N)) - 1)
        :param L: number of orientations
        :param order: Max scattering order (1 or 2)
        :param precision: 'single' or 'double'
        '''
        
        self.M = M
        self.N = N 
        self.J = J if J is not None else int(np.log2(min(M, N))) - 1
        self.L = L
        self.order = order
        self.precision = precision 
        
        # Generate filters: wavelets and lowpass
        filter_gen = Filter_Set(M, N, J, L)
        self.filters = filter_gen.generate_wavelets(precision=precision)
        
        self.dtype = torch.float32 if precision == 'single' else torch.float64
        self.cdtype = torch.complex64 if precision == 'single' else torch.complex128
        
    def forward(self, x):
        '''
        Compute scattering transform.
        
        :param x: Input signal of shape (M, N) or (batch, M, N)
        :returns: Dictionary with 'S0', 'S1', and optionally 'S2' coefficients
        '''
        # Handle batched input
        if x.dim() == 2:
            x = x.unsqueeze(0)
            squeeze_output = True 
        else: 
            squeeze_output = False 
            
        batch_size = x.shape[0]
        
        # Convert to correct dtype
        x = x.to(self.dtype)
        
        # Compute FFT of input
        x_fft = fft.fft2(x)
        
        # Storage for scattering coefficients
        S0 = []
        S1 = []
        S2 = []
        
        # Zeroth order: S0 = |x * phi|
        S0_coefficients = self._lowpass(x_fft)
        S0.append(S0_coefficients)
        
        # First order: S1 = ||x * psi[j1,l1]| * phi|
        U1_list = []  # Store for second order
        
        for j1 in range(self.J):
            for l1 in range(self.L):
                # Convolve with wavelet psi[j1, l1]
                U1 = self._wavelet_transform(x_fft, j1, l1)
                
                # Apply modulus
                U1_mod = torch.abs(U1)
                U1_mod_fft = fft.fft2(U1_mod)
                
                # Pass through lowpass filter
                S1_coeff = self._lowpass(U1_mod_fft)
                S1.append(S1_coeff)
                
                # Store for second order (only if needed)
                if self.order >= 2:
                    U1_list.append((U1_mod_fft, j1, l1))
        
        # Second order: S2 = |||x * psi[j1,l1]| * psi[j2,l2]| * phi|
        if self.order >= 2:
            for u1_mod_fft, j1, l1 in U1_list:  # FIXED: unpacked variable name
                for j2 in range(j1 + 1, self.J):  # j2 > j1 for scale separation
                    for l2 in range(self.L):
                        # FIXED: Use correct variable u1_mod_fft (not U1_mod_fft)
                        U2 = self._wavelet_transform(u1_mod_fft, j2, l2)
                        
                        # Apply modulus
                        U2_mod = torch.abs(U2)
                        U2_mod_fft = fft.fft2(U2_mod)
                        
                        # Lowpass filter
                        S2_coeff = self._lowpass(U2_mod_fft)
                        S2.append(S2_coeff)
        
        # Stack coefficients
        result = {
            'S0': torch.stack(S0, dim=1),  # (batch, 1, M, N)
            'S1': torch.stack(S1, dim=1),  # (batch, J*L, M, N)
        }
        
        # Only add S2 if it exists
        if S2:
            result['S2'] = torch.stack(S2, dim=1)  # (batch, n_pairs, M, N)
        
        # FIXED: Actually use squeeze_output
        if squeeze_output:
            result = {k: v.squeeze(0) for k, v in result.items()}
        
        return result
    
    def _lowpass(self, x_fft):
        '''
        Apply lowpass filter (scaling function phi).
        
        :param x_fft: Input signal in frequency domain
        :returns: Lowpass filtered signal in spatial domain
        '''
        phi = self.filters['phi'].to(x_fft.device)
        phi_complex = phi.to(self.cdtype)
        
        # Multiply in Fourier domain (convolution in spatial domain)
        y_fft = x_fft * phi_complex
        
        # Inverse FFT and take real part
        y = fft.ifft2(y_fft).real 
        
        return y
    
    def _wavelet_transform(self, x_fft, j, l):
        '''
        Apply wavelet convolution in Fourier domain.
        
        :param x_fft: Input signal in Fourier domain
        :param j: scale index
        :param l: orientation index
        :returns: Wavelet-transformed signal in spatial domain
        '''
        # Select specific filter
        psi = self.filters['psi'][j, l].to(x_fft.device)
        psi_complex = psi.to(self.cdtype)
        
        # Multiply in Fourier domain (convolution in spatial domain)
        y_fft = x_fft * psi_complex
        
        # Inverse FFT
        y = fft.ifft2(y_fft)
        
        return y
    
    def compute_scattering_coefficients(self, x, average=True):
        '''
        Compute scattering coefficients with optional spatial averaging.
        
        :param x: Input tensor
        :param average: If True, spatially average the coefficients
        :returns: Flattened scattering coefficient vector
        '''
        scatter_dict = self.forward(x)
        
        coeffs = []
        for key in ['S0', 'S1', 'S2']:
            if key in scatter_dict:
                coeff = scatter_dict[key]
                if average:
                    # Spatial averaging
                    coeff = coeff.mean(dim=(-2, -1))
                else:
                    # Flatten spatial dimensions
                    coeff = coeff.flatten(start_dim=-2)
                coeffs.append(coeff)
        
        # Concatenate all coefficients
        all_coeffs = torch.cat(coeffs, dim=-1)
        
        return all_coeffs


if __name__ == '__main__':
    # Test image
    M, N = 64, 64
    x = torch.randn(M, N)
    
    # Initialize scattering transform
    scat = ScatteringTransform(M, N, J=3, L=4, order=2)
    
    # Compute scattering transform
    result = scat.forward(x)
    
    print("Scattering Transform Results:")
    print(f"S0 shape: {result['S0'].shape}")
    print(f"S1 shape: {result['S1'].shape}")
    if 'S2' in result:
        print(f"S2 shape: {result['S2'].shape}")

    #display sample coefficients
    print("="*60)
    print("SCATTERING TRANSFORM STATISTICS")
    print("="*60)
    print(f"S0 shape: {result['S0'].shape}")
    print(f"S0 range: [{result['S0'].min():.3f}, {result['S0'].max():.3f}]")
    print(f"S0 mean: {result['S0'].mean():.3f}, std: {result['S0'].std():.3f}")
    print()
    print(f"S1 shape: {result['S1'].shape}")
    print(f"S1 range: [{result['S1'].min():.3f}, {result['S1'].max():.3f}]")
    print(f"S1 mean: {result['S1'].mean():.3f}, std: {result['S1'].std():.3f}")
   
    print(f"S2 shape: {result['S2'].shape}")
    # Compute averaged coefficients
    coeffs = scat.compute_scattering_coefficients(x, average=True)
    print(f"\nScattering coefficient vector shape: {coeffs.shape}")