import torch
import math

class FHRR_Algebra:
    def __init__(self, dim, device='cuda'):
        self.dim = dim
        self.device = device
        
    def generate_unitary_anchor(self):
        # Generate phases (angles) between -pi and pi
        phases = (torch.rand(self.dim, device=self.device) * 2 * math.pi) - math.pi
        # Return as complex numbers on the unit circle
        return torch.polar(torch.ones(self.dim, device=self.device), phases)

    def bind(self, vec1, vec2):
        # O(d) Hardware-native binding. 
        # Multiplying complex phases simply adds their angles! No FFT required.
        return vec1 * vec2 

    def unbind(self, bound_vec, key_vec):
        # O(d) Unbinding. Multiply by the complex conjugate to subtract angles.
        return bound_vec * torch.conj(key_vec)
        
    def to_real_space(self, complex_vec):
        # Interface Layer: Map back to Llama-3 real dimensions
        # Returns an [8192] real vector for a [4096] complex vector
        return torch.cat([complex_vec.real, complex_vec.imag], dim=-1)