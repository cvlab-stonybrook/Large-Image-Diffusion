import numpy as np
import torch
import math

class GaussianDiffusion():
    '''Gaussian Diffusion process with linear beta scheduling'''
    def __init__(self, T, schedule):
        # Diffusion steps
        self.T = T
    
        # Noise schedule
        if schedule == 'linear':
            b0=1e-4
            bT=2e-2
            self.beta = np.linspace(b0, bT, T)
        elif schedule == 'cosine':
            self.alphabar = self.__cos_noise(np.arange(0, T+1, 1)) / self.__cos_noise(0) # Generate an extra alpha for bT
            self.beta = np.clip(1 - (self.alphabar[1:] / self.alphabar[:-1]), None, 0.999)
        else:
            self.beta = schedule
            
        self.betabar = np.cumprod(self.beta)
        self.alpha = 1 - self.beta
        self.alphabar = np.cumprod(self.alpha)

    def __cos_noise(self, t):
        offset = 0.008
        return np.cos(math.pi * 0.5 * (t/self.T + offset) / (1+offset)) ** 2
   
    def sample(self, x0, t):        
        # Select noise scales
        noise_dims = (x0.shape[0],) + tuple((1 for _ in x0.shape[1:]))        
        atbar = torch.from_numpy(self.alphabar[t-1]).view(noise_dims).to(x0.device)
        assert len(atbar.shape) == len(x0.shape), 'Shape mismatch'
        
        # Sample noise and add to x0
        epsilon = torch.randn_like(x0)
        xt = torch.sqrt(atbar) * x0 + torch.sqrt(1-atbar) * epsilon        
        return xt, epsilon
    
    def inverse(self, net, shape=(1,64,64), steps=None, x=None, cond=None, start_t=None, 
                stride=1, ddim=1, 
                device='cpu'):
        # Specify starting conditions and number of steps to run for 
        if x is None:
            x = torch.randn((1,) + shape).to(device)
        if start_t is None:
            start_t = self.T
        if steps is None:
            steps = self.T

        for t in range(start_t, start_t-steps, -stride):
            at = self.alpha[t-1]
            atbar = self.alphabar[t-1]
            
            if t > 1:
                z = torch.randn_like(x)
                atbar_prev = self.alphabar[t-1-stride]
                beta_tilde = self.beta[t-1] * (1 - atbar_prev) / (1 - atbar) 
            else:
                z = torch.zeros_like(x)
                beta_tilde = 0

            with torch.no_grad():
                t = torch.tensor([t]).view(1)
                pred = net(x, t.float().to(device), cond)[:,:3,:,:]

            if ddim < 0:
                # DDPM
                x = (1 / np.sqrt(at)) * (x - ((1-at) / np.sqrt(1-atbar)) * pred) + np.sqrt(beta_tilde) * z
            else :
                # DDIM
                beta_tilde = beta_tilde * ddim
                if t > stride:
                    x = np.sqrt(atbar_prev) * (x - np.sqrt(1-atbar)*pred) / np.sqrt(atbar) + np.sqrt(1-atbar_prev-beta_tilde)*pred + np.sqrt(beta_tilde) * z
                elif t <= stride:
                    atbar_prev = 1
                    x = np.sqrt(atbar_prev) * (x - np.sqrt(1-atbar)*pred) / np.sqrt(atbar)


        return x    