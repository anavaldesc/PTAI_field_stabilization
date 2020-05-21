#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 12 15:50:22 2020

@author: banano
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.constants import h

# Constants:
pi = np.pi
hbar = 1.054571726e-34                        # Reduced Planck's constant
wavelength = 780e-9                          # Imaging wavelength
k0 = 2 * pi / wavelength
um = 1e-6
cm = 1e-2
mW = 1e-3

# Space:
nx = 2 ** 10
nz = 2 ** 5
x_max = 1000 * um
z_max = 2 * um
x = np.linspace(-x_max, x_max, nx, endpoint=False)
z = np.linspace(-z_max, z_max, nz)
dx = x[1] - x[0]
dz = z[1] - z[0]

# Fourier space:
k = 2*pi*np.fft.fftshift(np.fft.fftfreq(nx, d=dx))

k_2D = np.empty([nx,nx])

for i in range(nx):
    for j in range(nx):
        k_2D[i,j] = (k[i]**2 + k[j]**2)


# Wavenumber of nyquist mode - shortest wavenumber we can resolve:
k_nyquist = pi/dx

# The kinetic energy operator in Fourier space
K = k_2D / (2 * k0)

def chi(delta, E, rho):
    
    """
    I: intensity in units of Isat
    delta: detuning in units of linewidth
    rho: density
    """
    
    k0 = 2 * np.pi / 780e-9
    sigma0 = 6 * np.pi / k0 **2
    I = np.abs(E)**2
#    I_sat = 2.503 * mW / cm ** 2
#    I /= I_sat
    
    chi = sigma0 / k0 * (1.0j - 2 * delta) / (1 + I + 2 * delta ** 2) * rho
    
    return chi

def split_step2(E, dz, chi_val):

    """"Evolve E in time from z to z + dz using one
    step of the second order Fourier split-step method"""

    E *= np.exp(-1j / (2 * k0) * chi_val * dz/2)
#    plt.imshow(np.abs(E)**2)
#    plt.colorbar()
#    plt.title('E0')
#    plt.show()
    f_E = np.fft.fftshift(np.fft.fft2(E))
#    psd = np.abs(f_E)**2
#    plt.imshow(psd, vmax=psd.max()/10)
#    plt.colorbar()
#    plt.title('E1')
#    plt.show()
    f_E *= np.exp(-1j * K * dz)
#    plt.imshow(np.abs(f_E)**2)
#    plt.colorbar()
#    plt.title('E2')
#    plt.show()
    E = np.fft.ifft2(f_E)

    E *= np.exp(-1j / (2 * k0) * chi_val * dz/2)
#    plt.imshow(np.abs(E)**2)
#    plt.colorbar()
#    plt.title('E3')
#    plt.show()
    
    return E

def TF_3D(xyz_vals, n0, x0, y0, z0, rx, ry, rz):
    
    """
    Compute 3D Thomas-Fermi density profile with peak density n0, 
    radi rx, ry, rz and centered at (x0, y0, z0)
    """
    

    condition = (1 - (xyz_vals[0]-x0) **2 / rx **2 - 
                (xyz_vals[1]-y0) **2 / ry **2 - 
                (xyz_vals[2]-z0) **2 / rz ** 2)
    condition[condition <= 0.0] = 0.0
    
    condition *= n0
    
    return condition

def gaussian_2D(xy_vals, bkg, amp, x0, y0, sigmax, sigmay) :

    gauss2D = bkg + amp*np.exp(-1*(xy_vals[0]-x0)**2/(2*sigmax**2)
                                -1*(xy_vals[1]-y0)**2/(2*sigmay**2))
    
    
    return gauss2D


#%%
#ok, let's make a BEC
# consider density of 10^13 cm^-3
r_tf = 20 * um
xyz_vals = np.meshgrid(x,x,z)
n0 = 1e13 / cm ** 3
TF = TF_3D(xyz_vals, 0.1, 0, 0, 0, r_tf, r_tf, z_max)

plt.imshow(TF.sum(axis=2))
plt.colorbar()
plt.show()

#and a probe... how intense should it be?
xy_vals = np.meshgrid(x,x)
sigma = 0.9 * cm
probe = np.sqrt(gaussian_2D(xy_vals, 0, 1e-12, 0, 0, sigma, sigma))
plt.imshow(probe)
plt.colorbar()
plt.show()

#%%
# and now let's attempt to simulate an absoprtion image
delta = 0.
E0 = np.array(probe, dtype ='complex')
E0 *= 0
E0 += 1e-2
I_probe = np.abs(E0) ** 2
tf_2d = TF.sum(axis=2) / h #* 8e32
chi_TF = chi(delta, E0, tf_2d)
E_new = split_step2(E0, dz, chi_TF)
I_atoms = np.abs(E_new) ** 2
OD = np.log(I_atoms/I_probe)
OD_int = OD[int(nx/2)-50:int(nx/2)+50,int(nx/2)-50:int(nx/2)+50]
plt.imshow(OD_int)
plt.colorbar()
plt.show()

#E0 = np.array(probe, dtype ='complex')
#E0 *= 0
#E0 += 1e-2
#for i in range(nz-1):
#    if i == 0:
#        E = E0
#    rho = TF[:,:,i] / h
#    chi_TF = chi(delta, E, rho)
#    E_new = split_step2(E, dz, chi_TF)
#    E = E_new
#    if i % 3 == 0:
##        plot_psd = True
##        plt.imshow(chi_TF.imag)
##        plt.colorbar()
##        plt.show()
#        I_atoms = np.abs(E_new) ** 2
#        OD = np.log(I_atoms/I_probe)
#        OD = OD[int(nx/2)-50:int(nx/2)+50,int(nx/2)-50:int(nx/2)+50]
#        plt.imshow(OD)
#        plt.colorbar()
##        plt.imshow(np.abs(E_new)** 2)
#        plt.show()

#%%
E0 = np.array(probe, dtype ='complex')
E0 *= 0
E0 += 1e-2
E0 = split_step2(E0, dz, chi_TF)
for i in range(20):
    if i == 0:
        E = E0
    rho = TF[:,:,0] 
    chi_TF = chi(delta, E, rho) * 0
    E_new = split_step2(E, 60 * dz, chi_TF)
    E = E_new
    if i % 3 == 0:
#        plt.imshow(chi_TF.imag)
#        plt.colorbar()
#        plt.show()
        I_atoms = np.abs(E_new) ** 2
        OD = np.log(I_atoms/I_probe)
        OD = OD[int(nx/2)-50:int(nx/2)+50,int(nx/2)-50:int(nx/2)+50]
        plt.imshow(OD)
        plt.title(i)
        plt.colorbar()
#        plt.imshow(np.abs(E_new)** 2)
        plt.show()