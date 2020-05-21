#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 13 16:37:52 2020

@author: banano
"""

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.gridspec import GridSpec

def h(k_2D, z, delta=0):
    
#    norm_k = abs_k(k_2D)
    k_0 = 2 * np.pi / 780e-9
    sigma_0 = 6 * np.pi / k_0 **2
    prefactor = -sigma_0 / (2 * k_0 * (1 + 4 * delta ** 2))
    h = 2 * delta * np.sin(z * k_2D / (2 * k_0))
    h += np.cos(z * k_2D / (2 * k_0))
    h *= prefactor
    
    return h

def U(k_2D, z, alpha, delta=0):
    
    U = h(k_2D, z, delta) / (h(k_2D, z, delta) **2 + alpha ** 2)
    
    return U

def refocus(img,k_2D,  dz, alpha, delta=0, plotme=False):
    
    k_0 = 2 * np.pi / 780e-9
    fft_img = np.fft.fftshift(np.fft.fft2(img-img.mean()))
    fft_img_refocused = U(k_2D, dz, alpha, delta) * fft_img    
    img_refocused = np.fft.ifft2(fft_img*0+fft_img_refocused)
#    mod = (img_refocused * np.conjugate(img_refocused)).real
    mod = np.abs(img_refocused)  / (2 * k_0)
    
    if plotme:
        
        xc = 321
        yc = 218
        radius = 20  
        mod_crop = mod = mod[yc-2*radius:yc+2*radius, xc-2*radius:xc+2*radius]
    
        plt.figure(0, figsize=(10, 3))
        gs = GridSpec(1, 2)
        gs.update(left=0.1, right=0.95, bottom=0.1, top=0.96,
                  wspace=0.2, hspace=0.1)
    
        plt.subplot(gs[0])
        plt.imshow(np.abs(fft_img_refocused)**2, vmax=7e3)
        plt.colorbar()
    
        plt.subplot(gs[1])
        plt.imshow((mod_crop)+img.mean())
        plt.colorbar()
        plt.show()
    
    return mod

def gaussian_TF_2D(xy_vals, bkg, amp_g , amp_tf, 
                x0, sigmax, rx_tf, y0, sigmay, ry_tf) :

    gauss2D = bkg + amp_g*np.exp(-1*(xy_vals[0]-x0)**2/(2*sigmax**2)
                                -1*(xy_vals[1]-y0)**2/(2*sigmay**2))
    
    condition = (1 - (xy_vals[0]-x0) **2 / rx_tf - 
                   (xy_vals[1]-y0) **2 / ry_tf) 
    condition[condition <= 0.0] = 0.0
#    
#    plt.plot(condition)
#    plt.show()
    
    TF = amp_tf *(condition)**(3.0/2.0)
    
    return gauss2D + TF

#%%
size = 400.0
r_crop = 40
xyvals = np.meshgrid(np.arange(size)-size/2,
                   np.arange(size)-size/2)
yxvals = np.array(xyvals)

r = 200
bkg = 0
amp_g = 0
amp_tf = 1
x0 = 0
sigmax = 0.1
rx_tf = r
y0 = 0
sigmay = 0.1
ry_tf = r

TF_model = gaussian_TF_2D(xyvals, bkg, amp_g , amp_tf, 
                          x0, sigmax, rx_tf, y0, sigmay, ry_tf)

plt.imshow(TF_model)
plt.show()


x,y = TF_model.shape
k_2D = np.empty([x,y])
kx = np.arange(-x/2, x/2) + 0.5
ky = np.arange(-y/2, y/2) + 0.5
pixel = 1e-6
kx /= pixel
ky /= pixel

for i in range(x):
    for j in range(y):
        k_2D[i,j] = kx[i]**2 + ky[j]**2


slices = []
alpha = 0.2
delta = 0
distance = 6e-7*0.25
for dz in np.linspace(-distance, distance, 10):

    ref = refocus(TF_model, k_2D, dz, alpha, delta)
    x_slice = ref[int(size/2)]
    slices.append(x_slice)
    plt.plot(x_slice[100:300])
#plt.show()
#slices = np.array(slices)
#plt.imshow(slices[:,int(size/2)-int(size/5):int(size/2)+int(size/5)], cmap='jet')
#plt.show()
#plt.imshow(ref[int(size/2)-r_crop:int(size/2+r_crop),
#               int(size/2)-r_crop:int(size/2+r_crop)])
#plt.show()
