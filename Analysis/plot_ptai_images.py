#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May  1 12:14:07 2020

@author: banano
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import lmfit
from matplotlib import rcParams
import utils
import nrefocus
from skimage.restoration import denoise_tv_chambolle

# uwavelock
date = 20180315
sequence = 88 #87
date = 20190507
sequence = 152
scanned_parameter = 'ARP_FineBias'
camera = 'XY_Mako'
sequence_type = 'lock_vs_t'
#df, sorted_img = utils.data_processing(camera, 
#                                       date, 
#                                       sequence, 
#                                       sequence_type, 
#                                       scanned_parameter='ARP_FineBias',
#                                       redo_prepare=True)

image_keys = ['bare_lock_atoms0', 
              'bare_lock_probe0', 
              'bare_lock_atoms1', 
              'bare_lock_probe1',
              'TOF_dark']

df, img = utils.get_uwavelock_images(camera,
                                       date,
                                       sequence,
                                       sequence_type,
                                       image_keys=image_keys,
                                       scanned_parameter='ARP_FineBias',
                                       redo_prepare=True)
dark = img[:,-1]
img = img[:,0:-1]
atoms = img[:,0::2]
probe = img[:,1::2] 
for i in range(2):
    atoms[:,i] -= dark
    probe[:,i] -= dark

od = np.log(atoms/probe)[:,:]

#bg = sorted_img[:,2::3]
#
#atoms -= bg
#probe -= bg
#%%

od1 = od[0,0]
z = 0.1
alpha = 0.5

def puck(xc, yc, radius):
    return (x-xc)**2 + (y-yc)**2 < radius**2


def h(k_2D, z, delta=0):
    
#    norm_k = abs_k(k_2D)
    k_0 = 2 * np.pi / 780e-9
    h = 2 * delta * np.sin(z * k_2D / (2 * k_0))
    h += np.cos(z * k_2D / (2 * k_0))
    
    return h

def U(k_2D, z, alpha, delta=0):
    
    U = h(k_2D, z, delta) / (h(k_2D, z, delta) **2 + alpha ** 2)
    
    return U

def abs_k(k_2D):
    
    x, y, _ = k_2D.shape
    abs_k = np.empty([x,y])
    for i in range(x):
        for j in range(y):

            abs_k[i,j] = np.dot(k_2D[i,j], k_2D[i,j])
            
    return abs_k


x,y = od1.shape
k_2D = np.empty([x,y])
kx = np.arange(-x/2, x/2) + 0.5
ky = np.arange(-y/2, y/2) + 0.5
pixel = 7.2e-6 / 3.3
kx /= pixel
ky /= pixel

for i in range(x):
    for j in range(y):
        k_2D[i,j] = kx[i]**2 + ky[j]**2

def refocus(img,k_2D,  dz, alpha, delta=0):
    
    xc = 321
    yc = 218
    radius = 20

    
#    plt.imshow(img)
#    plt.colorbar()
#    plt.show()
    
    fft_img = np.fft.fftshift(np.fft.fft2(img-img.mean()))
    fft_img_refocused = U(k_2D, dz, alpha, delta) * fft_img    
    img_refocused = np.fft.ifft2(fft_img*0+fft_img_refocused)
    mod = (img_refocused * np.conjugate(img_refocused)).real
    mod = np.abs(img_refocused) 
#    mod = mod[yc-2*radius:yc+2*radius, xc-2*radius:xc+2*radius]
    
    
#    fig = plt.figure(0, figsize=(10, 3))
#    gs = GridSpec(1, 2)
#    gs.update(left=0.1, right=0.95, bottom=0.1, top=0.96,
#              wspace=0.2, hspace=0.1)
#
#    plt.subplot(gs[0])
#    plt.imshow(np.abs(fft_img_refocused)**2, vmax=7e3)
#    plt.colorbar()
#
#    plt.subplot(gs[1])
#    plt.imshow((mod)+img.mean())
#    plt.colorbar()
#    plt.show()
    
    return mod
            
dz = 1.42e-7 * 3
for dz in np.linspace(7e-7, 3e-6, 20):
#dz = 1e-9
    print(dz)
    alpha = 0.25
    delta = -0.45
    ref = refocus(od1, k_2D, dz, alpha, delta)

#%%
#hh = h(k_2d, z)
#uu = U(k_2d, z, alpha)
#plt.imshow(hh)
#plt.show()
#plt.imshow(uu)
#plt.show()#%%
    
    
    
focal_point = 330
focal_points = np.arange(300, 350, 2)
xc = 321
yc = 218
radius = 12
dz = 1.5473684210526316e-06 / 3.3 ** 2
errs = []
for i in range(50):
    
    od1 = od[i,0]
    od2 = od[i,1]
#    od1 = denoise_tv_chambolle(od1, weight=0.001, 
#                               multichannel=False)
#    od2 = denoise_tv_chambolle(od1, weight=0.001, 
#                               multichannel=False)
#    od1 = utils.refocus(od1, focal_point)
#    od2 = utils.refocus(od2, focal_point)
    
    od1 = refocus(od1, k_2D, dz, alpha, delta)
    od2 = refocus(od2, k_2D, dz, alpha, delta)

    x, y = np.meshgrid(np.arange(od1.shape[1]*1.0),
                   np.arange(od1.shape[0]*1.0))
    
    pk = puck(xc, yc, radius)
    pops = np.array([(od1 * pk).sum(), (od2 * pk).sum()])
    
    od1 = od1[yc-2*radius:yc+2*radius, xc-2*radius:xc+2*radius]
    od2 = od2[yc-2*radius:yc+2*radius, xc-2*radius:xc+2*radius]
    #
    #od_re = ref[y0-w:y0+w,x0-w:x0+w]
    #od0 = od[5,1,y0-w:y0+w,x0-w:x0+w]
    
    
    #    mpl.rcParams.update({'xtick.labelsize': 8, 'ytick.labelsize': 8})
    
    fig = plt.figure(0, figsize=(8, 5))
    gs = GridSpec(1, 2)
    gs.update(left=0.1, right=0.95, bottom=0.1, top=0.96,
              wspace=0.2, hspace=0.1)
    
    ax1 = plt.subplot(gs[0])
    plt.imshow(od1, vmin=0, vmax=1, interpolation='none')
    
    ax2 = plt.subplot(gs[1], sharex=ax1, sharey=ax1)
    plt.imshow(od2, vmin=0, vmax=1, interpolation='none')
    
    axes = [ax1, ax2]
    
    for idx in range(2):
        # c = plt.Circle((xc, yc), radius,
                       # color='yellow', linewidth=1, fill=False)
        # now that they are cropped the center of the circle is always the
        # center of the image
        c = plt.Circle((2*radius, 2*radius), radius,
                       color='yellow', linewidth=1, fill=False)
        axes[idx].add_patch(c)
    plt.show()
    
    # zero them if they are negative
    pops[pops < 0] = 0
    print('populations {} {}'.format(*pops))

    err = np.diff(pops)[0] / pops.sum()
    if np.isnan(err):
        err = 0.0
    
    errs.append(err)
#    img_sqrt = np.sqrt(od[5,1].clip(0))

  #%%  
#    plt.figure(figsize=(12, 3.5))
#    gs = GridSpec(1,3)
#    plt.subplot(gs[0])
#    plt.imshow(od0, vmin=-0.05, vmax=0.3)
#    plt.colorbar()
#    
#    plt.subplot(gs[1])
#    plt.imshow(od_re)
#    plt.colorbar()
#
#    plt.subplot(gs[2])    
#    plt.imshow(od_re-od0, cmap='RdBu',
#               vmin=-0.2, vmax=0.2)
#    plt.colorbar()
#    plt.show()
    
    

#    #dark counts ~28
#    #read noise ~22
#    
#    gmod = lmfit.Model(utils.gaussian2D)
#    xy_vals = utils.make_xy_grid(od0)
#    gauss2d = gmod.eval(xy_vals=xy_vals, bkg=0, amp=1, x0=320, sigmax=50, y0=150, sigmay=50)
#    
#    
#    gmod.set_param_hint('bkg', value=0)#, min, max=+0.1)
#    gmod.set_param_hint('amp', value=0.3)#, min=0.5, max=1.8)
#    gmod.set_param_hint('x0', value=50)
#    gmod.set_param_hint('y0', value=50)
#    gmod.set_param_hint('sigmax', value=30)#, min=10, max=100)
#    gmod.set_param_hint('sigmay', value=30)#, min=10, max=100)
#    
#    #gmod.set_param_hint('cenv', value=cv, min=cv -2, max=cv+2)
#    #gmod.set_param_hint('wid', value=wid, min=0.1, max=5)
#    params = gmod.make_params()
#    result = gmod.fit(od_re.ravel(order='F'), xy_vals=utils.make_xy_grid(od_re), params=params)
#
#    res = result.residual.reshape(od_re.shape, order='F')
    res = od_re
    res -= res.mean()
    fft = np.fft.fftshift(np.fft.fft2(res))
    res_fft.append((np.abs(fft)**2)[:,w])

#plt.plot(focal_points, res)

    plt.figure(figsize=(12, 3.5))
    gs = GridSpec(1,3)
    
    plt.subplot(gs[0])
    plt.imshow(od_re, vmin=-0.01, vmax=0.7)
    plt.colorbar()
    plt.title('od')
    plt.subplot(gs[1])
    plt.imshow(od0)#, vmin=-0.1, vmax=0.8)
    plt.colorbar()
    plt.title('fit')
    plt.subplot(gs[2])
    plt.imshow(np.abs(fft)**2)
    plt.title(focal_point)
    plt.colorbar()
    plt.show()
#print (result.fit_report(min_correl=0.25))


res_fft = np.array(res_fft)
plt.imshow(res_fft.T[0:50], vmin=0)

#%%
import numpy as np
import scipy
from scipy import ndimage
import matplotlib.pyplot as plt

f = scipy.misc.face(gray=True)
sx, sy = f.shape
X, Y = np.ogrid[0:sx, 0:sy]


r = np.hypot(X - sx/2, Y - sy/2)

rbin = (20* r/r.max()).astype(np.int)
radial_mean = ndimage.mean(f, labels=rbin, index=np.arange(1, rbin.max() +1))

plt.figure(figsize=(5, 5))
plt.axes([0, 0, 1, 1])
plt.imshow(rbin, cmap=plt.cm.nipy_spectral)
plt.axis('off')

plt.show()