# -*- coding: utf-8 -*-
"""
Created on Tue Sep 24 13:43:01 2019

@author: ispielma
"""

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec 
plt.style.use('matplotlibrc')        

import numpy as np
import lmfit
import qgas.IgorFileIO
import qgas.IgorLegacy.IgorBin

import scipy
import scipy.ndimage
from astropy.convolution import Gaussian2DKernel, interpolate_replace_nans
kernel = Gaussian2DKernel(x_stddev=1)

import h5py

from tqdm import tqdm

def MakeDiskMask(xyvals, Options):
    """
    Makes a disk shaped mask that is zero inside the mask
    """

    rad = Options['rad']
    x0 = Options['x0']
    y0 = Options['y0']
    
    reject = ((meshgrid[0]-x0)**2 + (meshgrid[1] - y0)**2) < rad**2
    mask = np.ones_like(meshgrid[0])
    
    mask[reject] = 0
    pts = np.sum(mask)
    
    Options['mask'] = mask
    Options['pts'] = pts

def MakeSquareMask(xyvals, Options, Fill=0):
    """
    Makes a rectangular mask that is one inside the mask
    """

    mask = np.zeros_like(meshgrid[0])
    mask.fill(Fill)
    dx = 2*Options['dx']+1
    dy = 2*Options['dy']+1

    method = Options.get('method', None)
    
    # Positions referenced to matrix ordering
    if method == 'tl':
        accept = mask[0:dx,0:dy] # Fill a view
    elif method == 'tr':
        accept = mask[0:dx,-dy:] # Fill a view
    elif method == 'bl':
        accept = mask[-dx:,0:dy] # Fill a view
    elif method == 'br':
        accept = mask[-dx:,-dy:] # Fill a view
    else:
        x0 = Options['x0']
        y0 = Options['y0']
        accept = mask[x0-Options['dx']:x0+Options['dx']+1,y0-Options['dy']:y0+Options['dy']+1]
        
    accept.fill(1)

    pts = np.sum(mask)
    
    Options['mask'] = mask
    Options['pts'] = pts

def SNR_Model(OD, std_bg):
    """
    Expected SNR from absorption imaging
    """
    
    return (OD / std_bg) * np.sqrt(2 / (1 + np.exp(OD)))

def SNR_Model_BG(OD, std_bg, bg_counts):
    """
    Expected SNR from absorption imaging, including BG term
    """
    
    return (OD / std_bg) * np.sqrt(2 / (1 + np.exp(OD)))


# ###########################################################################
#
# START: Configuration Options
#
# ###########################################################################

Mag = 16
pixel_dx = 16

# Grid of xy values to be used throughout
vals = np.linspace(-32,31,64)
meshgrid = np.meshgrid(vals, vals)

#
# Files
#

scan1_folder = 'scan1'
scan1_file_string = 'ProEM_27Jan2017'
scan1_range = [1091, 1140]

scan2_folder = 'scan2'
scan2_file_string = 'ProEM_27Jan2017'
scan2_range = [1142, 1191]

Load_Igor = False # Reload all the igor files and do pre-processing (slow)
h5_file = 'SNR_Processed_data.h5'

#
# Masks: We use several masks
#

# Background mask for image processing

BG_Mask = {
        'rad': 19.2,
        'x0':-1.6,
        'y0':-1.6
        }

MakeDiskMask(meshgrid, BG_Mask)

# Masks for noise analysis
dx = 3
dy = 3
TL_Mask = {'method':'tl', 'dx':dx,'dy':dx}
TR_Mask = {'method':'tr', 'dx':dx,'dy':dx}
BL_Mask = {'method':'bl', 'dx':dx,'dy':dx}
BR_Mask = {'method':'br', 'dx':dx,'dy':dx}
Center_Mask = {'dx':dx,'dy':dx, 'x0': 31, 'y0':31}

MakeSquareMask(meshgrid, TL_Mask, Fill=np.nan)
MakeSquareMask(meshgrid, TR_Mask, Fill=np.nan)
MakeSquareMask(meshgrid, BL_Mask, Fill=np.nan)
MakeSquareMask(meshgrid, BR_Mask, Fill=np.nan)
MakeSquareMask(meshgrid, Center_Mask, Fill=np.nan)

# ###########################################################################
#
# Setup lmfit models
#
# ###########################################################################


# Next we setup a fit: I'll do a deformed gaussian peak.
def gaussian_2d(xy_mesh, amp, xc, yc, sigma_x, sigma_y, alpha):
    
    # unpack 1D list into 2D x and y coords
    (x, y) = xy_mesh
    
    # make the 2D Gaussian matrix
    gauss = amp*np.exp(-((x-xc)**2/(2*sigma_x**2)+(y-yc)**2/(2*sigma_y**2))**alpha)
    
    # flatten the 2D Gaussian down to 1D
    return np.ravel(gauss)

lmfit_gauss_model = lmfit.Model(gaussian_2d)

# Next we setup a fit: I'll do a deformed gaussian peak.
def single_exp(tvals, amp, tau, amp2, tau2, C):
    
    exp = amp*np.exp(-(tvals / tau))  
    exp += amp2 / (tvals-tau2)   + C
    return exp

lmfit_exp_model = lmfit.Model(single_exp)


# ###########################################################################
#
# Read in data!
#
# ###########################################################################

if Load_Igor:
    num_files = (scan1_range[1] - scan1_range[0]) + 1
    upulsetimes = []
    counts = []
    
    darks = []
    probes = []
    ODs = []
    OD_NOISEs = []
    amps = []
    xcs = []
    ycs = []
    sigma_xs = []
    sigma_ys = []
    alphas = []

    # Run through and load all the dark frames first?
    for index in range(scan1_range[0],scan1_range[1]+1):
        FileName = f'{scan1_folder}/{scan1_file_string}_{index:04d}.ibw'
        IBWData = qgas.IgorLegacy.IgorBin.LoadIBW(FileName)
        
        dark = qgas.IgorFileIO.DicePackedImage(IBWData["Data"], NumFrames=60)[0]
        dark = scipy.ndimage.gaussian_filter(dark, sigma=[3,0.1])
        darks.append(dark)
    
    darks = np.array(darks)
    dark_mean = darks.mean(axis=0)
    
    for index in tqdm(range(scan1_range[0],scan1_range[1]+1)):
        FileName = f'{scan1_folder}/{scan1_file_string}_{index:04d}.ibw'
        IBWData = qgas.IgorLegacy.IgorBin.LoadIBW(FileName)
        
        # Store the experimental information
        Note = IBWData["Note"];
        
        # Internal properties associlated with igor waves
        ExpInf = {"FileName": FileName};
        
        # Begin acquiring experimental information from note
        qgas.IgorFileIO.StandardExperimentInfo(ExpInf, Note);
        
        upulsetimes.append ( float(ExpInf['IndexedValues'][:-1]) )
            
        # Break the image into it's composite pieces
        Data = {}
        Data["Raw"] = qgas.IgorFileIO.DicePackedImage(IBWData["Data"], NumFrames=60)
    
        # Now with not much "extra" work, I am going to compute the OD
        
        probe = Data["Raw"][1]
    
        Data["OD"] = []
        Data["OD_NOISE"] = []
        Data['amp'] = []
        Data['xc'] = []
        Data['yc'] = []
        Data['sigma_x'] = []
        Data['sigma_y'] = []
        Data['alpha'] = []
        
        Count = []
        for index in range(4,60):
            atoms = Data["Raw"][index] - dark_mean
            
            # Generic Image preperation
            
            atoms_nans = (atoms <= 0)
            atoms[atoms_nans] = 1
            OD = -np.log((atoms) / (probe-dark_mean))
            OD[atoms_nans] = np.nan
            
            # Now use astropy to clean this up
            OD = interpolate_replace_nans(OD, kernel)
    
            BGCounts = np.sum(OD*BG_Mask['mask'])/BG_Mask['pts']
            OD -= BGCounts
            
            Data["OD"].append(OD)
            
            # Construct smooth background images
            # Next we setup a fit: I'll do a deformed gaussian peak.
            pguess = lmfit.Parameters()
            pguess.add('amp', value=1, min=-0.5, max=6)
            pguess.add('xc', value=0, min=-10, max=10)
            pguess.add('yc', value=0, min=-10, max=10)
            pguess.add('sigma_x', value=10, min=2, max=30)
            pguess.add('sigma_y', value=10, min=2, max=30)
            pguess.add('alpha', value=1, min=0.1, max=4)
    
            lmfit_result = lmfit_gauss_model.fit(np.ravel(OD), 
                                           pguess,
                                           xy_mesh=meshgrid)
            
            model = gaussian_2d(meshgrid, **lmfit_result.values).reshape(OD.shape)
            
            OD_NOISE = OD - model
            OD_NOISE = scipy.ndimage.gaussian_filter(OD_NOISE, sigma=3)
            OD_NOISE = (OD-model) - OD_NOISE 
    
            Data["OD_NOISE"].append(OD_NOISE)
            
            Data['amp'].append(lmfit_result.values['amp'])
            Data['xc'].append(lmfit_result.values['xc'])
            Data['yc'].append(lmfit_result.values['yc'])
            Data['sigma_x'].append(lmfit_result.values['sigma_x'])
            Data['sigma_y'].append(lmfit_result.values['sigma_y'])
            Data['alpha'].append(lmfit_result.values['alpha'])
            
        
        probes.append(probe)
        ODs.append(np.array(Data["OD"]))
        OD_NOISEs.append(np.array(Data["OD_NOISE"]))
        amps.append(np.array(Data["amp"]))
        xcs.append(np.array(Data["xc"]))
        ycs.append(np.array(Data["yc"]))
        sigma_xs.append(np.array(Data["sigma_x"]))
        sigma_ys.append(np.array(Data["sigma_y"]))
        alphas.append(np.array(Data["alpha"]))
                
    
    upulsetimes = np.array(upulsetimes)
    probes = np.array(probes)
    ODs = np.array(ODs)
    OD_NOISEs = np.array(OD_NOISEs)
    amps = np.array(amps)
    xcs = np.array(xcs)
    ycs = np.array(ycs)
    sigma_xs = np.array(sigma_xs)
    sigma_ys = np.array(sigma_ys)
    alphas = np.array(alphas)
    
    ordering = np.argsort(upulsetimes)
    upulsetimes = upulsetimes[ordering]
    probes = probes[ordering, :, :]
    darks = darks[ordering, :, :]
    ODs = ODs[ordering, :, :, :]
    OD_NOISEs = OD_NOISEs[ordering, :, :, :]
    amps = amps[ordering, :]
    xcs = amps[ordering, :]
    ycs = amps[ordering, :]
    sigma_xs = amps[ordering, :]
    sigma_ys = amps[ordering, :]
    alphas = amps[ordering, :]
    
    # Now save this junk to the h5 file
    with h5py.File(h5_file, 'w') as f:
        f['upulsetimes'] = upulsetimes
        f['probes'] = probes
        f['darks'] = darks
        f['ODs'] = ODs
        f['OD_NOISEs'] = OD_NOISEs
        f['amps'] = amps
        f['xcs'] = xcs
        f['ycs'] = ycs
        f['sigma_xs'] = sigma_xs
        f['sigma_ys'] = sigma_ys
        f['alphas'] = alphas
        
#
# Now open the existing h5 file
#

with h5py.File(h5_file, 'r') as f:
    upulsetimes = f['upulsetimes'][:]
    probes = f['probes'][:]
    darks = f['darks'][:]
    ODs = f['ODs'][:]
    OD_NOISEs = f['OD_NOISEs'][:]
    amps = f['amps'][:]
    xcs = f['xcs'][:]
    ycs = f['ycs'][:]
    sigma_xs = f['sigma_xs'][:]
    sigma_ys = f['sigma_ys'][:]
    alphas = f['alphas'][:]

#
# Do fast, live processing of these images
#

shape = list(ODs.shape)
shape[0] = 14
counts = np.zeros( shape[0:2] )
BGcounts = np.zeros( shape[0:2] )
TR_std = np.zeros( shape[0:2] )
TL_std = np.zeros( shape[0:2] )
BL_std = np.zeros( shape[0:2] )
BR_std = np.zeros( shape[0:2] )
Center_std = np.zeros( shape[0:2] )
Center_mean = np.zeros( shape[0:2] )
Center_mean_model = np.zeros( shape[0:2] )
shots = np.arange(shape[1])

for i in range(shape[0]):
    upulsetime = upulsetimes[i]
    for j in shots:
        counts[i,j] = np.sum(ODs[i,j,:,:])
        BGcounts[i,j] = np.sum(OD_NOISEs[i,j,:,:])
        
        TR_std[i,j] = np.nanstd(OD_NOISEs[i,j,:,:]*TR_Mask['mask'])
        TL_std[i,j] = np.nanstd(OD_NOISEs[i,j,:,:]*TL_Mask['mask'])
        BR_std[i,j] = np.nanstd(OD_NOISEs[i,j,:,:]*BR_Mask['mask'])
        BL_std[i,j] = np.nanstd(OD_NOISEs[i,j,:,:]*BL_Mask['mask'])

        Center_std[i,j] = np.nanstd(OD_NOISEs[i,j,:,:]*Center_Mask['mask'])
        Center_mean[i,j] = np.nanmean(ODs[i,j,:,:]*Center_Mask['mask'])

    # Now do a fit to the center_mean to smooth the number vs time to compute
    # SNR
    
    pguess = lmfit.Parameters()
    pguess.add('amp', value=1, min=-0.1, max=6)
    pguess.add('tau', value=5, min=0.1, max=20)
    pguess.add('amp2', value=0.5)
    pguess.add('tau2', value=-1, max=0)
    pguess.add('C', value=0, min=-1, max=1)

    lmfit_result = lmfit_exp_model.fit(Center_mean[i,:], 
                                   pguess,
                                   tvals=shots)
    
    Center_mean_model[i,:] = single_exp(shots, **lmfit_result.values)


SNR = Center_mean / Center_std
SNR_Computed = SNR_Model(Center_mean_model , np.mean(BR_std) )

fig = plt.figure()

gs = gridspec.GridSpec(2,2)
fig.set_size_inches(4.0, 4)
fig.subplots_adjust(left=0.13, bottom=.14, right=.90, top=.88, hspace=0.5, wspace=0.1)

ax = plt.subplot(gs[0,0])
ax.imshow(SNR, aspect='auto', vmin=0, vmax=5)
# ax.imshow(Center_mean - Center_mean_model, aspect='auto')
ax.set_title(r'(a) Measured SNR', x=0, y=1.0, loc='left')

ax.set_xlabel(f'Shot number $m$')
ax.set_ylabel(f'Command $\epsilon$')

yticks = np.arange(0,14,3)
ax.set_yticks(yticks)

epsilons = (1-np.cos(2*np.pi*upulsetimes[yticks]/200e-6))/2
epsilons = [f'{epsilon:0.2f}' for epsilon in epsilons]
ax.set_yticklabels(epsilons)


ax = plt.subplot(gs[0,1])
pos = ax.imshow(SNR_Computed, aspect='auto', vmin=0, vmax=5)

# Colorbar
box = ax.get_position()
pad, width = 0.02, 0.015
cax = fig.add_axes([box.xmax + pad, box.ymin, width, box.height])
cb = fig.colorbar(pos, cax=cax)
cb.set_label(r'SNR')
cb.ax.yaxis.set_label_position('right')


ax.set_title(r'(b) Modeled SNR', x=0, y=1.0, loc='left')
ax.set_xlabel(f'Shot number $m$')
ax.set_yticklabels([])

ax = plt.subplot(gs[1,0])

epsilon = (1-np.cos(2*np.pi*upulsetimes[1]/200e-6))/2
ax.set_title(f'(c) SNR at $\\epsilon={epsilon:.2f}$', x=0, y=1.0, loc='left')
ax.plot(SNR[1,:], "o", color='k')
ax.plot(SNR_Computed[1,:], color='red')

ax.axhline(0)
ax.set_xlabel(f'Shot number $m$')
ax.set_ylabel(f'SNR')
ax.set_xlim(0,None)
ax.set_ylim(-0.2,5)

ax = plt.subplot(gs[1,1])

epsilon = (1-np.cos(2*np.pi*upulsetimes[6]/200e-6))/2
ax.set_title(f'(d) SNR at $\\epsilon={epsilon:.2f}$', x=0, y=1.0, loc='left')
ax.plot(SNR[6,:], "o", color='k')
ax.plot(SNR_Computed[6,:], color='red')

# ax.plot(Center_mean[1,:]*6, "o", color='k')
# ax.plot(Center_mean_model[1,:]*6, color='red')

ax.axhline(0)
ax.set_xlabel(f'Shot number $m$')
ax.set_xlim(0,None)
ax.set_ylim(-0.2,5)
ax.set_yticklabels([])

fig.savefig('Fig_SNR.pdf')

