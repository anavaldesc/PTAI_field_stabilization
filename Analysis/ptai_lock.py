#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 15 16:10:59 2019

@author: banano
"""


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import lmfit
from matplotlib import rcParams
import utils

fontsize = 11
rcParams['axes.labelsize'] = fontsize
rcParams['xtick.labelsize'] = fontsize
rcParams['ytick.labelsize'] = fontsize
rcParams['legend.fontsize'] = fontsize

rcParams['pdf.fonttype'] = 42 # True type fonts
#rcParams['font.family'] = 'sans-serif'
#rcParams['font.family'] = 'serif'
#rcParams['font.serif'] = ['Computer Modern Roman']
rcParams['text.usetex'] = True
rcParams['text.latex.preamble'] = [ r'\usepackage{amsmath}']
#rcParams['text.latex.preamble'] = [
#       r'\usepackage{siunitx}',   # i need upright \micro symbols, but you need...
#       r'\sisetup{detect-all}',   # ...this to force siunitx to actually use your fonts
#       r'\usepackage{helvet}',    # set the normal font here
#       r'\usepackage{sansmath}',  # load up the sansmath so that math -> helvet
#       r'\sansmath'               # <- tricky! -- gotta actually tell tex to use!
##]  

rcParams['axes.linewidth'] = 0.75
rcParams['lines.linewidth'] = 0.75

rcParams['xtick.major.size'] = 3      # major tick size in points
rcParams['xtick.minor.size'] = 2      # minor tick size in points
rcParams['xtick.major.width'] = 0.75       # major tick width in points
rcParams['xtick.minor.width'] = 0.75      # minor tick width in points

rcParams['ytick.major.size'] = 3      # major tick size in points
rcParams['ytick.minor.size'] = 2      # minor tick size in points
rcParams['ytick.major.width'] = 0.75       # major tick width in points
rcParams['ytick.minor.width'] = 0.75      # minor tick width in points




#%%

fig = plt.figure(figsize=(5.9/1.5,2.2*2))
#    
gs = GridSpec(2, 1)   
ax = plt.subplot(gs[0,0])

# uwavelock
date = 20180315
sequence = 87
scanned_parameter = 'ARP_FineBias'
df1 = utils.simple_data_processing(date, sequence, scanned_parameter)

sequence = 88
scanned_parameter = 'ARP_FineBias'
df2 = utils.simple_data_processing(date, sequence, scanned_parameter)

df = pd.concat([df1, df2], axis=0)
#df = df[1::]
p1 = df['od1']
p2 = df['od2']
#%%

delta = df[scanned_parameter]*1.44e3
om=0.1e1
params = lmfit.Parameters()
params.add('om', value=om, min=0)
params.add('t', value=2/om)
params.add('delta0', value=-5)
params.add('amp', value=300)
params.add('off', value=50)
minner = lmfit.Minimizer(utils.rabi_peak, params, 
                         fcn_args=(delta, p1), nan_policy='omit')
                    
result = minner.minimize('leastsq')
delta_resampled = np.linspace(delta.min(), delta.max(), 200)
peak = utils.rabi_peak(result.params, delta_resampled)
plt.plot(delta_resampled-delta.mean(), peak, 'k--')
plt.plot(delta-delta.mean(), p1, 'o', mec='k', ms=6, mfc='r', label='$\delta_+$')

#%%
params = lmfit.Parameters()
params.add('om', value=om, min=0)
params.add('t', value=2/om)
params.add('delta0', value=-7.5)
params.add('amp', value=300)
params.add('off', value=37)
minner = lmfit.Minimizer(utils.rabi_peak, params, 
                         fcn_args=(delta, p2), nan_policy='omit')
                    
result = minner.minimize('leastsq')
delta_resampled = np.linspace(delta.min()-1, delta.max()+1, 200)
peak2 = utils.rabi_peak(result.params, delta_resampled)
plt.plot(delta_resampled-delta.mean(), peak2, 'k--')
plt.plot(delta-delta.mean(), p2, 'o', mec='k', ms=6, mfc='b', label='$\delta_-$')

plt.xlabel('$\delta$ [kHz]')
plt.ylabel(r'$n$ [arb. u.]')
plt.xlim([delta.min()-delta.mean(), delta.max()-delta.mean()])
plt.legend()
plt.text(-8.03368421, 173, '$\mathbf{a.}$')
#plt.xlim([-30, 30])
#plt.ylim([0,1])
#ax.set_yticks([0, 0.5, 1])
#plt.text(-30, 1.05, '$\mathbf{a.}$')
#%%
ax = plt.subplot(gs[1,0])
# error signal


plt.plot(delta_resampled-delta.mean(), (peak-peak2)/(peak+peak2), '--', color='k')
plt.plot(delta-delta.mean(), df['err'], 'o', mec='k', ms=6, mfc='lightgray')
plt.xlim([delta.min()-delta.mean(), delta.max()-delta.mean()])
plt.xlabel('$\delta$ [kHz]')
plt.ylabel('$n_{\mathrm{imb}}$')
plt.tight_layout()
ax.set_yticks([-0.5, 0, 0.5])
plt.text(-8.03368421, 0.65, '$\mathbf{b.}$')
plt.savefig('uwave_lock.pdf')
#ax.set_yticklabels([])
#plt.ylim([0,1])
#plt.xlabel('Pulse time [$\mu$s]')
#plt.text(0, 1.05, '$\mathbf{b.}$')
#plt.tight_layout()
#plt.savefig('rabi_cycle.pdf')
plt.show()
#%%
om=0.5
delta = np.linspace(-2*om, 2*om, 500)
params = lmfit.Parameters()
params.add('om', value=om, min=0)
params.add('t', value=2/om)
params.add('delta0', value=-om*pi/2)
params.add('amp', value=100)
params.add('off', value=0)

peak = utils.rabi_peak(params, delta)
params.add('delta0', value=+om*pi/2)
peak2 = utils.rabi_peak(params, delta)

plt.plot(delta, peak)
plt.plot(delta, peak2)
plt.show()

delta_imb = np.sqrt((2*peak2/(peak+peak2)**2)**2 +(2*peak/(peak+peak2)**2)**2)*0.18*25*1
imb = (peak-peak2)/(peak+peak2)
plt.plot(delta, imb, 'k', linewidth=1.5)
plt.fill_between(delta, imb+delta_imb, imb-delta_imb, color='gray', alpha=0.5)
#plt.plot(delta, delta_imb+imb, '--')
#plt.plot(delta, imb-delta_imb, '--', color='gray')
plt.show()
#plt.plot(delta,delta_imb)


#%%




def zero_crossing(imb, delta):
    
    min_idx = np.argmin(np.abs(imb))
    
    return delta[min_idx]


amps = np.linspace(10, 300, 200)
om=0.5
delta = np.linspace(-2*om, 2*om, 500)
widths = []
for amp in amps:

    params.add('delta0', value=-om*pi/2)
    params.add('amp', value=amp)
    peak = utils.rabi_peak(params, delta)
    params.add('delta0', value=+om*pi/2)
    peak2 = utils.rabi_peak(params, delta)
    delta_imb = np.sqrt((2*peak2/(peak+peak2)**2)**2 +(2*peak/(peak+peak2)**2)**2)*0.18*25*1
    imb = (peak-peak2)/(peak+peak2)
    
    delta_0 = zero_crossing(imb, delta)
    delta_p = zero_crossing(imb+delta_imb, delta)
    width = delta_p - delta_0
    widths.append(width)

plt.plot(amps, widths, label='Omega={} Hz'.format(500))
plt.legend()

om=0.8
delta = np.linspace(-2*om, 2*om, 500)
widths = []
for amp in amps:

    params.add('delta0', value=-om*pi/2)
    params.add('amp', value=amp)
    peak = utils.rabi_peak(params, delta)
    params.add('delta0', value=+om*pi/2)
    peak2 = utils.rabi_peak(params, delta)
    delta_imb = np.sqrt((2*peak2/(peak+peak2)**2)**2 +(2*peak/(peak+peak2)**2)**2)*0.18*25*1
    imb = (peak-peak2)/(peak+peak2)
    
    delta_0 = zero_crossing(imb, delta)
    delta_p = zero_crossing(imb+delta_imb, delta)
    width = delta_p - delta_0
    widths.append(width)

plt.plot(amps, widths, label='Omega=900 Hz')
plt.legend()
#%%
y_diff_p = []
y_diff_m = []
for i in range(100,len(delta_imb)-100):
    y_diff = imb[i]-(imb+delta_imb)[i-70:i+70]
    min_idx = np.argmin(np.abs(y_diff))
    y_diff_p.append(y_diff[min_idx])
    y_diff = imb[i]-(imb-delta_imb)[i-70:i+70]
    min_idx = np.argmin(np.abs(y_diff))
    y_diff_m.append(y_diff[min_idx])

plt.plot(y_diff_p)
plt.plot(y_diff_m)


        
    
    