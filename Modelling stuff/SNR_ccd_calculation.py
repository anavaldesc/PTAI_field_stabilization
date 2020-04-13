# -*- coding: utf-8 -*-
"""
Created on Tue Jan 17 13:46:02 2017

@author: Francisco Salces 
"""

from __future__ import division
import numpy as np
from matplotlib import pyplot as plt

# Constants
planck = 6.626e-34 # Plancks constant
Isat = 1.67e-3     # Rb87
magnification = 5.33
frequency = 3e8/780.24e-9

FLEA3 =    { 'Quantum Efficiency': 0.3,
             'Max Well Depth'    : 23035.06, #e-
             'Pixel Size'        : 5.6e-4, #cm^2
             'Read Noise'        : 38.74, #e-
             'Dark Current'      : 307.92, #e-/s
                                                 }
                                             
ZYLA_5p5 = { 'Quantum Efficiency': 0.35,
             'Max Well Depth'    : 30000, #e-
             'Pixel Size'        : 6.5e-4, #cm^2
             'Read Noise'        : 2.7, #e-
             'Dark Current'      : 0.01, #e-/s
                                                 }
                                             
PRIME95B = { 'Quantum Efficiency': 0.5,
             'Max Well Depth'    : 80000, #e-
             'Pixel Size'        : 11e-4, #cm^2
             'Read Noise'        : 1.8, #e-
             'Dark Current'      : 0.1, #e-/s
                                                 }  

Mako = { 'Quantum Efficiency': 0.3,
         'Max Well Depth'    : 9300, #e-
         'Pixel Size'        : 7.4e-4, #cm^2
         'Read Noise'        : 13.4, #e-
         'Dark Current'      : 12.9 #e-/s
                                                 }              

#Mako isat in counts per pixel Isat = 3.550505e3                                                                            
                                             
def SNR_calculator(target_OD, pulse_duration, CCD_attrs, IoverIsat):
    qe = CCD_attrs['Quantum Efficiency']
    pixel_area = (CCD_attrs['Pixel Size']/magnification)**2
    single_photon_energy = planck*frequency
    N_sat = (qe*pixel_area*pulse_duration*Isat)/(single_photon_energy)
    N_probe = N_sat*IoverIsat
    N_atoms = N_probe*np.exp(-target_OD)
    N_back = 0.0
    sigma_Na = np.sqrt(N_atoms)
    sigma_Np = np.sqrt(N_probe)
    sigma_Nb = np.sqrt(CCD_attrs['Read Noise']**2 + (CCD_attrs['Dark Current']*pulse_duration)**2)    
    partial_Na = -1/N_sat - 1/(N_atoms-N_back)
    partial_Np = 1/N_sat + 1/(N_probe-N_back)
    partial_Nb = (N_probe-N_atoms)/((N_atoms-N_back)*(N_probe-N_back))   
    sigma_OD = np.sqrt(partial_Na**2*sigma_Na**2+partial_Np**2*sigma_Np**2+partial_Nb**2*sigma_Nb**2)                    
    signal_to_noise = target_OD/sigma_OD    
    signal_to_noise[0] = 0 # avoid NaN, should be zero nevertheless
    print('Pixel saturates at I/Isat = %.2f' %(CCD_attrs['Max Well Depth']/N_sat))
    return signal_to_noise

intensities = np.linspace(1e-6, 5., 2**10)
OD = 0.15
#signaltonoise =SNR_calculator(OD, 15e-6, PRIME95B, intensities)
#signaltonoise =SNR_calculator(OD, 20e-6, FLEA3, intensities)
signaltonoise =SNR_calculator(OD, 20e-6, Mako, intensities)


fig = plt.figure(num = 1, frameon=False)
plt.plot(intensities, signaltonoise, label='Makos')
plt.xlabel('$I/I_{sat}$', fontsize=16)
plt.ylabel('Signal To Noise Ratio', fontsize=16)
plt.xlim(np.amin(intensities), np.amax(intensities))
plt.tight_layout()
plt.legend()
plt.show()



