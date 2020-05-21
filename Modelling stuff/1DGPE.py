from __future__ import division, print_function
import matplotlib as mpl
mpl.use("QT4Agg")
import matplotlib.pyplot as plt

import numpy as np


# Constants:
pi = np.pi
hbar = 1.054571726e-34                        # Reduced Planck's constant
a_0  = 5.29177209e-11                         # Bohr radius
u    = 1.660539e-27                           # unified atomic mass unit
m  = 86.909180*u                              # 87Rb atomic mass
a  = 98.98*a_0                                # 87Rb |2,2> scattering length
g  = 4*pi*hbar**2*a/m                         # 87Rb self interaction constant
omega = 15                                    # Trap frequency
rhomax = 2.49e12 * 1e6                         # Desired peak condensate density

# Space:
nx = 256
x_max = 100e-6
x = np.linspace(-x_max, x_max, nx, endpoint=False)
dx = x[1] - x[0]

# Fourier space:
k = 2*pi*np.fft.fftfreq(nx, d=dx)

# Wavenumber of nyquist mode - shortest wavenumber we can resolve:
k_nyquist = pi/dx

# Phase velocity of the nyquist mode - highest speed we can resolve
v_max = hbar * k_nyquist / m

# The time it takes the nyquist mode to move one gridpoint. This sets an upper
# limit for our simulation timestep size - a timestep any larger and we would
# not be able to resolve the motion of the nyquist mode:
dt_max = dx / v_max


# Oscillator period and oscillator length:
T_osc = 2*pi / omega
l_osc = np.sqrt(hbar/(m*omega))


# A harmonic trap:
V = 0.5 * m * omega**2 *x**2

# The kinetic energy operator in Fourier space
K = hbar**2 * k**2/(2*m)

def split_step2(psi, dt):

    """"Evolve psi in time from t to t + dt using one
    step of the second order Fourier split-step method"""

    psi *= np.exp(-1j/hbar * (V + g*np.abs(psi)**2) * dt/2)

    f_psi = np.fft.fft(psi)
    f_psi *= np.exp(-1j/hbar * K * dt)
    psi = np.fft.ifft(f_psi)

    psi *= np.exp(-1j/hbar * (V + g*np.abs(psi)**2) * dt/2)

    return psi

def split_step2_imag(psi, dt):

    """"Evolve psi in imaginary time from t to t + dt using one
    step of the second order Fourier split-step method"""

    psi *= np.exp(-1/hbar * (V + g*np.abs(psi)**2) * dt/2)

    f_psi = np.fft.fft(psi)
    f_psi *= np.exp(-1/hbar * K * dt)
    psi = np.fft.ifft(f_psi)
    normalise(psi)

    psi *= np.exp(-1/hbar * (V + g*np.abs(psi)**2) * dt/2)
    normalise(psi)

    return psi


def normalise(psi):
    """Normalise psi to the desired maximum density"""
    psi[:] *= np.sqrt(rhomax/np.abs(psi**2).max())


def find_groundstate():

    # Find the groundstate, subject to given peak density:
    # Initial guess: flat density:
    psi = np.sqrt(rhomax) * np.ones(nx, dtype=complex)

    # Evolve in imaginary time for one oscillator period to get groundstate:
    t = 0
    dt = 10 * dt_max # Can take bigger steps for imaginary time evolution

    i = 0
    while t < T_osc:
        psi = split_step2_imag(psi, dt)
        t += dt
        i += 1

        if i % 100 == 0:
            # Print time and plot the density once every 100 timesteps: 
            print('t/T_osc = ', round(t/T_osc, 3))
            #plt.ion()
            plt.plot(x*1e6, np.abs(psi)**2)
            plt.savefig('000%d.png' %i)
            plt.axis([-x_max*1e6, x_max*1e6, 0, rhomax])
            plt.xlabel('x (um)')
            #plt.draw()
            plt.clf()

    return psi


def evolve(psi):
    # Evolve in time for 10 oscillator periods:
    t = 0
    dt = dt_max / 8

    i = 0
    while t < 10 * T_osc:
        psi = split_step2(psi, dt)
        t += dt
        i += 1

        if i % 1000 == 0:
            # Print and plot the density once every 100 timesteps: 
            print('t/T_osc = ', round(t/T_osc, 3))
            #plt.ion()
            plt.plot(x*1e6, np.abs(psi)**2)
            plt.savefig('E000%d.png' %i)
            plt.axis([-x_max*1e6, x_max*1e6, 0, rhomax])
            plt.axvline(l_osc*1e6, color='k')
            #plt.draw()
            plt.clf()
    return psi


if __name__ == '__main__':
    psi = find_groundstate()

    # Displace the potential in space to induce sloshing:
    x_displacement = l_osc
    V = 0.5 * m * omega**2 * (x - x_displacement)**2

    psi = evolve(psi)