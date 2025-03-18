# -*- coding: utf-8 -*-
"""
Created on Tue Mar 18 15:22:16 2025

@author: MONDAL
"""

import numpy as np
import scipy.integrate as spi
import scipy.optimize as spo
import matplotlib.pyplot as plt

# Given temperature (T) and volume (V(T)) data
T_values = np.array([270, 240, 210, 170, 130, 90, 70, 50, 30, 20])
V_values = np.array([587.613, 587.246, 586.94, 586.649, 586.416, 586.134, 
                      586.107, 586.055, 586.083, 586.065])

# Constants
N = 24  # Number of atoms in unit cell
k_B = 1.38e7  # Boltzmann constant converted to angstrom units
Nkb = N * k_B  # Effective thermal constant

# Define the Debye integral
def debye_integral(x):
    return (x**3) / (np.exp(x) - 1)

# Compute U(T) based on Debye model
def U_T(T, theta_D):
    integral, _ = spi.quad(debye_integral, 0, theta_D / T)
    return 9 * Nkb * T * ((T / theta_D)**3) * integral

# Model function for V(T)
def V_model(T, theta_D, gamma_B0, V0):
    U_values = np.array([U_T(Ti, theta_D) for Ti in T])  # Compute U(T) for all T
    return gamma_B0 * U_values + V0

# Initial guess for parameters: theta_D, gamma/B0, V0
initial_guess = [200, 1e-10, 586.15]  

# Perform curve fitting
params_opt, params_cov = spo.curve_fit(V_model, T_values, V_values, p0=initial_guess)

# Extract optimized parameters
theta_D_fit, gamma_B0_fit, V0_fit = params_opt

# Generate fitted curve
T_fit = np.linspace(min(T_values), max(T_values), 100)
V_fit = V_model(T_fit, theta_D_fit, gamma_B0_fit, V0_fit)

# Plot results
plt.figure(figsize=(8, 6))
plt.scatter(T_values, V_values, label="Data", color="red")
plt.plot(T_fit, V_fit, label=f"Fit: $\\theta_D$={theta_D_fit:.2f}, $\\gamma/B_0$={gamma_B0_fit:.6e}, $V_0$={V0_fit:.3f}", color="blue")
plt.xlabel("Temperature (T)")
plt.ylabel("Volume V(T)")
plt.legend()
plt.title("Fitting V(T) vs. T")
plt.grid()
plt.show()

# Display fitted parameters
print(f"Debye temperature (theta_D): {theta_D_fit:.2f} K")
print(f"Gamma/B0 ratio: {gamma_B0_fit:.6e}")
print(f"Reference volume (V0): {V0_fit:.3f}")