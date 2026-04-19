import numpy as np
import matplotlib as mpl
mpl.use("TkAgg")
import matplotlib.pyplot as plt


data = np.load("isotropic1024_slice.npz")
u = data["u"]
v = data["v"]
w = data["w"]

nx = 1024
dx = 2.0 * np.pi / nx

U = np.sqrt(np.mean(u**2 + v**2 + w**2))
L = 1.346
nu = 0.000185


#q1

u_line = u[:,0] 
en_physical = np.sum(np.abs(u_line) ** 2) # total energy in physical space

u_line_fourier = np.fft.fft(u_line)
en_spectral = np.sum(np.abs(u_line_fourier) ** 2) / nx # spectral space (normalized by N)

print(f"Total energy in physical space: {en_physical}")
print(f"Total energy in spectral space: {en_spectral}")

#q2

uhat = np.fft.fft(u, axis=0) #DFT in x axis only
vhat = np.fft.fft(v, axis=0)

k = 2 * np.pi * np.fft.fftfreq(nx, dx) # wavenumbers
E = (np.abs(uhat)**2 + np.abs(vhat)**2) / 2 
Ek = np.mean(E, axis=1) #mean along y axis

positive_k = np.arange(0, nx//2 + 1)  
k_plus = k[positive_k] # energy spectrum only cares about magnitude

Ek_plus = Ek[positive_k]
Ek_plus[1:-1] *= 2  # Double for k=1 to k=nx//2-1 to account for negative frequencies

fig, ax = plt.subplots(figsize=(20, 18))
ax.loglog(k_plus, Ek_plus, 'b-', linewidth=2, label='Energy Spectrum') # plot spectrum

mink = 0.01
maxk = 30 #inertial subrange comes from hit and trial
inertialsubrange = (k_plus > mink) & (k_plus < maxk)
k_inertial = k_plus[inertialsubrange]
Ek_inertial = Ek_plus[inertialsubrange] # for polyfit

logk = np.log(k_inertial)
logE = np.log(Ek_inertial)
coeffs = np.polyfit(logk, logE, 1) #fitting
    
k_fit = np.linspace(k_inertial[0], k_inertial[-1], 100)
E_fit = np.exp(coeffs[1] + coeffs[0] * np.log(k_fit))
    
ax.loglog(k_fit, E_fit, 'r--', linewidth=4, label=f'Fit: k^({coeffs[0]:.3f})') #plot fit
    
print(f"Inertial range: k = [{mink}, {maxk}]")
print(f"Fitted scaling exponent: {coeffs[0]}")
print(f"Expected scaling exponent: {-5/3}")

ax.set_xlabel('k', fontsize=30)
ax.set_ylabel('E(k)', fontsize=30)
ax.set_title('Energy Spectrum (1D)', fontsize=30)
ax.grid(True, which='both', alpha=0.3)
ax.set_xlim([1e-3, 1e3])
ax.set_ylim([1e-9, 1e6])

plt.savefig('energy_spectrum.png', bbox_inches='tight', dpi=240)
plt.show()

#q3

uhat_2d = np.fft.fftn(u)
vhat_2d = np.fft.fftn(v)
what_2d = np.fft.fftn(w) #2D DFT of all velocity components

E_2d = (np.abs(uhat_2d)**2 + np.abs(vhat_2d)**2 + np.abs(what_2d)**2) / 2 #include w 
E_2d_shifted = np.fft.fftshift(E_2d) # energy spectrum in 2D wavenumber space (shifted to center zero frequency)

kx = 2 * np.pi * np.fft.fftfreq(nx, dx)
ky = 2 * np.pi * np.fft.fftfreq(nx, dx)
dk = kx[1] - kx[0]  # Actual wavenumber spacing from FFT resolution
kx_shifted = np.fft.fftshift(kx)
ky_shifted = np.fft.fftshift(ky) # shifted wavenumber arrays for 2D spectrum
# spectral field
Kx, Ky = np.meshgrid(kx_shifted, ky_shifted, indexing='ij') 
K = np.sqrt(Kx**2 + Ky**2) 

k_flat = K.flatten()
E_flat = E_2d_shifted.flatten()

k_max = np.ceil(np.max(k_flat)).astype(int) # maximum wavenumber for binning
k_bins = np.arange(-0.5*dk, k_max + 0.5*dk, dk) #shell-averaging: k-dk/2 <= k < k+dk/2

Ek_2d = np.zeros(len(k_bins) - 1)
k_2d_centers = np.zeros(len(k_bins) - 1)

for i in range(len(k_bins) - 1):
    mask = (k_flat >= k_bins[i]) & (k_flat < k_bins[i+1])
    if np.sum(mask) > 0:
        Ek_2d[i] = np.mean(E_flat[mask])
        k_2d_centers[i] = k_bins[i] + 0.5*dk #center at proper wavenumber spacing

valid_2d = k_2d_centers > 0
k_2d_centers = k_2d_centers[valid_2d]
Ek_2d = Ek_2d[valid_2d]

fig, ax = plt.subplots(figsize=(20, 18))
ax.loglog(k_plus, Ek_plus, 'b-', linewidth=2, label='1D Spectrum', alpha=0.7)
ax.loglog(k_2d_centers, Ek_2d, 'g-', linewidth=2, label='2D Spectrum', alpha=0.7)

mink_2d = 0.5
maxk_2d = 30
inertial_idx_2d = (k_2d_centers > mink_2d) & (k_2d_centers < maxk_2d)
k_inertial_2d = k_2d_centers[inertial_idx_2d]
E_inertial_2d = Ek_2d[inertial_idx_2d]

log_k_2d = np.log(k_inertial_2d)
log_E_2d = np.log(E_inertial_2d)
coeffs_2d = np.polyfit(log_k_2d, log_E_2d, 1)
k_fit_2d = np.linspace(k_inertial_2d[0], k_inertial_2d[-1], 100)
E_fit_2d = np.exp(coeffs_2d[1] + coeffs_2d[0] * np.log(k_fit_2d))
    
ax.loglog(k_fit_2d, E_fit_2d, 'r--', linewidth=4, label=f'2D Fit: k^({coeffs_2d[0]:.3f})')

print(f"Inertial range: k = [{mink_2d}, {maxk_2d}]")
print(f"Fitted exponent (2D): {coeffs_2d[0]:.4f}")
print(f"Fitted exponent (1D): {coeffs[0]:.4f}")
print(f"Expected: {-5/3:.4f}")

ax.set_xlabel('k = $\sqrt{k_x^2 + k_y^2}$', fontsize=20)
ax.set_ylabel('E(k)', fontsize=20)
ax.set_title('Energy Spectrum: 1D vs 2D', fontsize=20)
ax.grid(True, which='both', alpha=0.3)
ax.legend(fontsize=16)
ax.set_xlim([1e-3, 1e3])
ax.set_ylim([1e-9, 1e12])

plt.savefig('energy_spectrum_2d_comparison.png', bbox_inches='tight', dpi=240)
plt.show()



#q4 

r_max = nx // 2
r_distances = np.arange(1, r_max) * dx

#Longitudinal correlation 
R_L = np.zeros(len(r_distances))
for i, r_sep in enumerate(r_distances):
    r_int = int(np.round(r_sep / dx))
    du_prod = np.mean(u[r_int:, :] * u[:-r_int, :])
    R_L[i] = du_prod / np.mean(u**2)

#Transverse correlation
R_T = np.zeros(len(r_distances))
for i, r_sep in enumerate(r_distances):
    r_int = int(np.round(r_sep / dx))
    dv_prod = np.mean(v[r_int:, :] * v[:-r_int, :])
    R_T[i] = dv_prod / np.mean(v**2)

fig, ax = plt.subplots(figsize=(22, 18))
ax.plot(r_distances, R_L, 'b-', linewidth=2.5, label='Longitudinal $R_L(r)$')
ax.plot(r_distances, R_T, 'r-', linewidth=2.5, label='Transverse $R_T(r)$')
ax.axhline(y=0, color='k', linestyle='--', linewidth=0.8, alpha=0.5)
ax.set_xlabel('Separation distance r', fontsize=30)
ax.set_ylabel('Correlation coefficient R(r)', fontsize=30)
ax.set_title('Longitudinal and Transverse Velocity Correlation Functions', fontsize=30)
ax.grid(True, alpha=0.3)
ax.legend(fontsize=30)
ax.set_xlim([r_distances[0], r_distances[-1]])
plt.savefig('correlation_functions.png', bbox_inches='tight', dpi=240)
plt.show()

#q5 
p_values = np.array([1, 2, 3, 4, 5, 6, 7])

r_max = int(L / (2 * dx))
r_array = np.arange(1, r_max) * dx

S_p = {}
for p in p_values:
    S_p[p] = np.zeros(len(r_array))

# Calculate structure functions
for i, r_sep in enumerate(r_array):
    r_int = int(np.round(r_sep / dx))
    # For each order p, calculate the p-th moment
    for p in p_values:
        sum_val = 0
        count = 0
        for j in range(len(u) - r_int):
            for k in range(len(u[0])):
                # Get velocity difference at this separation
                delta_u = u[j + r_int, k] - u[j, k]
                sum_val += np.abs(delta_u)**p
                count += 1
        S_p[p][i] = sum_val / count

valid = np.any([S_p[p] == 0 for p in p_values], axis=0)
r_array = r_array[valid]
for p in p_values:
    S_p[p] = S_p[p][valid]

fig, ax = plt.subplots(figsize=(22, 18))
colors = plt.cm.rainbow(np.linspace(0, 1, len(p_values)))
for i, p in enumerate(p_values):
    ax.loglog(r_array, S_p[p], 'o-', color=colors[i], linewidth=2.5, markersize=4, label=f'$S_{{{p}}}(r)$')
ax.set_xlabel('Separation r', fontsize=30)
ax.set_ylabel('$S_p(r)$', fontsize=30)
ax.set_title('Longitudinal Structure Functions', fontsize=30)
ax.grid(True, which='both', alpha=0.3)
ax.legend(fontsize=30, ncol=2)
ax.set_xlim([r_array[0], r_array[-1]])
plt.savefig('structure_functions_sp.png', bbox_inches='tight', dpi=240)
plt.show()

# (b)

epsilon_approx = U**3 / L
kolmogorov_45_law = -4/5 * epsilon_approx * r_array

inertial_range_45 = (r_array > 0.05) & (r_array < L/4)

if np.sum(inertial_range_45) > 2:
    S3_inertial = S_p[3][inertial_range_45]
    kolmogorov_inertial = kolmogorov_45_law[inertial_range_45]
    rel_error = np.mean(np.abs(S3_inertial - kolmogorov_inertial) / np.abs(kolmogorov_inertial))
else:
    rel_error = np.nan

fig, ax = plt.subplots(figsize=(14, 10))
ax.loglog(r_array, S_p[3], 'bo-', linewidth=2.5, markersize=6, label='$S_3(r)$ (our data)', markerfacecolor='none')
ax.loglog(r_array, -kolmogorov_45_law, 'r--', linewidth=3, label="Kolmogorov 4/5th law: $-\\frac{4}{5}\\epsilon r$")
ax.set_xlabel('Separation r', fontsize=30)
ax.set_ylabel('$S_3(r)$', fontsize=30)
ax.set_title('Kolmogorov 4/5th Law Verification', fontsize=30)
ax.grid(True, which='both', alpha=0.3)
ax.legend(fontsize=30)
plt.savefig('kolmogorov_45law.png', bbox_inches='tight', dpi=240)
plt.show()

# (c,d) 
zeta_p = np.zeros(len(p_values)) 
zeta_p_err = np.zeros(len(p_values))  

fig, ax = plt.subplots(figsize=(22, 18))
for i, p in enumerate(p_values):
    # Plot S_p vs S_3 
    ax.loglog(S_p[3], S_p[p], 'o-', color=colors[i], linewidth=2.5, markersize=4, label=f'$S_{{{p}}}$')
   
    log_S3 = np.log(S_p[3])
    log_Sp = np.log(S_p[p])
    
    
    n_data = len(log_S3)
    fit_idx = np.arange(int(0.2*n_data), int(0.8*n_data)) #(ignore edges)
    
    
    coeffs_ess = np.polyfit(log_S3[fit_idx], log_Sp[fit_idx], 1, cov=True)
    slope = coeffs_ess[0][0]
    intercept = coeffs_ess[0][1]
    cov_matrix = coeffs_ess[1]
    slope_err = np.sqrt(cov_matrix[0, 0])
        
    zeta_p[i] = slope
    zeta_p_err[i] = slope_err
        
    S3_fit = np.logspace(np.log10(S_p[3][fit_idx[0]]), np.log10(S_p[3][fit_idx[-1]]), 100)
    Sp_fit = np.exp(intercept + slope * np.log(S3_fit))
    ax.loglog(S3_fit, Sp_fit, '--', color=colors[i], linewidth=2, alpha=0.7)

ax.set_xlabel('$S_3(r)$', fontsize=30)
ax.set_ylabel('$S_p(r)$', fontsize=30)
ax.set_title('Extended Self-Similarity (ESS)', fontsize=30)
ax.grid(True, which='both', alpha=0.3)
ax.legend(fontsize=30, ncol=2)
plt.savefig('structure_functions_ess.png', bbox_inches='tight', dpi=240)
plt.show()

kolmogorov_pred = p_values / 3

fig, ax = plt.subplots(figsize=(12, 8))
ax.errorbar(p_values, zeta_p, yerr=zeta_p_err, fmt='bo-', linewidth=2.5, markersize=8, 
            capsize=5, capthick=2, label='Our measurements', markerfacecolor='none')
ax.plot(p_values, kolmogorov_pred, 'r--', linewidth=2.5, label='Simple theory: $\\zeta_p = p/3$')
ax.set_xlabel('Order p', fontsize=14)
ax.set_ylabel('Scaling exponent $\\zeta_p$', fontsize=14)
ax.set_title('Do We See Anomalous Scaling?', fontsize=16)
ax.grid(True, alpha=0.3)
ax.legend(fontsize=12)
plt.savefig('scaling_exponents_zeta_p.png', bbox_inches='tight', dpi=240)
plt.show()

print("\n=== Q5(d) - Our Measured Scaling Exponents ===")
print("p\tOur ζ_p\t\tUncertainty\tTheory ζ_p\tHow different?")
for i, p in enumerate(p_values):
    dev = zeta_p[i] - p/3
    print(f"{p}\t{zeta_p[i]:.4f}\t\t{zeta_p_err[i]:.4f}\t\t{p/3:.4f}\t\t{dev:.4f}")

