import numpy as np
import matplotlib as mpl
mpl.use("TkAgg")
import matplotlib.pyplot as plt

#q1

data = np.load("isotropic1024_slice.npz")
u = data["u"]
v = data["v"]
w = data["w"]
nx = 1024
dx = 2.0*np.pi/nx


U = np.sqrt(np.mean(u*u + v*v + w*w))
L = 1.346
nu = 0.000185


Re = U * L / nu
print("U")
print("Re")

#q2

k = 0.5 * (u*u + v*v + w*w)
k_mean = np.mean(k)
k_tilde = k / k_mean

subsection = k_tilde[:512, :512]
vmin = np.min(k_tilde)
vmax = np.max(k_tilde)

fig, ax = plt.subplots(figsize=(22, 18))
im1 = ax.pcolor(k_tilde, cmap='inferno', vmin=vmin, vmax=vmax)
ax.set_title(r'Normalized Kinetic Energy Field')
ax.set_xlabel('x')
ax.set_ylabel('y')
cbar1 = plt.colorbar(im1, ax=ax)
cbar1.set_label('Normalized Kinetic Energy')
plt.savefig('kinetic1.png', bbox_inches='tight', dpi=240)
plt.show()

vmin = np.min(subsection)
vmax = np.max(subsection)
fig, ax = plt.subplots(figsize=(22, 18))
ax.set_title(r'Normalized Kinetic Energy Field (512x512 Subsection)')
ax.set_xlabel('x')
ax.set_ylabel('y')
im2 = ax.pcolor(subsection, cmap='inferno', vmin=vmin, vmax=vmax)
cbar2 = plt.colorbar(im2, ax=ax)
cbar2.set_label('Normalized Kinetic Energy')
plt.savefig('kinetic2.png', bbox_inches='tight', dpi=240)
plt.show()

#q3

def ddx(f):
    return (np.roll(f, -1, axis=0) - np.roll(f, 1, axis =0 )) / (2*dx)

def ddy(f):
    return (np.roll(f, -1, axis =1) - np.roll(f, 1,axis =1 )) / (2*dx)

def ddz(f):
    # z axis data is not given, so we assume it is  constant
    return np.zeros_like(f)

ux = ddx(u)
uy = ddy(u)
uz = ddz(u)
vx = ddx(v)
vy = ddy(v)
vz = ddz(v)
wx = ddx(w)
wy = ddy(w)
wz = ddz(w)

Sxx = ux
Syy = vy
Szz = wz
Sxy = 0.5 * (uy+vx)
Sxz = 0.5 * (uz+wx)
Syz = 0.5 * (vz+wy)

SijSij = (
    Sxx**2 +
    Syy**2 + Szz**2 +
    2.0*Sxy**2 +
    2.0*Sxz**2 +
    2.0*Syz**2
)

epsilon = 2.0 * nu * SijSij
epsilon_mean = np.mean(epsilon)
epsilon_rms = np.sqrt(np.mean((epsilon - epsilon_mean)**2))
epsilon_tilde = epsilon / epsilon_mean

subsection = epsilon_tilde[:512, :512]

vmin = np.min(epsilon_tilde)

fig, ax = plt.subplots(figsize=(22, 18))
ax.set_title(r'Normalized Dissipation Field')
ax.set_xlabel('x')
ax.set_ylabel('y')
im1 = ax.pcolor(epsilon_tilde, cmap='inferno', vmin=vmin, vmax=25*epsilon_rms)
cbar1 = plt.colorbar(im1, ax=ax)
cbar1.set_label('Normalized Dissipation')
plt.savefig('dissip1.png', bbox_inches='tight', dpi=240)
plt.show()

vmin = np.min(subsection)
fig, ax = plt.subplots(figsize=(22, 18))
ax.set_title(r'Normalized Dissipation Field')
ax.set_xlabel('x')
ax.set_ylabel('y')
im2 = ax.pcolor(subsection, cmap='inferno', vmin=vmin,vmax=25*epsilon_rms)
cbar2 = plt.colorbar(im2, ax=ax)
cbar2.set_label('Normalized Dissipation')
plt.savefig('dissip2.png', bbox_inches='tight', dpi=240)
plt.show()

#q4

omega_z = vx - uy
omega_rms = np.sqrt(np.mean(omega_z**2))
omega_tilde = omega_z / omega_rms
val=max(abs(np.min(omega_tilde)), abs(np.max(omega_tilde)))
subsection = omega_tilde[:512, :512]

fig, ax = plt.subplots(figsize=(22, 18))
ax.set_title(r'Vorticity Field')
ax.set_xlabel('x')
ax.set_ylabel('y')
im1 = ax.pcolor(omega_z, cmap='inferno', vmin=-0.2*val,vmax=0.2*val)
cbar1 = plt.colorbar(im1, ax=ax)
cbar1.set_label('Vorticity')
plt.savefig('omega1.png', bbox_inches='tight', dpi=240)
plt.show()
fig, ax = plt.subplots(figsize=(22, 18))
ax.set_title(r'Normalized Vorticity Field')
ax.set_xlabel('x')
ax.set_ylabel('y')
im2 = ax.pcolor(subsection, cmap='inferno', vmin=-0.2*val,vmax=0.2*val)
cbar2 = plt.colorbar(im2, ax=ax)
cbar2.set_label('Normalized Vorticity')
plt.savefig('omega2.png', bbox_inches='tight', dpi=240)
plt.show()


#q5

def pdf(f, bins=3000, label=None):

    phi = f.ravel()
    mean = np.mean(phi)
    std = np.std(phi)

    phi_hat = (phi - mean)
    skewness = np.mean(phi_hat**3)/std**3
    kurtosis = np.mean(phi_hat**4)/std**4
    print(f"skewness: {skewness}")
    print(f"kurtosis: {kurtosis}")

    hist, bin_edges = np.histogram(phi_hat, bins)
    delt = bin_edges[1] - bin_edges[0]
    pdf = hist / (len(phi_hat) * delt)
    centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    pdf=pdf * std
    plt.semilogy(centers/std, pdf, markersize=4, label=label)
    
    #gaussian
    xg = np.linspace(centers[0], centers[-1], bins)
    gauss = (1/np.sqrt(2*np.pi)) * np.exp(-0.5*xg**2)
    plt.semilogy(xg, gauss, label='Gaussian for ' + (label if label else 'data'))
    
    return 

#5a

plt.figure(figsize=(16,15))
pdf(u, label='u')
pdf(v, label='v')
pdf(w, label='w')
plt.xlabel(r'$u_i/\sigma$')
plt.ylabel(r'$p(u_i)\sigma$')
plt.legend()
plt.grid(True, which='both')
plt.savefig('velocity_pdf.png', bbox_inches='tight', dpi=240)
plt.show()


#5b

plt.figure(figsize=(16,15))
pdf(omega_z, label='$\omega_z$')
plt.xlabel(r'$\omega_z/\sigma$')
plt.ylabel(r'$p(\omega_z)\sigma$')
plt.legend()
plt.grid(True, which='both')
plt.xlim(-30, 30) #only plot within 30stds as there are few data points beyond this that make the features unclear
plt.savefig('vorticity_pdf.png', bbox_inches='tight', dpi=240)
plt.show()

#5c

plt.figure(figsize=(16,15))
pdf(ux, label='$\delta u / \delta x$')
plt.xlabel(r'$\delta u / \delta x/\sigma$')
plt.ylabel(r'$p(\delta u / \delta x)\sigma$')
plt.legend()
plt.grid(True, which='both')
plt.xlim(-30, 30)
plt.savefig('ux_pdf.png', bbox_inches='tight', dpi=240)
plt.show()

plt.figure(figsize=(16,15))
pdf(vy, label='$\delta v / \delta y$')
plt.xlabel(r'$\delta v / \delta y/\sigma$')
plt.ylabel(r'$p(\delta v / \delta y)\sigma$')
plt.legend()
plt.grid(True, which='both')
plt.xlim(-30, 30)
plt.savefig('vy_pdf.png', bbox_inches='tight', dpi=240)
plt.show()

#5d

plt.figure(figsize=(16,15))
pdf(omega_z**2, label='${\omega_z}^2$')
plt.xlabel(r'${\omega_z}^2/\sigma$')
plt.ylabel(r'$p({\omega_z}^2)\sigma$')
plt.legend()
plt.grid(True, which='both')
plt.xlim(0, 15)
plt.savefig('enstrophy_pdf.png', bbox_inches='tight', dpi=240)
plt.show()

#q6

def laplacian_5pt(f):
    return (
        np.roll(f, -1, axis=0) + np.roll(f,  1, axis=0) + np.roll(f, -1, axis=1) + np.roll(f,  1, axis=1) - 4.0 * f) / (dx**2)

N_lin = u * ux + v * uy
Visc = nu * laplacian_5pt(u)

Re_local = np.abs(N_lin) / (np.abs(Visc))

sub_k   = k_tilde[:512, :512]
sub_eps = epsilon_tilde[:512, :512]
sub_Re  = Re_local[:512, :512]

vmin_k, vmax_k = np.min(sub_k), np.max(sub_k)
vmin_eps, vmax_eps = np.min(sub_eps), 25.0 * epsilon_rms

fig, axes = plt.subplots(1, 3, figsize=(44, 12))

im0 = axes[0].pcolormesh(sub_k, cmap='inferno', vmin=vmin_k, vmax=vmax_k)
axes[0].set_title(r'Normalized Kinetic Energy Field $k$')
axes[0].axis('off')
fig.colorbar(im0, ax=axes[0], label='k')

im1 = axes[1].pcolormesh(sub_eps, cmap='inferno', vmin=vmin_eps, vmax=vmax_eps)
axes[1].set_title(r'Normalized Dissipation Field $\epsilon$')
axes[1].axis('off')
fig.colorbar(im1, ax=axes[1], label=r'$\epsilon$')

im2 = axes[2].pcolormesh(sub_Re, cmap='inferno', vmin=-50, vmax=50) #was not able to determine suitable vmin/vmax by scaling with rms etc., so set to arbitrary values for better visualization
axes[2].set_title(r'Local Reynolds number $Re$')
axes[2].axis('off')
fig.colorbar(im2, ax=axes[2], label='Re_local')

plt.tight_layout()
plt.savefig('fields_side_by_side.png', bbox_inches='tight', dpi=240)
plt.show()

def correlation(a, b):
    a0 = a.flatten() - np.mean(a)
    b0 = b.flatten() - np.mean(b)
    return np.mean(a0 * b0) / (np.std(a) * np.std(b))

rho_k_Re = correlation(k, Re_local)
rho_eps_Re = correlation(epsilon, Re_local)
rho_k_eps = correlation(k, epsilon)

print("Correlation coefficients:")
print("rho(k, Re)   =", rho_k_Re)
print("rho(eps, Re) =", rho_eps_Re)
print("rho(k, eps)  =", rho_k_eps)
