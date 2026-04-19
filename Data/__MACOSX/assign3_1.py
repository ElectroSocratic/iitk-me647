import numpy as np
import matplotlib as mpl
mpl.use("TkAgg")
import matplotlib.pyplot as plt

data = np.load("isotropic1024_stack3.npz")
u = data["u"]
v = data["v"]
w = data["w"]
u_mid = u[:, :, 1]
v_mid = v[:, :, 1]
w_mid = w[:, :, 1]
nx = u.shape[0]
ny = u.shape[1]
dx = 2.0 * np.pi / nx
dz = 1.0

def ddx(f):
    return (np.roll(f, -1, axis=0) - np.roll(f, 1, axis=0)) / (2 * dx)

def ddy(f):
    return (np.roll(f, -1, axis=1) - np.roll(f, 1, axis=1)) / (2 * dx)

def ddz_mid(f_zm1, f_zp1):
    return (f_zp1 - f_zm1) / (2 * dz)

#q1.2
ux = ddx(u_mid)
uy = ddy(u_mid)
uz = ddz_mid(u[:, :, 0], u[:, :, 2])
vx = ddx(v_mid)
vy = ddy(v_mid)
vz = ddz_mid(v[:, :, 0] ,v[:, :, 2])
wx = ddx(w_mid)
wy = ddy(w_mid)
wz = ddz_mid(w[:, :, 0], w[:, :, 2])

lambda1 = np.zeros((nx, ny))
lambda2 = np.zeros((nx, ny))
lambda3 = np.zeros((nx, ny))

for i in range(nx):
    for j in range(ny):
        A = np.array([[ux[i, j], uy[i, j], uz[i, j]],
                      [vx[i, j], vy[i, j], vz[i, j]],
                      [wx[i, j], wy[i, j], wz[i, j]]])
        evals = np.linalg.eigvals(A) # get eigenvalues
        # Sort for ordering highest to lowest
        evals_sorted = np.sort(evals)[::-1]
        lambda1[i, j] = evals_sorted[0]
        lambda2[i, j] = evals_sorted[1]
        lambda3[i, j] = evals_sorted[2]

# Plot PDFs 
fig, axes = plt.subplots(1, 3, figsize=(15, 4))

# Lambda 1
bins1 = np.linspace(np.min(lambda1), np.max(lambda1), 100)
counts1, _, _ = axes[0].hist(lambda1.flatten(), bins=bins1, density=True)
axes[0].set_yscale('log')
axes[0].set_xlabel(r'$\lambda_1$')
axes[0].set_ylabel('PDF')
axes[0].set_title(r'PDF of $\lambda_1$')
axes[0].grid(True, alpha=0.3)

# Lambda 2
bins2 = np.linspace(np.min(lambda2), np.max(lambda2), 100)
counts2, _, _ = axes[1].hist(lambda2.flatten(), bins=bins2, density=True)
axes[1].set_yscale('log')
axes[1].set_xlabel(r'$\lambda_2$')
axes[1].set_ylabel('PDF')
axes[1].set_title(r'PDF of $\lambda_2$')
axes[1].grid(True, alpha=0.3)

# Lambda 3
bins3 = np.linspace(np.min(lambda3), np.max(lambda3), 100)
counts3, _, _ = axes[2].hist(lambda3.flatten(), bins=bins3, density=True)
axes[2].set_yscale('log')
axes[2].set_xlabel(r'$\lambda_3$')
axes[2].set_ylabel('PDF')
axes[2].set_title(r'PDF of $\lambda_3$')
axes[2].grid(True, alpha=0.3)

plt.tight_layout()
# plt.savefig('part2_eigenvalue_pdfs.png', dpi=240, bbox_inches='tight')
# plt.show()

#q1.3
P = ux + vy + wz
print(f"<|P|> using lambda arrays: {np.mean(np.abs(P)):.6e}")

# From q1.1:  Q = -1/2 * A_ij * A_ji
# and R = -1/3 * A_ij * A_jk * A_ki

Sxx = ux
Syy = vy
Szz = wz
Sxy = 0.5 * (uy + vx)
Sxz = 0.5 * (uz + wx)
Syz = 0.5 * (vz + wy)

Rxx = 0.0
Ryy = 0.0
Rzz = 0.0
Rxy = 0.5 * (uy - vx)
Rxz = 0.5 * (uz - wx)
Ryz = 0.5 * (vz - wy)

Q = np.zeros((nx, ny))
R = np.zeros((nx, ny))

for i in range(nx):
    for j in range(ny):
        A = np.array([[ux[i, j], uy[i, j], uz[i, j]],
                      [vx[i, j], vy[i, j], vz[i, j]],
                      [wx[i, j], wy[i, j], wz[i, j]]])
        
    
        Q[i, j] = -0.5 * np.trace(A @ A)
        R[i, j] = -1.0/3.0 * np.trace(A @ A @ A)

# for normalization
Q_rms = np.sqrt(np.mean(Q**2))
R_rms = np.sqrt(np.mean(R**2))

Q_vmin, Q_vmax = np.percentile(Q / Q_rms, [5, 95])
R_vmin, R_vmax = np.percentile(R / R_rms, [5, 95])

fig, axes = plt.subplots(1, 2, figsize=(14, 6))

im1 = axes[0].pcolor(Q / Q_rms, cmap='RdBu_r', shading='auto', vmin=Q_vmin, vmax=Q_vmax)
axes[0].set_title(r'$Q/Q_{rms}$')
axes[0].set_xlabel('x')
axes[0].set_ylabel('y')
cbar1 = plt.colorbar(im1, ax=axes[0])
cbar1.set_label(r'$Q/Q_{rms}$')

im2 = axes[1].pcolor(R / R_rms, cmap='RdBu_r', shading='auto', vmin=R_vmin, vmax=R_vmax)
axes[1].set_title(r'$R/R_{rms}$')
axes[1].set_xlabel('x')
axes[1].set_ylabel('y')
cbar2 = plt.colorbar(im2, ax=axes[1])
cbar2.set_label(r'$R/R_{rms}$')

plt.tight_layout()
# plt.savefig('part4_QR_fields.png', dpi=240, bbox_inches='tight')
# plt.show()

#q1.5
# Calculate enstrophy from strain rate tensor
# ω² = 2 * S_ij * S_ji (proper tensor contraction = trace(S²))

SijSji = np.zeros((nx, ny))

for i in range(nx):
    for j in range(ny):
        S = np.array([[ux[i, j], 0.5 * (uy[i, j] + vx[i, j]), 0.5 * (uz[i, j] + wx[i, j])],
                      [0.5 * (uy[i, j] + vx[i, j]), vy[i, j], 0.5 * (vz[i, j] + wy[i, j])],
                      [0.5 * (uz[i, j] + wx[i, j]), 0.5 * (vz[i, j] + wy[i, j]), wz[i, j]]])
        SijSji[i, j] = np.trace(S @ S)

omega_squared = 2.0 * SijSji
omega_squared_mean = np.mean(omega_squared)
Q_w = omega_squared_mean / 4.0
Q_norm = Q / Q_w
R_norm = R / (Q_w ** 1.5)

mask_threshold = 1e-7 
Q_norm_masked = Q_norm.copy()
R_norm_masked = R_norm.copy()
Q_norm_masked[np.abs(Q_norm) < mask_threshold] = np.nan
R_norm_masked[np.abs(R_norm) < mask_threshold] = np.nan
valid_mask = ~np.isnan(Q_norm_masked) & ~np.isnan(R_norm_masked)
Q_norm_valid = Q_norm_masked[valid_mask]
R_norm_valid = R_norm_masked[valid_mask]

fig, ax = plt.subplots(figsize=(10, 8))

# ===== CUSTOMIZABLE HISTOGRAM RESOLUTION =====
# Increase bins for finer resolution
H, xedges, yedges = np.histogram2d(Q_norm_valid, R_norm_valid, bins=1000)
H = np.maximum(H, 1e-6)  

vmin_hist = 1 
vmax_hist = np.max(H) 


im = ax.pcolormesh(xedges, yedges, H.T, cmap='viridis', norm=mpl.colors.LogNorm(vmin=vmin_hist, vmax=vmax_hist))
cbar = plt.colorbar(im, ax=ax, label='Count (log scale)')

Q_line = np.linspace(np.min(Q_norm_valid), 0, 1000)
R_line_pos = (2.0 / np.sqrt(27)) * np.abs(Q_line) ** 1.5
R_line_neg = -(2.0 / np.sqrt(27)) * np.abs(Q_line) ** 1.5

ax.plot(Q_line, R_line_pos, 'r-', linewidth=2.5, label='Discriminant: $27R^2/4 + Q^3 = 0$')
ax.plot(Q_line, R_line_neg, 'r-', linewidth=2.5)


Q_min = np.percentile(Q_norm_valid, 0.3)  
Q_max = np.percentile(Q_norm_valid, 99) 
R_min = np.percentile(R_norm_valid, 0.3)
R_max = np.percentile(R_norm_valid, 99)

# Set limits with slight margins
ax.set_xlim([Q_min*0.4, Q_max*1.4])
ax.set_ylim([R_min*0.4, R_max*1.4])

ax.set_xlabel(r'$Q/\langle Q_w \rangle$', fontsize=12)
ax.set_ylabel(r'$R/\langle Q_w \rangle^{3/2}$', fontsize=12)
ax.set_title('Joint Distribution: Q-R Phase Space (Tear-Drop Profile)', fontsize=12)
ax.legend(fontsize=11)
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('part5_QR_scatter.png', dpi=150, bbox_inches='tight')
plt.show()

vortex_stretch = np.sum((Q_norm_valid < 0) & (R_norm_valid > 0)) / len(Q_norm_valid) * 100
vortex_compress = np.sum((Q_norm_valid < 0) & (R_norm_valid < 0)) / len(Q_norm_valid) * 100
stable_node_saddle = np.sum((Q_norm_valid < 0) & (np.abs(R_norm_valid) <= (2.0/np.sqrt(27)) * np.abs(Q_norm_valid)**1.5)) / len(Q_norm_valid) * 100
unstable_node_saddle = np.sum((Q_norm_valid < 0) & (np.abs(R_norm_valid) > (2.0/np.sqrt(27)) * np.abs(Q_norm_valid)**1.5)) / len(Q_norm_valid) * 100

print(f"Percentage in different topologies:")
print(f"  - Vortex-stretching region (Q<0, R>0): {vortex_stretch:.2f}%")
print(f"  - Vortex-compression region (Q<0, R<0): {vortex_compress:.2f}%")
print(f"  - Stable node-saddle-saddle (inside tear-drop): {stable_node_saddle:.2f}%")
print(f"  - Unstable node-saddle-saddle (outside tear-drop): {unstable_node_saddle:.2f}%")

