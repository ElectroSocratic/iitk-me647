import numpy as np
import matplotlib as mpl
mpl.use("TkAgg")
import matplotlib.pyplot as plt


data = np.load("isotropic1024_stack3.npz")
u3 = data["u"]
v3 = data["v"]
w3 = data["w"]
u_field = u3[:, :, 1]
v_field = v3[:, :, 1]
w_field = w3[:, :, 1]
nx = u_field.shape[0]
ny = u_field.shape[1]
dx = 2.0 * np.pi / nx

# Kolmogorov timescale  
nu = 0.000185  

def ddx(f):
    return (np.roll(f, -1, axis=0) - np.roll(f, 1, axis=0)) / (2 * dx)

def ddy(f):
    return (np.roll(f, -1, axis=1) - np.roll(f, 1, axis=1)) / (2 * dx)


ux = ddx(u_field)
uy = ddy(u_field)

vx = ddx(v_field)
vy = ddy(v_field)

#q2.1
SijSij = ux**2 + vy**2 + 0.5 * (uy + vx)**2
epsilon = 2.0 * nu * SijSij
epsilon_mean = np.mean(epsilon)

tau = np.sqrt(nu / epsilon_mean) 


#bilinear interpolation
def bilinear_interpolate(field, x, y, dx, nx):

    # physical to grid
    ix = x / dx
    iy = y / dx

    # Get integer 
    i0 = int(np.floor(ix)) % nx
    i1 = (i0 + 1) %nx
    j0 = int(np.floor(iy)) %nx
    j1 = (j0 + 1) %nx
    
    # Get fraction
    fx = ix - np.floor(ix)
    fy = iy - np.floor(iy)
    
    # Bilinear interpolation
    f00 = field[i0, j0]
    f10 = field[i1, j0]
    f01 = field[i0, j1]
    f11 = field[i1, j1]
    
    f0 = f00 * (1 - fx) + f10 * fx
    f1 = f01 * (1 - fx) + f11 * fx
    f = f0 * (1 - fy) + f1 * fy
    
    return f

def get_velocity_at_position(x, y, u_field, v_field, dx, nx):
    u = bilinear_interpolate(u_field, x, y, dx, nx)
    v = bilinear_interpolate(v_field, x, y, dx, nx)
    return u, v


np.random.seed(42)

Np = 20
T_values = [1, 5, 10]
dt = tau / 20  

# Initialize particles
x_init = np.random.uniform(0, 2*np.pi, Np)
y_init = np.random.uniform(0, 2*np.pi, Np)

trajectories = {}

for T in T_values:
    n_steps = int(T / dt)
    x = x_init.copy()
    y = y_init.copy()
    
    x_traj = np.zeros((n_steps + 1, Np))
    y_traj = np.zeros((n_steps + 1, Np))
    x_traj[0, :] = x
    y_traj[0, :] = y
    
    # Euler integration
    for step in range(n_steps):
    
        u = np.zeros(Np)
        v = np.zeros(Np)
        for p in range(Np):
            u[p], v[p] = get_velocity_at_position(x[p], y[p], u_field, v_field, dx, nx)
        
     
        x = x + u * dt
        y = y + v * dt
        
        x_traj[step + 1, :] = x
        y_traj[step + 1, :] = y
    
    trajectories[T] = (x_traj, y_traj)
    
    # Plot 
    fig, ax = plt.subplots(figsize=(10, 10))
    
    for p in range(Np):
        ax.plot(x_traj[:, p], y_traj[:, p], linewidth=0.8, alpha=0.7)
    
    # Mark initial 
    ax.scatter(x_init, y_init, c='red', s=100, marker='o', 
               label='Initial positions', zorder=5, edgecolors='darkred', linewidth=1)
    
    # Mark final 
    ax.scatter(x_traj[-1, :], y_traj[-1, :], c='blue', s=100, marker='s', 
               label='Final positions', zorder=5, edgecolors='darkblue', linewidth=1)
    
    ax.set_xlabel('x (physical)', fontsize=12)
    ax.set_ylabel('y (physical)', fontsize=12)
    ax.set_title(f'Lagrangian Trajectories (Np={Np}, T={T})', fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.set_xlim([0, 2*np.pi])
    ax.set_ylim([0, 2*np.pi])
    ax.grid(True, alpha=0.3)
    ax.set_aspect('equal')
    
    plt.tight_layout()
    plt.savefig(f'q1_trajectories_T{T}.png', dpi=150, bbox_inches='tight')
    plt.show()

#q2.2 

Np = 1000
T = 10
n_steps = int(T / dt)

x = np.random.uniform(0, 2*np.pi, Np)
y = np.random.uniform(0, 2*np.pi, Np)
x_traj = np.zeros((n_steps + 1, Np))
y_traj = np.zeros((n_steps + 1, Np))
x_traj[0, :] = x
y_traj[0, :] = y

# Euler integration
for step in range(n_steps):
    u = np.zeros(Np)
    v = np.zeros(Np)
    for p in range(Np):
        u[p], v[p] = get_velocity_at_position(x[p], y[p], u_field, v_field, dx, nx)
    
    x = x + u * dt
    y = y + v * dt
    
    x_traj[step + 1, :] = x
    y_traj[step + 1, :] = y


last_fraction = 0.1  # use last 10% of the total time 
n_time = x_traj.shape[0]
n_last = max(1, int(np.round(last_fraction * (n_time - 1))))
start_idx = max(0, n_time - 1 - n_last)

displacement_last = np.sqrt((x_traj[-1, :] - x_traj[start_idx, :])**2 +
                            (y_traj[-1, :] - y_traj[start_idx, :])**2)

threshold = 0.1
trapped_mask = displacement_last < threshold
n_trapped = np.sum(trapped_mask)
trapped_idx = trapped_mask

# Plot 
fig, ax = plt.subplots(figsize=(12, 12))

normal_idx = ~trapped_mask
if np.any(normal_idx):
    for p in np.where(normal_idx)[0][:1000]:  
        ax.plot(x_traj[:, p], y_traj[:, p], linewidth=0.5, alpha=0.4, color='blue')

# Plot trapped particles
if np.any(trapped_idx):
    for idx_i, p in enumerate(np.where(trapped_idx)[0]):
        
        ax.plot(x_traj[start_idx:, p], y_traj[start_idx:, p], linewidth=1.5, alpha=0.8, color='red', label='Trapped' if idx_i == 0 else '')

# mark initial positions (at start of last_fraction) for normal and trapped
ax.scatter(x_traj[0, normal_idx], y_traj[0, normal_idx], c='blue', s=20, alpha=0.3, label='Normal particles (init)')
# mark final positions for trapped particles
ax.scatter(x_traj[-1, trapped_idx], y_traj[-1, trapped_idx], c='darkred', s=40, marker='x', label='Trapped particles (final position)', zorder=5)

ax.set_xlabel('x (physical)', fontsize=12)
ax.set_ylabel('y (physical)', fontsize=12)
ax.set_title(f'Lagrangian Trajectories with Trapped Particles (Np={Np}, T={T})', 
             fontsize=14, fontweight='bold')
ax.legend(fontsize=11)
ax.set_xlim([0, 2*np.pi])
ax.set_ylim([0, 2*np.pi])
ax.grid(True, alpha=0.3)
ax.set_aspect('equal')

plt.tight_layout()
plt.savefig('q2_trapped_particles.png', dpi=150, bbox_inches='tight')
plt.show()

#q2.3

Np = 10000
T_total = 10
n_steps = int(T_total / dt)

x = np.random.uniform(0, 2*np.pi, Np)
y = np.random.uniform(0, 2*np.pi, Np)
x_init = x.copy()
y_init = y.copy()

msd_times = []
msd_values = []
sample_interval = max(1, n_steps // 100)

# Euler integration
for step in range(n_steps):
    u = np.zeros(Np)
    v = np.zeros(Np)
    for p in range(Np):
        u[p], v[p] = get_velocity_at_position(x[p], y[p], u_field, v_field, dx, nx)
    
    x = x + u * dt
    y = y + v * dt
    

    if (step + 1) % sample_interval == 0:
        t_pseudo = (step + 1) * dt
        dx_disp = x - x_init
        dy_disp = y - y_init
        msd = np.mean(dx_disp**2 + dy_disp**2)
        
        msd_times.append(t_pseudo)
        msd_values.append(msd)

msd_times = np.array(msd_times)
msd_values = np.array(msd_values)

# Plot 
fig, ax = plt.subplots(figsize=(11, 9))

ax.loglog(msd_times, msd_values, 'o-', linewidth=2, markersize=4, label='MSD data', color='navy')

t_ref = msd_times
# Ballistic scaling: ~t²
ballistic = 0.05 * t_ref**2
# Diffusive scaling: ~t
diffusive = 0.1 * t_ref

ax.loglog(t_ref, ballistic, '--', linewidth=2, alpha=0.6, label='Ballistic (∝ t²)', color='red')
ax.loglog(t_ref, diffusive, '--', linewidth=2, alpha=0.6, label='Diffusive (∝ t)', color='green')

ax.set_xlabel('Pseudo-time t (s)', fontsize=13)
ax.set_ylabel('Mean-Square Displacement ⟨Δx²(t)⟩ (m²)', fontsize=13)
ax.set_title('Mean-Square Displacement vs Time (Np=10⁴)', fontsize=14, fontweight='bold')
ax.legend(fontsize=12)
ax.grid(True, which='both', alpha=0.3)

plt.tight_layout()
plt.savefig('q3_msd_loglog.png', dpi=150, bbox_inches='tight')
plt.show()

#q2.4

# Use the diffusive regime (late times) to calculate Dr
# From ⟨Δx²(t)⟩ = Dr·t, we get Dr = ⟨Δx²(t)⟩ / t

# Use data from diffusive regime (last 30% of time points)
n_diffusive = len(msd_times) // 3
diffusive_times = msd_times[-n_diffusive:]
diffusive_msd = msd_values[-n_diffusive:]

# Linear fit in the diffusive regime
# MSD = Dr * t, so slope = Dr
coeffs = np.polyfit(diffusive_times, diffusive_msd, 1)
Dr_fitted = coeffs[0]
Dr_offset = coeffs[1]

# Alternative: average ratio in diffusive regime
Dr_ratios = diffusive_msd / diffusive_times
Dr_avg = np.mean(Dr_ratios)

print(f"\nDiffusivity coefficient estimation:")
print(f"  Dr (from linear fit): {Dr_fitted:.6e} m²/s")
print(f"  Dr (from average ratio): {Dr_avg:.6e} m²/s")
print(f"  Linear fit: MSD = {Dr_fitted:.6e} × t + {Dr_offset:.6e}")

# Compare with molecular diffusivity (dye in water)
# Typical dye diffusivity ~ 10^-5 to 10^-9 m²/s
D_molecule = 1e-9  # m²/s (approximate for dye in water)
print(f"\nComparison with molecular diffusivity:")
print(f"  Molecular diffusivity (dye): D_mol ≈ {D_molecule:.2e} m²/s")
print(f"  Turbulent diffusivity: D_r ≈ {Dr_fitted:.2e} m²/s")
print(f"  Enhancement factor: Dr/D_mol ≈ {Dr_fitted/D_molecule:.2e}")

# Plot linear fit in diffusive regime
fig, ax = plt.subplots(figsize=(11, 8))

ax.plot(msd_times, msd_values, 'o-', linewidth=2, markersize=5, label='MSD data', color='navy', alpha=0.7)

# Plot fitted line in diffusive regime
t_fit = np.array([diffusive_times[0], diffusive_times[-1]])
msd_fit = Dr_fitted * t_fit + Dr_offset
ax.plot(t_fit, msd_fit, '-', linewidth=3, color='red', label=f'Linear fit (Dr={Dr_fitted:.3e})', zorder=10)

# Highlight diffusive regime
ax.axvspan(diffusive_times[0], diffusive_times[-1], alpha=0.15, color='green', label='Diffusive regime')

ax.set_xlabel('Pseudo-time t (s)', fontsize=13)
ax.set_ylabel('Mean-Square Displacement ⟨Δx²(t)⟩ (m²)', fontsize=13)
ax.set_title('Turbulent Diffusivity Calculation (Dr from MSD ≈ Dr·t)', 
             fontsize=14, fontweight='bold')
ax.legend(fontsize=12)
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('q4_diffusivity.png', dpi=150, bbox_inches='tight')
plt.show()

