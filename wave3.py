"""
2D Tsunami Wave Propagation Simulation
- Finite difference leapfrog scheme for the 2D wave equation
- User specifies grid size; courant number calculated automatically
- Animation of wave propagation for both explicit and implicit methods
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d import Axes3D
from scipy.sparse import diags
from scipy.sparse.linalg import spsolve
import time

def simulate_tsunami(Nx=100, Ny=50, plot_3d=True, save_animation=False):
    # Domain parameters
    Lx, Ly = 1000.0, 500.0  # Domain size (m)
    T = 20.0                # Total simulation time (s)
    g = 9.81                # Gravitational acceleration (m/s^2)
    h = 100.0               # Water depth (m)
    c = np.sqrt(g * h)      # Wave speed (m/s)
    
   # Grid setup
    dx, dy = Lx/Nx, Ly/Ny
    x = np.linspace(0, Lx, Nx+1)
    y = np.linspace(0, Ly, Ny+1)
    X, Y = np.meshgrid(x, y, indexing='ij')

    # Time step based on stability (for explicit scheme)
    dt = 0.4 * min(dx, dy) / (c * np.sqrt(2))  # Using 0.4 for safety (below 0.7071)
    nt = int(T / dt)
    courant = c * dt / min(dx, dy) * np.sqrt(2)

    # Initial condition parameters
    A = 5.0                  # Initial amplitude (m)
    sigma_x = sigma_y = 50.0 # Width of disturbance (m)
    x0, y0 = 250.0, 250.0    # Center of disturbance (m)

    # Create initial Gaussian displacement
    initial_condition = A * np.exp(-(((X - x0)**2)/(2*sigma_x**2) + ((Y - y0)**2)/(2*sigma_y**2)))

    # Display simulation parameters
    print("2D Tsunami Simulation")
    print(f"Grid: {Nx}x{Ny}, dx={dx:.2f}m, dy={dy:.2f}m")
    print(f"dt={dt:.4f}s, Steps={nt}, Courant={courant:.3f}")

    # ------------------------------
    # Explicit Leap-Frog Implementation (unchanged)
    print("Solving using explicit method...")
    u_explicit = np.zeros((Nx+1, Ny+1, nt+1))
    u_explicit[:, :, 0] = initial_condition

    # Coefficients for explicit scheme
    r_x_sq = (c * dt / dx)**2
    r_y_sq = (c * dt / dy)**2

    # First time step (assuming zero initial velocity)
    for i in range(1, Nx):
        for j in range(1, Ny):
            u_explicit[i, j, 1] = u_explicit[i, j, 0] + 0.5 * (
                r_x_sq * (u_explicit[i+1, j, 0] - 2*u_explicit[i, j, 0] + u_explicit[i-1, j, 0]) +
                r_y_sq * (u_explicit[i, j+1, 0] - 2*u_explicit[i, j, 0] + u_explicit[i, j-1, 0])
            )
    # Reflecting boundary conditions for first step
    u_explicit[0, :, 1] = u_explicit[1, :, 1]
    u_explicit[Nx, :, 1] = u_explicit[Nx-1, :, 1]
    u_explicit[:, 0, 1] = u_explicit[:, 1, 1]
    u_explicit[:, Ny, 1] = u_explicit[:, Ny-1, 1]

    # Main loop for explicit scheme
    start_time = time.time()
    for n in range(1, nt):
        for i in range(1, Nx):
            for j in range(1, Ny):
                u_explicit[i, j, n+1] = 2*u_explicit[i, j, n] - u_explicit[i, j, n-1] + (
                    r_x_sq * (u_explicit[i+1, j, n] - 2*u_explicit[i, j, n] + u_explicit[i-1, j, n]) +
                    r_y_sq * (u_explicit[i, j+1, n] - 2*u_explicit[i, j, n] + u_explicit[i, j-1, n])
                )
        # Reflecting BCs at each step
        u_explicit[0, :, n+1] = u_explicit[1, :, n+1]
        u_explicit[Nx, :, n+1] = u_explicit[Nx-1, :, n+1]
        u_explicit[:, 0, n+1] = u_explicit[:, 1, n+1]
        u_explicit[:, Ny, n+1] = u_explicit[:, Ny-1, n+1]
        
        if n % (nt//5) == 0:
            print(f"  {n/nt*100:.1f}% complete")
    explicit_time = time.time() - start_time
    print(f"Explicit method completed in {explicit_time:.2f} seconds")

    # ------------------------------
    # Implicit Crank–Nicolson Implementation
    print("Solving using implicit Crank–Nicolson method...")
    u_implicit = np.zeros((Nx+1, Ny+1, nt+1))
    u_implicit[:, :, 0] = initial_condition
    # Use the same first time step as explicit (for consistency)
    u_implicit[:, :, 1] = u_explicit[:, :, 1]

    # Define r_x and r_y for the implicit scheme:
    # Note: These are half the explicit squared coefficients.
    r_x = 0.5 * (c * dt / dx)**2
    r_y = 0.5 * (c * dt / dy)**2

    # Internal grid indices: i=1...Nx-1, j=1...Ny-1
    Nxi = Nx - 1
    Nyi = Ny - 1
    N_unknown = Nxi * Nyi  # number of unknowns in the interior

    # Build the sparse matrix A once, since it does not change with time.
    # For an interior point u_{i,j} the discrete operator gives:
    #  (1 + 2*r_x + 2*r_y) u_{i,j} - r_x (u_{i+1,j}+ u_{i-1,j}) - r_y (u_{i,j+1}+ u_{i,j-1})
    main_diag = np.full(N_unknown, 1 + 2*r_x + 2*r_y)

    # Off-diagonals in the j-direction (neighbors in y)
    off_diag_y = np.full(N_unknown - 1, -r_y)
    # Zero out entries that cross row boundaries
    for k in range(1, N_unknown):
        if k % Nyi == 0:
            off_diag_y[k-1] = 0.0

    # Off-diagonals in the i-direction (neighbors in x): offset = Nyi
    off_diag_x = np.full(N_unknown - Nyi, -r_x)

    diagonals = [main_diag, off_diag_y, off_diag_y, off_diag_x, off_diag_x]
    offsets = [0, 1, -1, Nyi, -Nyi]
    A_matrix = diags(diagonals, offsets, shape=(N_unknown, N_unknown), format='csr')

    # Main time-stepping loop for implicit scheme
    start_time = time.time()
    for n in range(1, nt):
        # Build RHS for interior nodes
        # For each interior node (i,j), using:
        # RHS = 2u^n_{i,j} - u^(n-1)_{i,j} + r_x*(u^(n-1)_{i+1,j} - 2u^(n-1)_{i,j} + u^(n-1)_{i-1,j})
        #                           + r_y*(u^(n-1)_{i,j+1} - 2u^(n-1)_{i,j} + u^(n-1)_{i,j-1})
        RHS = np.zeros((Nxi, Nyi))
        for i in range(1, Nx):
            for j in range(1, Ny):
                # Use reflecting boundaries for neighbors:
                u_n = u_implicit[i, j, n]
                u_nm1 = u_implicit[i, j, n-1]
                u_ip1 = u_implicit[i+1, j, n-1] if i+1 <= Nx-1 else u_implicit[i, j, n-1]
                u_im1 = u_implicit[i-1, j, n-1] if i-1 >= 1 else u_implicit[i, j, n-1]
                u_jp1 = u_implicit[i, j+1, n-1] if j+1 <= Ny-1 else u_implicit[i, j, n-1]
                u_jm1 = u_implicit[i, j-1, n-1] if j-1 >= 1 else u_implicit[i, j, n-1]
                laplacian_nm1 = (u_ip1 - 2*u_nm1 + u_im1)/(dx**2) * (dx**2)  \
                                + (u_jp1 - 2*u_nm1 + u_jm1)/(dy**2) * (dy**2)
                # Note: The factors (dx**2) and (dy**2) cancel; we use r_x and r_y directly.
                RHS[i-1, j-1] = 2*u_implicit[i, j, n] - u_implicit[i, j, n-1] \
                                + r_x * (u_ip1 - 2*u_nm1 + u_im1) \
                                + r_y * (u_jp1 - 2*u_nm1 + u_jm1)
        # Flatten the RHS to a vector
        RHS_vec = RHS.flatten()
        # Solve the linear system for interior nodes at time n+1
        u_new_vec = spsolve(A_matrix, RHS_vec)
        # Reshape back to 2D and insert into u_implicit for interior nodes
        u_implicit[1:Nx, 1:Ny, n+1] = u_new_vec.reshape((Nxi, Nyi))
        # Enforce reflecting boundary conditions:
        u_implicit[0, :, n+1] = u_implicit[1, :, n+1]
        u_implicit[Nx, :, n+1] = u_implicit[Nx-1, :, n+1]
        u_implicit[:, 0, n+1] = u_implicit[:, 1, n+1]
        u_implicit[:, Ny, n+1] = u_implicit[:, Ny-1, n+1]
        
        if n % (nt//5) == 0:
            print(f"  {n/nt*100:.1f}% complete")
    implicit_time = time.time() - start_time
    print(f"Implicit method completed in {implicit_time:.2f} seconds")

    # ------------------------------
    # Create animation comparing explicit and implicit methods
    if plot_3d:
        fig = plt.figure(figsize=(15, 7))
        ax1 = fig.add_subplot(121, projection='3d')
        ax2 = fig.add_subplot(122, projection='3d')
        vmin, vmax = -A, A
        surf1 = ax1.plot_surface(X, Y, u_explicit[:, :, 0], cmap='coolwarm', vmin=vmin, vmax=vmax)
        surf2 = ax2.plot_surface(X, Y, u_implicit[:, :, 0], cmap='coolwarm', vmin=vmin, vmax=vmax)
        ax1.set_zlim(vmin, vmax)
        ax2.set_zlim(vmin, vmax)
        ax1.set_title("Explicit Method")
        ax2.set_title("Implicit Method")
        
        def update(frame):
            ax1.clear()
            ax2.clear()
            surf1 = ax1.plot_surface(X, Y, u_explicit[:, :, frame], cmap='coolwarm', vmin=vmin, vmax=vmax)
            surf2 = ax2.plot_surface(X, Y, u_implicit[:, :, frame], cmap='coolwarm', vmin=vmin, vmax=vmax)
            ax1.set_title(f"Explicit Method (t={frame*dt:.2f}s)")
            ax2.set_title(f"Implicit Method (t={frame*dt:.2f}s)")
            ax1.set_xlabel("x (m)")
            ax1.set_ylabel("y (m)")
            ax1.set_zlabel("Wave Height (m)")
            ax2.set_xlabel("x (m)")
            ax2.set_ylabel("y (m)")
            ax2.set_zlabel("Wave Height (m)")
            ax1.set_zlim(vmin, vmax)
            ax2.set_zlim(vmin, vmax)
            return surf1, surf2
        
        frames = np.linspace(0, nt, min(100, nt+1), dtype=int)
        ani = FuncAnimation(fig, update, frames=frames, interval=50, blit=False)
        if save_animation:
            ani.save('tsunami_comparison.gif', writer='pillow', fps=15, dpi=200)
        plt.tight_layout()
        plt.show()
            
    return u_explicit, u_implicit, courant


if __name__ == "__main__":
    # Run simulation with user-specified grid
    Nx = int(input("Enter number of grid points in x-direction (default=100): ") or "100")
    Ny = int(input("Enter number of grid points in y-direction (default=50): ") or "50")
    
    u_explicit, u_implicit, courant = simulate_tsunami(Nx=Nx, Ny=Ny, save_animation=True)
    
    print(f"Simulation complete with Courant number: {courant:.4f}")
    if courant > 1:
        print("WARNING: Courant number exceeds stability limit of 1 for 2D.")
