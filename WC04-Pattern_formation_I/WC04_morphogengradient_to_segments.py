# Installed modules
import numpy as np
import matplotlib.pyplot as plt
import os

os.makedirs("output", exist_ok=True)

##############
# PARAMETERS #
##############
ABC=False # Flag to indicate whether to plot A, B, C or not

# Grid and simulation parameters Question1
Lx = 40.0  # Length of the domain in x in micrometers
Ly = 10.0  # Length of the domain in y in micrometers
T = 200  # Total time in seconds
dx = 0.5  # Grid spacing in x
dt = 0.1  # Time step
nx = int(Lx/dx)+2  # Number of grid points in x + padding grid points
ny = int(Ly/dx)+2  # Number of grid points in y + padding grid points
# Padding grid points to account for boundary conditions
nt = int(T/dt)  # Number of time steps
D = 0.4  # Diffusion coefficient in mm^2/s
decayM =0.001 # Decay rate in 1/s


# Parameters for A, B, C
... # TODO define parameters for A, B, C as needed in Question5

#############
# FUNCTIONS #
#############

# Reaction-diffusion equation
def reaction_diffusion_step(u, D, dt, dx, decay):
    un = u.copy()
    u[1:-1, 1:-1] = un[1:-1, 1:-1] +  D *dt / dx**2 * (un[2:, 1:-1] + un[:-2, 1:-1] + \
                    un[1:-1, 2:]  + un[1:-1, :-2] - 4 * un[1:-1, 1:-1]) - \
                    decay * un[1:-1, 1:-1] * dt
    ## for loop version to understand the equation Question1
    # for i in range(1, nx-1):
    #     for j in range(1, ny-1):
    #         u[i, j] = (un[i, j] +
    #                    D * dt / dx**2 * (un[i+1, j] + un[i-1, j] +
    #                    un[i, j+1] + un[i, j-1] - 4 * un[i, j]) - decay * un[i, j] * dt)
    #boundary conditions
    u[-1, :] = (u[-2, :]/u[-3, :])*u[-2, :]  if sum(u[-3, :]) != 0 else np.zeros(ny)
    #to understand this line:
    #if sum(u[-3, :]) != 0:
    #    u[-1, :] = (u[-2, :]/u[-3, :])*u[-2, :]#extrapolate from third to last row
    #else:
    #    u[-1, :] = np.zeros(ny) #if already zero in third to last row, also zero in last row
    u[:, 0] = u[:, 1]
    u[:, -1] = u[:, -2]

    return u

##################
# EXECUTION CODE #
##################

# Stability criterion
if D * dt / dx**2 > 0.5:
    raise ValueError("Stability criterion not met")

# Initial condition Question1
u = np.zeros((nx, ny))
u[0, :] = 100


# initilize figure and axes for plotting
if ABC==False:
    fig, (ax_M, ax_lines) = plt.subplots(2, figsize=(10, 8), gridspec_kw={'height_ratios': [3, 1]})  # Make the first graph 3 times the height of the second
else:
    # initilize figure and axes for plotting
    fig, (ax_M,ax_ABC, ax_lines) = plt.subplots(3, figsize=(10, 8), gridspec_kw={'height_ratios': [3, 3, 1]})  # Make the first graph 3 times the height of the second


# Time-stepping simulation loop
for n in range(nt):
    # Update all variables
    u = reaction_diffusion_step(u, D, dt, dx, decayM)
    
    if n == 0:  # Initial plot
        imshow_M_plot = ax_M.imshow(u.T, cmap='viridis', origin='lower', aspect='auto')
        ax_M.set_title(f"Time: {n*dt}")
        ax_M.set_xlabel('x direction')
        ax_M.set_ylabel('y direction')
        ax_M.set_xticks([])
        ax_M.set_yticks([])
        if ABC==True: 
            max_indices = np.argmax(np.array([A, B, C]), axis=0)
            max_indices[0,0]=0
            max_indices[0,1]=1
            max_indices[0,2]=2
            imshow_ABC = ax_ABC.imshow(max_indices.T, cmap='RdBu_r', vmin=0, vmax=2, origin='lower', aspect='auto')
            ax_ABC.set_title('Max Concentration Phase')
            ax_ABC.set_xlabel('x direction')
            ax_ABC.set_ylabel('y direction')
            ax_ABC.set_xticks([])
            ax_ABC.set_yticks([])
            Aplot = ax_lines.plot([x*dx for x in range(nx)], A[:, 2], label='A', color='blue')
            Bplot = ax_lines.plot([x*dx for x in range(nx)], B[:, 2], label='B', color='orange')
            Cplot = ax_lines.plot([x*dx for x in range(nx)], C[:, 2], label='C', color='red')

    
        # Plot the concentration at a specific y index (e.g., y=2)    
        line_plot = ax_lines.plot([x*dx for x in range(nx)], u[:, 2], label='M', color='green')

        
        ax_lines.legend(loc='upper right')
        ax_lines.set_ylim(0, 100)
        ax_lines.tick_params(axis='y')
        ax_lines.set_xlim(0, dx*nx)
        ax_lines.set_xlabel('x')
        ax_lines.set_ylabel('Concentration at y=2')
        ax_lines.tick_params(axis='x')

    if n % 20 == 0:  # Update plot every so many time steps
        #update the imshow M plot with the new data
        imshow_M_plot.set_data(u.T)
        ax_M.set_title(f"Time: {n*dt}")
        # Update the line plots with new data
        line_plot[0].set_ydata(u[:, 2])

        if ABC==True: 
            # Get which gene is maximal at each index
            max_indices = np.argmax(np.array([A, B, C]), axis=0)
            imshow_ABC.set_data(max_indices.T)        
            Aplot[0].set_ydata(A[:, 2])
            Bplot[0].set_ydata(B[:, 2])
            Cplot[0].set_ydata(C[:, 2])


        #fig.savefig(f"output/morphogen_simulation_{n:05d}.png", dpi=300) #save png (makes simulation run slower!)
    plt.pause(0.01)  # Pause to refresh the plot, make it 0.1 if the plotting goes to slow

# And keep the last plot open
plt.show()

# Or close the plot window when done
#plt.close(fig)
