"""
This file uses PyElastica to model a tendon-driven robot with disks
and visualizes its motion using matplotlib.

The characteristics and dimensions of the tendon-driven robot with disks are the same as used in the 
paper titled "How to Model Tendon-Driven Continuum Robots and Benchmark Modelling Performance" by Rao et al. 2021.
They are as follows:
Disks:
    Number of disks: 10
    Disk radius: 10 mm (Distance from center of backbone to tendon attachment point)
    Disk thickness: 7.5 mm
    Mass of each disk: 0.2 g
Backbone:
    Length: 400 mm
    Radius: 0.7 mm
    Young's Modulus (E): 54*10^9 Pa 
    Poisson's Ratio (nu): 0.3
    Moment of Inertia (I): 1/4 * pi * radius^4
    Mass of backbone: 0.0115 kg/m
Actuation:
    Only one tendon will be pulled at a time to observe the bending motion of the robot.
Characteristics:
    Robot will be aligned with the z-axis at starting position.
    It will contain 2 tendons symmetrically placed around the backbone at 180 degrees from each other.
    Tendons will be located on the left and right sides of the backbone when looking along the positive z-axis.

The modeling assumptions and principles are based on the Cosserat rod modeling theory presented in Caleb Rucker's thesis and on the 
assumptions made in the paper titled "How to Model Tendon-Driven Continuum Robots and Benchmark Modelling Performance" by Rao et al. 2021.
Assumptions:
    Frictionless interaction between tendons and the channel through which they travel.
        Tension is constant along the length of each tendon.
    Locations of the tendons within the cross-section of the rod do not change during deformation.
    Tendon elongation is negligible.
    The weight of the backbone and disks is negligible.
        Note: If you want to add gravity effects, you can use the GravityForces external force class to consider the gravitational force
        on the backbone. For the disks, you can use the custom external force class called "TendonForcesGravity"
Modeling Principles:
    At the point where the tendon terminates along the length of the robot, it applies a point force to its attachment point AND a point moment at the backbone centroid.
    Distributed forces and moments exerted along the backbone due to the tendons being pulled.
    The backbone is the Cosserat rod being modeled.
"""
# Imports for PyElastica
import numpy as np
import elastica as ea
import sys
from elastica.rod.cosserat_rod import CosseratRod
from elastica.boundary_conditions import OneEndFixedBC
from elastica.external_forces import TendonForces
from elastica.dissipation import AnalyticalLinearDamper
from elastica.callback_functions import CallBackBaseClass
from elastica.timestepper.symplectic_steppers import PositionVerlet
from elastica.timestepper import integrate

# Imports for visualization
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import animation


class TendonDrivenWithDisksSimulator(
    ea.BaseSystemCollection, ea.Connections, ea.Constraints, 
    ea.Forcing, ea.Damping, ea.CallBacks
):
    pass

# Define callback class to store data during simulation
class MyCallBack(CallBackBaseClass):
        def __init__(self, step_skip: int, callback_params):
            CallBackBaseClass.__init__(self)
            self.every = step_skip
            self.callback_params = callback_params
        
        def make_callback(self, system, time, current_step: int):
            if current_step % self.every == 0:
                pos = system.position_collection
                vel = system.velocity_collection

                # NaN checks
                if np.isnan(pos).any():
                    print("NAN in position at step:", current_step, "time:", time)
                    print("sample positions:", pos.flatten()[:12])
                    sys.exit(1)
                if np.isnan(vel).any():
                    print("NAN in velocity at step:", current_step, "time:", time)
                    sys.exit(1)
                self.callback_params["time"].append(time)
                self.callback_params["position"].append(system.position_collection.copy())
                self.callback_params["step"].append(current_step)
                self.callback_params["directors"].append(system.director_collection.copy())
                self.callback_params["velocity"].append(system.velocity_collection.copy())
                self.callback_params["external_forces"].append(system.external_forces.copy())
                self.callback_params["external_torques"].append(system.external_torques.copy())

# Create simulator instance
simulator = TendonDrivenWithDisksSimulator()

# Define rod parameters
n_elements = 100
base_length = 200e-3 # in meters
base_radius = 0.7e-3 # in meters
start = np.array([0.0, 0.0, 0.0])
direction = np.array([0.0, 0.0, 1.0])
normal = np.array([1.0, 0.0, 0.0])
E = 54e9  # Young's modulus in Pa
nu = 0.3  # Poisson's ratio
G = E / (2 * (1 + nu))  # Shear modulus in Pa
density = (0.0115 * 200e-3) / (np.pi * base_radius**2 * base_length)  # Density in kg/m^3
dtmax = (base_length/n_elements) * np.sqrt(density / max(E, G)) # Maximum size of time step
print("Maximum time_step magnitude: ",dtmax)

# Define simulation parameters
final_time = 1.0 # in seconds
time_step = min(1.0e-6, 0.5*dtmax) # in seconds
total_steps = int(final_time / time_step)
step_skip = 1500 

# Create backbone rod
rod1 = CosseratRod.straight_rod(
    n_elements= n_elements,                       # Number of elements
    start=start,                                  # Starting position of first node in rod    
    direction=direction,                          # Direction the rod extends
    normal=normal,                                # Normal vector of rod
    base_length=base_length,                      # Original length of rod (m)
    base_radius=base_radius,                      # Original radius of rod (m)
    density=density,                              # Density of rod (kg/m^3)
    youngs_modulus=E,                             # Elastic Modulus (Pa)
    shear_modulus=G,                              # Shear Modulus (Pa)
)

# Append rod to simulator
simulator.append(rod1)

# Fix one end of the rod (proximal end)
simulator.constrain(rod1).using(
    OneEndFixedBC,                  # Displacement BC being applied
    constrained_position_idx=(0,),  # Node number to apply BC
    constrained_director_idx=(0,)   # Element number to apply BC
)

# Actuate the tendon (add tension to the tendon) --> apply forces and torques on rod caused by tendon
tension = 5 # in Newtons
simulator.add_forcing_to(rod1).using(
    TendonForces,                       # Tendon forcing with gravity on disks
    vertebra_height = 0.007,            # Height of each disk (m)
    num_vertebrae = 10,                 # Number of disks
    first_vertebra_node = 10,           # First node with disk
    final_vertebra_node = 100,          # Last node with disk
    tension = tension,                  # Tension in the tendon (N)
    vertebra_height_orientation = np.array([0.0, 1.0, 0.0]), # Sets tendon to the right of the backbone
    n_elements = n_elements             # Total number of elements in the rod
)

# Add damping to the rod for numerical stability
simulator.dampen(rod1).using(
    AnalyticalLinearDamper,
    damping_constant=0.5,
    time_step=time_step,
)

# Create dictionary to hold data obtained
callback_data_rod1 = {
    "time": [],
    "position": [],
    "step": [],
    "directors": [],
    "velocity": [],
    "external_forces": [],
    "external_torques": [],
}

# Add callback to the simulator
simulator.collect_diagnostics(rod1).using(
    MyCallBack,
    step_skip=step_skip,
    callback_params=callback_data_rod1,
)

# Finalize the simulator
simulator.finalize()

# Time integration loop
timestepper = PositionVerlet()
integrate(timestepper, simulator, final_time, total_steps)

# -------------------------
# Visualization 
# -------------------------
position_data = callback_data_rod1["position"]
directors_data = callback_data_rod1["directors"]

# Set equal aspect ratio for 3D plots
def set_axes_equal(ax: plt.Axes):
    x_limits = ax.get_xlim3d()
    y_limits = ax.get_ylim3d()
    z_limits = ax.get_zlim3d()

    x_range = abs(x_limits[1] - x_limits[0])
    x_middle = np.mean(x_limits)
    y_range = abs(y_limits[1] - y_limits[0])
    y_middle = np.mean(y_limits)
    z_range = abs(z_limits[1] - z_limits[0])
    z_middle = np.mean(z_limits)

    plot_radius = 0.5 * max([x_range, y_range, z_range])

    ax.set_xlim3d([x_middle - plot_radius, x_middle + plot_radius])
    ax.set_ylim3d([y_middle - plot_radius, y_middle + plot_radius])
    ax.set_zlim3d([z_middle - plot_radius, z_middle + plot_radius])

# Extract saved data into arrays
positions_list = callback_data_rod1["position"]
n_frames = len(positions_list)
if n_frames == 0:
    raise RuntimeError("No callback positions saved. Make sure step_skip and simulation length produce saved frames.")

positions = np.stack(positions_list, axis=0) 
_, _, n_nodes = positions.shape

first_vertebra_node = 10
final_vertebra_node = 100
num_vertebrae = 10

vertebra_nodes = np.linspace(first_vertebra_node, final_vertebra_node, num_vertebrae, dtype=int)

# Build the figure
fig = plt.figure(figsize=(9, 7))
ax = fig.add_subplot(111, projection="3d")
# Force view onto the YZ plane (looking along X)
# ax.view_init(elev=0, azim=0)
ax.set_xlabel("X (m)")
ax.set_ylabel("Y (m)")
ax.set_zlabel("Z (m)")
ax.set_title("Tendon-driven rod — backbone + disks")

# initial data (frame 0)
x0 = positions[0, 0, :]
y0 = positions[0, 1, :]
z0 = positions[0, 2, :]

# final tip position data
xf = positions[-1, 0, :]
yf = positions[-1, 1, :]
zf = positions[-1, 2, :]
print("Final tip position: ", xf[-1], yf[-1], zf[-1])

# final midpoint position data
xm = positions[-1, 0, n_nodes // 2]
ym = positions[-1, 1, n_nodes // 2]
zm = positions[-1, 2, n_nodes // 2]
print("Final midpoint position: ", xm, ym, zm)


# backbone line 
(backbone_line,) = ax.plot(x0, y0, z0, lw=2, label="backbone")

# disk markers
disk_scatter = ax.scatter(
    positions[0, 0, vertebra_nodes],
    positions[0, 1, vertebra_nodes],
    positions[0, 2, vertebra_nodes],
    s=50,
    marker="o",
    label="disks (attachments)"
)

# markers for proximal end
prox_marker = ax.scatter(x0[0:1], y0[0:1], z0[0:1], s=60, marker="^", label="proximal (fixed)")

ax.legend(loc="upper left")

# set axis limits using the entire dataset for stable framing
all_x = positions[:, 0, :].ravel()
all_y = positions[:, 1, :].ravel()
all_z = positions[:, 2, :].ravel()
ax.set_xlim(all_x.min() - 0.01, all_x.max() + 0.01)
ax.set_ylim(all_y.min() - 0.01, all_y.max() + 0.01)
ax.set_zlim(all_z.min() - 0.01, all_z.max() + 0.01)
set_axes_equal(ax)

# Animation update function
def update_frame(i):
    x = positions[i, 0, :]
    y = positions[i, 1, :]
    z = positions[i, 2, :]

    # update backbone line
    backbone_line.set_data(x, y)
    backbone_line.set_3d_properties(z)

    # update disk markers
    disk_scatter._offsets3d = (
        positions[i, 0, vertebra_nodes],
        positions[i, 1, vertebra_nodes],
        positions[i, 2, vertebra_nodes],
    )

    # update proximal marker
    prox_marker._offsets3d = (x[0:1], y[0:1], z[0:1])

    # update title
    if "time" in callback_data_rod1 and len(callback_data_rod1["time"]) == n_frames:
        ax.set_title(f"Tendon-driven rod — t = {callback_data_rod1['time'][i]:.4f} s")
    else:
        ax.set_title(f"Tendon-driven rod — frame {i+1}/{n_frames}")

    return backbone_line, disk_scatter, prox_marker

# Run the animation
fps = 30
interval_ms = 1000.0 / fps  # milliseconds between frames
anim = animation.FuncAnimation(
    fig, update_frame, frames=n_frames, interval=interval_ms, blit=False
)

# Save animation as MP4 video
# anim.save("tdcr_sim_5N__400e-3_3min.mp4", fps=30, dpi=200, writer="ffmpeg")

plt.show()