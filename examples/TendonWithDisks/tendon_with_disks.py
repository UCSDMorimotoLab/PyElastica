"""
This file uses PyElastica to model a tendon-driven robot with disks and visualizes its motion using matplotlib.

The characteristics and dimensions of the tendon-driven robot with disks are the same as used in the 
paper titled "How to Model Tendon-Driven Continuum Robots and Benchmark Modelling Performance" by Rao et al. 2021.
They are as follows:
Disks:
    Number of disks: 10
    Disk radius: 10 mm (Distance from center of backbone to tendon attachment point)
    Mass of each disk: 0.2 g
Backbone:
    Length: 200 mm
    Radius: 0.7 mm
    Young's Modulus (E): 54*10^9 Pa 
    Poisson's Ratio (nu): 0.3
    Moment of Inertia (I): 1/4 * pi * radius^4
    Mass of backbone: 0.0115 kg/m
Actuation:
    Tendon will be pulled to observe the bending motion of the robot.
Characteristics:
    Robot will be aligned with the z-axis at starting position.
    It contains 1 tendon for actuation (more can be added).

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
                # Note: A rod has N elements with N+1 nodes and N-1 voronoi elements
                # Simulation time
                self.callback_params["time"].append(time)
                # Stores 3D position coordinates of all nodes of the rod at the current time
                self.callback_params["position"].append(system.position_collection.copy())
                # Adds current simulation step number
                self.callback_params["step"].append(current_step)
                # Stores element director matrices (3 x 3) of all elements of the rod at the current time (directors = rotation matrices)
                self.callback_params["directors"].append(system.director_collection.copy())
                # Stores 3D velocity components of all nodes of the rod at the current time
                self.callback_params["velocity"].append(system.velocity_collection.copy())
                # Stores 3D acceleration components of all nodes of the rod at the current time
                self.callback_params["acceleration"].append(system.acceleration_collection.copy())
                # Stores 3D angular velocity components of all elements of the rod at the current time
                self.callback_params["angular_velocity"].append(system.omega_collection.copy())
                # Stores 3D angular acceleration components of all elements of the rod at the current time
                self.callback_params["angular_acceleration"].append(system.alpha_collection.copy())
                # Stores rod element lengths at rest configuration (1D (n_elements) array)
                self.callback_params["rest_lengths"].append(system.rest_lengths.copy())
                # Stores rod elements densities (1D (n_elements) array)
                self.callback_params["densities"].append(system.density.copy())
                # Stores rod elements volume (1D (n_elements) array)
                self.callback_params["volumes"].append(system.volume.copy())
                # Stores rod node masses (1D (n_nodes) array)
                self.callback_params["masses"].append(system.mass.copy())
                # Stores rod element mass second moment of intertia (3x3) at the current time
                self.callback_params["mass_second_moments_of_inertia"].append(system.mass_second_moment_of_inertia.copy())
                # Stores rod element inverse mass moment of inertia (3x3) at the current time
                self.callback_params["inv_mass_second_moments_of_inertia"].append(system.inv_mass_second_moment_of_inertia.copy())
                # Stores rod lengths on the voronoi domain at the rest configuration (1D (n_voronoi) array)
                self.callback_params["rest_voronoi_lengths"].append(system.rest_voronoi_lengths.copy())
                # Stores 3D internal force components for all nodes of the rod at the current time
                self.callback_params["internal_forces"].append(system.internal_forces.copy())
                # Stores 3D internal torque components for all elements of the rod at the current time
                self.callback_params["internal_torques"].append(system.internal_torques.copy())
                # Stores 3D external force components acting on rod nodes at the current time
                self.callback_params["external_forces"].append(system.external_forces.copy())
                # Stores 3D external torque components acting on rod elements at the current time
                self.callback_params["external_torques"].append(system.external_torques.copy())
                # Stores rod element lengths (1D (n_elements) array) at the current time
                self.callback_params["lengths"].append(system.lengths.copy())
                # Stores 3D tangent vectors of all elements of the rod at the current time
                self.callback_params["tangents"].append(system.tangents.copy())
                # Stores rod element radius (1D (n_elements) array)
                self.callback_params["radii"].append(system.radius.copy())
                # Stores rod element dilatation (1D (n_elements) array)
                self.callback_params["dilatations"].append(system.dilatation.copy())
                # Stores rod dilatation on voronoi domain (1D (n_voronoi) array)
                self.callback_params["voronoi_dilatation"].append(system.voronoi_dilatation.copy())
                # Stores rod element dilatation rates (1D (n_elements) array)
                self.callback_params["dilatation_rates"].append(system.dilatation_rate.copy())
                # Stores bend/twist strain vector of all elements of the rod at the current time (First two values per element correspond to normal and binormal bending curvatures respectively and third value is the twist/torsion)
                self.callback_params["curvature"].append(system.kappa.copy())
                # Stores resting bend/twist strain vector of all elements of the rod
                self.callback_params["rest_curvature"].append(system.rest_kappa.copy())
                # Stores shear/stretch strain of all nodes of the rod at the current time
                self.callback_params["sigma"].append(system.sigma.copy())
                # Stores resting shear/stretch strain of all nodes of the rod
                self.callback_params["rest_sigma"].append(system.rest_sigma.copy())
                # Stores internal stress of all nodes of the rod at the current time
                self.callback_params["internal_stress"].append(system.internal_stress.copy())
                # Stores the internal couple (internal torques and bending moments) of all elements of the rod at the current time
                self.callback_params["internal_couple"].append(system.internal_couple.copy()) 

# Create simulator instance
simulator = TendonDrivenWithDisksSimulator()

# Define rod parameters
n_elements = 40
base_length = 200e-3 # in meters
base_radius = 0.7e-3 # in meters
start = np.array([0.0, 0.0, 0.0])
direction = np.array([0.0, 0.0, 1.0])
normal = np.array([1.0, 0.0, 0.0])
E = 54e9  # Young's modulus in Pa
nu = 0.3  # Poisson's ratio
G = E / (2 * (1 + nu))  # Shear modulus in Pa
mass_second_moment_of_inertia = 0.25 * np.pi * base_radius**4
density = (0.0115 * base_length) / (np.pi * base_radius**2 * base_length)  # Density in kg/m^3
dtmax = (base_length/n_elements) * np.sqrt(density / max(E, G)) # Maximum size of time step
print("Maximum time_step magnitude: ",dtmax)

# Define simulation parameters
final_time = 2.0 # in seconds
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
    mass_second_moment_of_inertia=mass_second_moment_of_inertia, # Mass moment of inertia (m^4)
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
    TendonForces,                      # Tendon forcing with gravity on disks
    vertebra_radius = 0.01,            # Distance from center of backbone to tendon attachment point (m)
    num_vertebrae = 10,                # Number of disks
    first_vertebra_node = 4,           # First node with disk
    final_vertebra_node = 39,          # Last node with disk
    tension = tension,                 # Tension in the tendon (N)
    vertebra_radius_orientation = np.array([1.0, 0.0, 0.0]), # Sets tendon to the right of the backbone
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
    "acceleration": [],
    "angular_velocity": [],
    "angular_acceleration": [],
    "rest_lengths": [],
    "densities": [],
    "volumes": [],
    "masses": [],
    "mass_second_moments_of_inertia": [],
    "inv_mass_second_moments_of_inertia": [],
    "rest_voronoi_lengths": [],
    "internal_forces": [],
    "internal_torques": [],
    "external_forces": [],
    "external_torques": [],
    "lengths": [],
    "tangents": [],
    "radii": [],
    "dilatations": [],
    "voronoi_dilatation": [],
    "dilatation_rates": [],
    "curvature": [],
    "rest_curvature": [],
    "sigma": [],
    "rest_sigma": [],
    "internal_stress": [],
    "internal_couple": [],
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

first_vertebra_node = 4
final_vertebra_node = 40
num_vertebrae = 10

vertebra_nodes = np.linspace(first_vertebra_node, final_vertebra_node, num_vertebrae, dtype=int)

# Build the figure
fig = plt.figure(figsize=(9, 7))
ax = fig.add_subplot(111, projection="3d")
# Force view onto the YZ plane (looking along X) when azim = 0, azim = 90 is looking down Y
ax.view_init(elev=0, azim=-90) 
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

# calculate final curvature (and radius of curvature) of robot
strain_vals = callback_data_rod1["curvature"][-1]  # shape (3, n_elements)
# Uncomment the line below if there are two segments in the rod
# strain_vals = strain_vals[:, : n_elements // 2] 
curvature_magnitudes = np.linalg.norm(strain_vals[:2, :], axis=0)  # shape (n_elements,)
avg_curvature = np.mean(curvature_magnitudes)
avg_radius_of_curvature = 1.0 / avg_curvature 
print("Final robot curvature (1/m): " , avg_curvature)
print("Final robot radius of curvature (m): " , avg_radius_of_curvature)

# Obtain position data for disk location at the end of simulation
x_disk = positions[-1, 0, vertebra_nodes]
y_disk = positions[-1, 1, vertebra_nodes]
z_disk = positions[-1, 2, vertebra_nodes]
print("Final disk positions (x, y, z):")
for i in range(num_vertebrae):
    print(f"Disk {i+1}: ({x_disk[i]}, {y_disk[i]}, {z_disk[i]})")

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
# anim.save("01-06_tdr_2seg_5N_xdir.mp4", fps=30, dpi=200, writer="ffmpeg")

plt.show()
