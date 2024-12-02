import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
from matplotlib.animation import FuncAnimation

# Define constants for the ellipse and fluid properties
a = 2.0          # semi-major axis length of the ellipse
b = 0.2          # semi-minor axis length of the ellipse
rho_s = 1.2      # density of the ellipse
rho_f = 1.0      # density of the fluid
g = 9.81         # gravity
I = 1.0          # moment of inertia for the ellipse

# Parameters for the forces and torque
A, B = 1.4, 1.0
mu, nu = 0.2, 0.2
CT, CR = 1.2, np.pi

# Derived non-dimensional parameters
beta = b / a
rho_star = rho_s / rho_f

# Set up time and simulation parameters
dt = 0.01        # time step
T = 5            # total simulation time
num_steps = int(T / dt)

# Initial conditions for position, orientation, and velocity
x, y = 0.0, 0.0        # initial position
theta = 0.0     # initial orientation
u, v, w = 0.0, 0.0, 0.0  # initial translational and rotational velocities


# goals 
theta_G = np.pi/4
x_g = 100
y_g = -50
tau = 1.0


# action is constant control torque
a_t = 0.1
tau_t = np.tanh(a_t) # in {-1, 1}.


# Arrays to store the results
x_values, y_values = [x], [y]
theta_values = [theta]

# Function to calculate forces and torque with stability checks
def calculate_forces(u, v, w):
    speed = np.sqrt(u**2 + v**2) + 1e-6  # Add small value to avoid division by zero
    F = (1 / np.pi) * (A - B * (u**2 - v**2) / (u**2 + v**2 + 1e-6)) * speed
    M = 0.2 * (mu + nu * abs(w)) * w
    Gamma = (2 / np.pi) * (CR * w - CT * (u * v / speed))
    return F, M, Gamma

# Run simulation loop
for _ in range(num_steps):
    # Calculate forces and torque
    F, M, Gamma = calculate_forces(u, v, w)

    # Update equations for u, v, w based on equations (2.1), (2.2), and (2.3)
    u_dot = ((I + beta**2) * v * w - Gamma * v - np.sin(theta) - F*u) / (I + beta**2)
    v_dot = (-(I + 1) * u * w + Gamma * u - np.cos(theta) - F*v) / (I + 1)
    w_dot = (-(1 - beta**2) * u * v - M + tau) / (0.25 * (I * (1 + beta**2) + 0.5 * (1 - beta**2)**2))

    # Update velocities and position/orientation
    u += u_dot * dt
    v += v_dot * dt
    w += w_dot * dt

    x += (u * np.cos(theta) - v * np.sin(theta)) * dt
    y += (u * np.sin(theta) + v * np.cos(theta)) * dt
    theta += w * dt

    # Store results for plotting
    x_values.append(x)
    y_values.append(y)
    theta_values.append(theta)

# Set up the figure and axis
fig, ax = plt.subplots(figsize=(8, 6))
ax.set_xlim(-3, 3)
ax.set_ylim(-3, 3)
ax.axhline(0, color='black', linewidth=0.5)  # Add x-axis
ax.axvline(0, color='black', linewidth=0.5)  # Add y-axis
ax.set_xlabel("x-position")
ax.set_ylabel("y-position")
ax.set_title("Animated Trajectory of an Ellipse in 2D")

# Create an ellipse patch
ellipse_patch = Ellipse((x_values[0], y_values[0]), 2*a, 2*b, angle=np.degrees(theta_values[0]), fill=True, color="green", alpha=0.5)
ax.add_patch(ellipse_patch)

# Animation update function
def update(frame):
    ellipse_patch.set_center((x_values[frame], y_values[frame]))
    ellipse_patch.angle = np.degrees(theta_values[frame])
    return ellipse_patch,

# Create the animation
ani = FuncAnimation(fig, update, frames=num_steps, blit=True, interval=20)

plt.show()
