import numpy as np
import json
import os
# from solver_ODE_RL import solve_example_RL
# from animation import animate_trajectories
from load_policy_script import load_policy
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
from matplotlib.animation import FuncAnimation

def simulate_ellipse(policy):

    # Define constants for the ellipse and fluid properties
    a = 1.0          # semi-major axis length of the ellipse
    b = 0.5          # semi-minor axis length of the ellipse
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
    dt = 0.5        # time step
    T = 100            # total simulation time
    num_steps = int(T / dt)

    # Initial conditions for position, orientation, and velocity
    x, y = 0.0, 0.0        # initial position
    theta = 0.0     # initial orientation
    u, v, w = 0.0, 0.0, 0.0  # initial translational and rotational velocities

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
    done = False
    step = 0
    while step < num_steps and not done:
        # Calculate forces and torque
        F, M, Gamma = calculate_forces(u, v, w)


        state = np.array([x, y, theta, u, v, w])
        action = policy(state)
        tau_t = np.tanh(action)

        # Update equations for u, v, w based on equations (2.1), (2.2), and (2.3)
        u_dot = ((I + beta**2) * v * w - Gamma * v - np.sin(theta) - F*u) / (I + beta**2)
        v_dot = (-(I + 1) * u * w + Gamma * u - np.cos(theta) - F*v) / (I + 1)
        w_dot = float((-(1 - beta**2) * u * v - M + tau_t) / (0.25 * (I * (1 + beta**2) + 0.5 * (1 - beta**2)**2)))

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
        step += 1
        if y < -20:
            done = True

    # Set up the figure and axis
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.set_xlim(-5, 50)
    ax.set_ylim(-50, 5)
    ax.axhline(0, color='black', linewidth=0.5)  # Add x-axis
    ax.axvline(0, color='black', linewidth=0.5)  # Add y-axis
    ax.axhline(-20, color='red', linewidth=0.5)  # Add x-axis
    ax.axvline(20, color='red', linewidth=0.5)  # Add y-axis
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
    ani = FuncAnimation(fig, update, frames=step, blit=True, interval=20)

    plt.show()
    return {"x_values": x_values, "y_values":y_values, "theta_values": theta_values }



def add_trajectory(reward, trajectory, weights, file_path='./saved_data/trajectories.json'):
    # Check if the file already exists
    if os.path.exists(file_path):
        with open(file_path, 'r') as file:
            data = json.load(file)
    else:
        data = {"data": []}
    
    # Add the new entry
    new_entry = {
        "reward": reward,
        "trajectory": trajectory,
        "weights": weights
    }
    data["data"].append(new_entry)
    
    # Save back to the file
    with open(file_path, 'w') as file:
        json.dump(data, file, indent=4)

def load_policy(korali_file: str) -> callable:
    with open(korali_file, "r") as f:
        e = json.load(f)

    nin = e["Problem"]["State Vector Size"]
    nout = e["Problem"]["Action Vector Size"]
    nout_tot = nout * 2 + 1 # V, mus, stds
    NN = e["Solver"]["Neural Network"]

    global weights
    try:
        # weights = np.array(e["Solver"]["Training"]["Best Policy"]["Policy Hyperparameters"]["Policy"])
        weights = np.array(e["Solver"]["Testing"]['Best Policies']["Policy Hyperparameters"][0])
    except KeyError:
        try:
            weights = np.array(e["Solver"]["Training"]["Current Policies"]["Policy Hyperparameters"][0])
            # print(e["Solver"]["Training"].keys())
            # print(e["Solver"]["Training"]['Current Policies']["Policy Hyperparameters"])
            

        except KeyError:
            weights = np.array(e["Solver"]["Training"]["Best Policy"]["Policy"])

    layers = []
    weights_copy = weights.copy()
    best_reward = e["Solver"]["Training"]['Best Return']

    action_scales = np.array(e["Solver"]["Action Scales"])
    action_shifts = np.array(e["Solver"]["Action Shifts"])
    action_min = np.array(e["Solver"]["Action Lower Bounds"])
    action_max = np.array(e["Solver"]["Action Upper Bounds"])

    def linear(n0, n1):
        global weights
        A = weights[:n0*n1]
        weights = weights[n0*n1:]
        b = weights[:n1]
        weights = weights[n1:]
        A = A.reshape(n1,n0)
        return lambda x: np.matmul(A, x) + b

    nprev = nin
    for desc in NN['Hidden Layers']:
        layer_type = desc["Type"]
        if layer_type == "Layer/Linear":
            n = desc["Output Channels"]
            layers.append(linear(nprev, n))
        elif layer_type == "Layer/Activation":
            n = nprev
            function = desc["Function"]
            if function == "Elementwise/Tanh":
                layers.append(lambda x: np.tanh(x))
            else:
                raise NotImplementedError(f"not implemented activation function {function}")
        else:
            raise NotImplementedError(f"not implemented layer type {layer_type}")

        nprev = n

    layers.append(linear(nprev, nout_tot))

    sr = e["Solver"]["State Rescaling"]

    if sr["Enabled"] == 1:
        state_shifts = np.array(sr["Means"]).flatten()
        state_scales = np.array(sr["Sigmas"]).flatten()
    else:
        state_shifts = np.zeros(nin)
        state_scales = np.ones(nin)

    def evaluate(x):
        x = np.array(x)
        x = (x - state_shifts) / state_scales
        for layer in layers:
            x = layer(x)
        a = x[1:1+nout]
        #a = action_scales * a + action_shifts
        a = a + action_shifts # TODO check with korali people;
        a = np.maximum(a, action_min)
        a = np.minimum(a, action_max)
        return a

    return evaluate, best_reward, weights_copy


def store_trajectory():
    policy, best_reward, weights = load_policy("_korali_result/genLatest.json")
    solution = simulate_ellipse(policy)
    add_trajectory(best_reward, solution, weights.tolist())



#making a fucntion that searches throught the rewards in the trajectories.json file and returns the best reward
def get_best_reward(file_path='./saved_data/trajectories.json'):
    # Check if the file already exists
    if os.path.exists(file_path):
        with open(file_path, 'r') as file:
            data = json.load(file)
    else:
        data = {"data": []}
    
    # Add the new entry
    best_reward = float('-inf')
    best_indx = 0
    for i, entry in enumerate(data["data"]):
        if entry["reward"][0] > best_reward:
            best_reward = entry["reward"][0]
            best_indx = i
    return best_indx, best_reward





def simulate_trajectory(x_values, y_values, theta_values):
    # Define constants for the ellipse and fluid properties
    a = 1.0          # semi-major axis length of the ellipse
    b = 0.5          # semi-minor axis length of the ellipse

    # Set up the figure and axis
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.set_xlim(-5, 50)
    ax.set_ylim(-50, 5)
    ax.axhline(0, color='black', linewidth=0.5)  # Add x-axis
    ax.axvline(0, color='black', linewidth=0.5)  # Add y-axis
    ax.axhline(-20, color='red', linewidth=0.5)  # Add x-axis
    ax.axvline(20, color='red', linewidth=0.5)  # Add y-axis
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
    ani = FuncAnimation(fig, update, frames=len(x_values), blit=True, interval=20)

    plt.show()