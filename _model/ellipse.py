#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import ode

class Ellipse:
    def __init__(self):
        # Constants for the ellipse and fluid properties
        self.a = 1.0          # semi-major axis
        self.b = 0.5          # semi-minor axis
        self.rho_s = 1.2      # density of the ellipse
        self.rho_f = 1.0      # density of the fluid
        self.g = 9.81         # gravity
        self.I = 1.0          # moment of inertia
        self.tau = 0.0        # control torque
        self.K = 50.0
        self.beta = self.b / self.a
        self.rho_star = self.rho_s / self.rho_f
        
        # Simulation parameters
        self.dt = 1.0         # time step
        self.t = 0
        self.step = 0
        self.s = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])  # initial conditions for x, y, thera, u, v, w
        self.prev_state = self.s
        self.theta_goal = np.pi / 4              # target orientation theta_G
        self.goal = np.array([20.0, -30.0])    # goal position x_g, y_g
        
        # Constants for forces and torque
        self.A, self.B = 1.4, 1.0
        self.mu, self.nu = 0.2, 0.2
        self.CT, self.CR = 1.2, np.pi
        self.epsilon = 1.0                      # small constant to avoid division by zero in speed
        
        # ODE solver
        self.ODE = ode(self.system).set_integrator('dopri5')

    def reset(self, seed):
        np.random.seed(seed)
        self.s = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])  # [x, y, u, v, w, theta]
        self.t = 0
        self.step = 0

    def getReward(self):
        # Calculate the reward based on the distance to the goal
        current_dist = np.linalg.norm(self.s[:2] - self.goal)
        prev_dist = np.linalg.norm(self.prev_state[:2] - self.goal)

        if self.reached_goal() or self.isOver():
            return -0.01*self.tau**2 * self.dt + self.K*(np.exp(-(current_dist)**2) + np.exp(-10*(self.s[-1] - self.theta_goal)**2))
        return -0.01*self.tau**2 * self.dt + prev_dist - current_dist
    
    def calculate_forces(self, u, v, w):
        # Calculate the speed and force terms with stability check
        speed = np.sqrt(u**2 + v**2) + 1e-6  # Avoid division by zero
        F = (1 / np.pi) * (self.A - self.B * (u**2 - v**2) / (u**2 + v**2 + 1e-6)) * speed
        M = 0.2 * (self.mu + self.nu * abs(w)) * w
        Gamma = (2 / np.pi) * (self.CR * w - self.CT * (u * v / speed))
        return F, M, Gamma
    
    def system(self, t, state, tau):
        x, y, theta, u, v, w = state
        
        # Calculate forces and torque
        F, M, Gamma = self.calculate_forces(u, v, w)

        # Differential equations for u, v, and w
        u_dot = ((self.I + self.beta**2) * v * w - Gamma * v - np.sin(theta) - F * u) / (self.I + self.beta**2)
        v_dot = (-(self.I + 1) * u * w + Gamma * u - np.cos(theta) - F * v) / (self.I + 1)
        w_dot = (-(1 - self.beta**2) * u * v - M + tau) / (0.25 * (self.I * (1 + self.beta**2) + 0.5 * (1 - self.beta**2)**2))
        
        # Position and orientation updates
        x_dot = u * np.cos(theta) - v * np.sin(theta)
        y_dot = u * np.sin(theta) + v * np.cos(theta)
        theta_dot = w

        return [x_dot, y_dot, theta_dot, u_dot, v_dot, w_dot]

    def advance(self, action):
        self.tau = np.tanh(action[0])  # Control torque bounded in [-1, 1]

        # Set up the ODE with the current state and time
        self.ODE.set_initial_value(self.s, self.t).set_f_params(self.tau)
        new_state = self.ODE.integrate(self.t + self.dt)
        
        # Update state and time
        self.prev_state = self.s
        self.s = new_state
        self.t += self.dt
        self.step += 1

        return self.reached_goal()  # Check if goal is reached

    def reached_goal(self):
        # Calculate the Euclidean distance to goal and return if within epsilon
        dist = np.linalg.norm(self.s[:2] - self.goal)
        return dist < self.epsilon
    
    def isOver(self):
        return self.s[1] < self.goal[1]

    def getState(self):
        return self.s