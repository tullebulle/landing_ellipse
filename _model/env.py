#!/usr/bin/env python3
from ellipse import *
import pdb
import numpy as np
import json
import os


ellipse = Ellipse()
maxSteps = 100

def env(s):
  # Initializing environment and random seed
  sampleId = s["Sample Id"]
  ellipse.reset(sampleId)
  s["State"] = ellipse.getState().tolist()
  step = 0
  done = False


  filename = "./stored_states/trajectories.json"
  trajectory = {
          "sample_id": sampleId,
          "states": [],
          "actions": [],
          "rewards": []
      }

  while not done and step < maxSteps:

    # Getting new action
    s.update()

    trajectory["states"].append(s["State"])
    
    # Performing the action
    # print(s["State"])
    # print(s["Action"])
    done = ellipse.advance(s["Action"])

    trajectory["actions"].append(s["Action"])

    
    # Getting Reward
    s["Reward"] = ellipse.getReward()
    trajectory["rewards"].append(s["Reward"])

    
    # Storing New State
    s["State"] = ellipse.getState().tolist()
    
    # Advancing step counter
    step = step + 1

  # Setting finalization status
  if (ellipse.isOver()):
    s["Termination"] = "Terminal"
  else:
    s["Termination"] = "Truncated"
  

# ## TESTING
test_s = {
            "Sample Id": 123,
            "Action": [0.1],  # Example action, could be any valid action
        }

# t_arr = np.linspace(0,(500*0.04), 500)
sol = env(test_s)