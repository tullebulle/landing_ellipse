from load_policy_script import load_policy
from utils import simulate_ellipse
from utils import get_best_reward
import os
import json
import argparse
from utils import simulate_trajectory


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="A script that greets the user with the provided name.")
    
    # Add a string argument
    parser.add_argument("name", type=str, help="Run the latest or the best policy.")
    
    # Parse the command-line arguments
    args = parser.parse_args()

    file_path = "./saved_data/trajectories.json"

    if args.name == "best":
        best_idx, best_rew = get_best_reward()
        if os.path.exists(file_path):
            with open(file_path, 'r') as file:
                data = json.load(file)
        else:
            raise NameError("No file found")
        
        trajectory = data["data"][best_idx]["trajectory"] #dic with xvals, yvals and theta vals
        x_vals = trajectory["x_values"]
        y_vals = trajectory["y_values"]
        theta_vals = trajectory["theta_values"]
        simulate_trajectory(x_vals, y_vals, theta_vals)
    elif args.name == "latest":
        policy = load_policy("_korali_result/genLatest.json")
        solution = simulate_ellipse(policy)
