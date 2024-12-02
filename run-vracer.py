#!/usr/bin/env python3
import os
import sys
sys.path.append('./_model')
from env import *
import argparse
import numpy as np
import pdb; 
from utils import store_trajectory, get_best_reward
import multiprocessing



def run_ellipselanding():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--engine',
        help='NN backend to use',
        default='OneDNN',
        required=False)
    parser.add_argument(
        '--maxGenerations',
        help='Maximum Number of generations to run',
        default=10,
        type=int,
        required=False)    
    parser.add_argument(
        '--optimizer',
        help='Optimizer to use for NN parameter updates',
        default='Adam',
        type=str,
        required=False)
    parser.add_argument(
        '--learningRate',
        help='Learning rate for the selected optimizer',
        default=3e-3,
        type=float,
        required=False)
    parser.add_argument(
        '--concurrentWorkers',
        help='Number of concurrent workers / environments',
        default=multiprocessing.cpu_count(),
        type=int,
        required=False)

    args = parser.parse_args()

    print("Running Ellipse landing example with arguments:")
    print(args)
    ####### Defining Korali Problem

    ###clearing results file
    # with open('./stored_states/trajectories.json', 'w') as file:
    #     pass
    # with open('./stored_states/best_trajectory.json', 'w') as file:
    #     pass

    import korali
    k = korali.Engine()
    e = korali.Experiment()

    ### Defining the Cartpole problem's configuration
    e["Problem"]["Type"] = "Reinforcement Learning / Continuous"
    e["Problem"]["Environment Function"] = env
    e["Problem"]["Testing Frequency"] = 10

    e["Variables"][0]["Name"] = "Ellipse x-coordinate"
    e["Variables"][0]["Type"] = "State"

    e["Variables"][1]["Name"] = "Ellipse y-coordinate"
    e["Variables"][1]["Type"] = "State"

    e["Variables"][2]["Name"] = "Ellipse theta"
    e["Variables"][2]["Type"] = "State"

    e["Variables"][3]["Name"] = "Ellipse u"
    e["Variables"][3]["Type"] = "State"

    e["Variables"][4]["Name"] = "Ellipse v"
    e["Variables"][4]["Type"] = "State"

    e["Variables"][5]["Name"] = "Ellipse w"
    e["Variables"][5]["Type"] = "State"


    e["Variables"][6]["Name"] = "a_t"
    e["Variables"][6]["Type"] = "Action"
    e["Variables"][6]["Lower Bound"] = -1.0
    e["Variables"][6]["Upper Bound"] = +1.0
    e["Variables"][6]["Initial Exploration Noise"] = 0.3

    ### Defining Agent Configuration 

    e["Solver"]["Type"] = "Agent / Continuous / VRACER"
    e["Solver"]["Mode"] = "Training"
    e["Solver"]["Experiences Between Policy Updates"] = 1
    e["Solver"]["Episodes Per Generation"] = 10
    e["Solver"]["Concurrent Workers"] = args.concurrentWorkers

    e["Solver"]["Experience Replay"]["Start Size"] = 1000
    e["Solver"]["Experience Replay"]["Maximum Size"] = 10000
    e["Solver"]["Experience Replay"]["Off Policy"]["REFER Beta"]= 0.3

    e["Solver"]["Discount Factor"] = 0.99
    e["Solver"]["Learning Rate"] = args.learningRate
    e["Solver"]["Mini Batch"]["Size"] = 32
    e["Solver"]["State Rescaling"]["Enabled"] = True
    e["Solver"]["Reward"]["Rescaling"]["Enabled"] = True

    ### Configuring the neural network and its hidden layers

    e["Solver"]["Neural Network"]["Engine"] = args.engine
    e["Solver"]["Neural Network"]["Optimizer"] = args.optimizer
    e["Solver"]["Policy"]["Distribution"] = "Clipped Normal"

    e["Solver"]["Neural Network"]["Hidden Layers"][0]["Type"] = "Layer/Linear"
    e["Solver"]["Neural Network"]["Hidden Layers"][0]["Output Channels"] = 32

    e["Solver"]["Neural Network"]["Hidden Layers"][1]["Type"] = "Layer/Activation"
    e["Solver"]["Neural Network"]["Hidden Layers"][1]["Function"] = "Elementwise/Tanh"

    e["Solver"]["Neural Network"]["Hidden Layers"][2]["Type"] = "Layer/Linear"
    e["Solver"]["Neural Network"]["Hidden Layers"][2]["Output Channels"] = 32

    e["Solver"]["Neural Network"]["Hidden Layers"][3]["Type"] = "Layer/Activation"
    e["Solver"]["Neural Network"]["Hidden Layers"][3]["Function"] = "Elementwise/Tanh"

    ### Defining Termination Criteria

    e["Solver"]["Termination Criteria"]["Max Generations"] = args.maxGenerations

    ### Setting file output configuration

    e["File Output"]["Enabled"] = True
    e["File Output"]["Use Multiple Files"] = False
    e["File Output"]["Frequency"] = 5


    ### Running Experiment
    k.run(e)
    return e


def eval_run():
    # id = multiprocessing.current_process().pid
    e = run_ellipselanding()
    best_indx, best_return = get_best_reward()
    best_reward_run = e["Solver"]["Training"]['Best Return'][0]
    if best_reward_run > best_return:
        store_trajectory()


if __name__ == "__main__":

    num_processes = 2 #multiprocessing.cpu_count()
    # print(num_processes)
    

    for _ in range(num_processes):
        eval_run()

    best_indx, best_return = get_best_reward()
    print("DONE RUNNING EXPERIMENTS")
    print("Best Reward = ", best_return)