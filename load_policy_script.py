#!/usr/bin/env python3

import json
import numpy as np

def load_policy(korali_file: str) -> callable:
    with open(korali_file, "r") as f:
        e = json.load(f)

    nin = e["Problem"]["State Vector Size"]
    nout = e["Problem"]["Action Vector Size"]
    nout_tot = nout * 2 + 1 # V, mus, stds
    NN = e["Solver"]["Neural Network"]

    global weights
    try:
        weights = np.array(e["Solver"]["Training"]["Best Policy"]["Policy Hyperparameters"]["Policy"])
    except KeyError:
        try:
            weights = np.array(e["Solver"]["Training"]["Current Policies"]["Policy Hyperparameters"][0])
        except KeyError:
            weights = np.array(e["Solver"]["Training"]["Best Policy"]["Policy"])

    layers = []

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

    return evaluate


def is_policy_2D(korali_file: str) -> bool:
    with open(korali_file, "r") as f:
        e = json.load(f)

    nin = e["Problem"]["State Vector Size"]
    return nin == 2


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("korali_file", type=str, help="Korali file")
    args = parser.parse_args()

    p = load_policy(args.korali_file)
    print(p([1.0, 0.0, -1.0, 0.0, 0.0, 0.0]))