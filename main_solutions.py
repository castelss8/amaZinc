from z3 import *

import numpy as np
from itertools import combinations
from utils import *
import math
import time
from importlib import reload
from functions.create_output import MODELS
from functions.create_output import run as run
from functions import Instances_Reader as IR
import pathlib as path



# The main function of the project.
# It shows the possibilities for each solver and allow to run an instance on them.
def run_project():
    print("Combinatorial and Decision Making Optimization\nProject 1\n Student, Group 1, 2024")

    print("--- Choose the solving approach: ---\n")
    #modify name of the models
    for i, model_name in enumerate(MODELS):
        print(f"Press {i} for: {model_name}")    
    index = int(input())

    model_name = MODELS[index]
    print(f"--- Choose the instances [01:21] ---")
    print(f"\n~~ A number, a comma separated list or 'all'~~")

    instances = input()

    print(f"\n--- Choose 1 if need output graph or 0 if not ---")
    output_graph = int(input()) == 1

    res_folder = path.Path("res")
    graph_folder = path.Path("graph")

    print(model_name)
    print(instances)
    print(res_folder)

    run(model_name,instances, res_folder,output_graph, graph_folder)


if __name__ == '__main__':
    run_project()