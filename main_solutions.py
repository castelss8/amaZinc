from z3 import *
from functions.create_output import MODELS
from functions.create_output import run as run
from functions import Instances_Reader as IR
import pathlib as path





# The main function of the project.
# It shows the possibilities for each solver and allow to run an instance on them.
def run_project():
    print("Combinatorial and Decision Making Optimization\nProject-aMaZINC\nDavide Bombardi davide.bombardi@studio.unibo.it\nGiorgia Castelli giorgia.castelli2@studio.unibo.it\nAlice Fratini alice.fratini2@studio.unibo.it\nMadalina Ionela Mone madalina.mone@studio.unibo.it")

    print("--- Choose the solving approach: ---\n")
    #modify name of the models
    for i, model_name in enumerate(MODELS):
        print(f"Press {i} for:\t{model_name}")    
    index = int(input())

    model_name = MODELS[index]
    print(f"--- Choose the instances [1:21] ---")
    print(f"A number or a comma separated list of numbers (e.g. 1,2,3)")

    instances = input()

    res_folder = path.Path("res")

    print(f"Model name: {model_name}")
    print(f"Instances: {instances}")
    #print(res_folder)

    run(model_name,instances, res_folder)


if __name__ == '__main__':
    run_project()