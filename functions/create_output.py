'''
This file will generate the output file for the given model and the given instance.

It will run the model on the instance and save the output in the output file, in res folder
'''

from CP.CP_runner import run_CP_model as CP_runner
from SAT.SAT_Model import SAT_MCP as SAT_runner
#from MIP.MIP_runner import MIP_MCP as MIP_runner
from SMT.SMT_Model_Int import SMT_MCP as SMT_runner
from functions.Instances_Reader import inst_read as IR

import pathlib as path

MODELS = ["CP_Gecode_1", "CP_Gecode_2", "CP_Gecode_3", "CP_Chuffed_1", 
          "CP_Chuffed_2","SAT_Normal","SAT_Cluster", "MIP_Normal", "MIP_Cluster",
          "SMT_Normal", "SMT_Cluster" ]

#function to get the function based on the model name
def run(
        model_name: str,
        instances : str,
        res_folder: path.Path,
        output_graph: bool = True,
        graph_folder: path.Path = path.Path("graph")
):
    inst_list = []
    instances = instances.split(',')

    #read all the instances from files
    for inst in instances:
        inst_list.append(IR(int(inst)))

    sol = []

    #call the differents models
    #control 
    for inst in inst_list:
        print(type(inst))
        if model_name == "CP_Gecode_1":
            sol.append(CP_runner(inst['inst'], inst['n'], inst['m'], "Gecode", "IndomainRandom"))
        elif model_name == "CP_Gecode_2":
            sol.append(CP_runner(inst['inst'], inst['n'], inst['m'], "Gecode", "IndomainRandom_RelAndRec"))
        elif model_name == "CP_Gecode_3":
            sol.append(CP_runner(inst['inst'], inst['n'], inst['m'], "Gecode", "IndomainMin_RelAndRec"))
        elif model_name == "CP_Chuffed_1":
            sol.append(CP_runner(inst['inst'], inst['n'], inst['m'], "Chuffed", "Smallest"))
        elif model_name == "CP_Chuffed_2":
            sol.append(CP_runner(inst['inst'], inst['n'], inst['m'], "Chuffed", "IndomainMin_RelAndRec"))
        elif model_name == "SAT_Normal":
            sol.append(SAT_runner(inst['n'], inst['m'], inst['s'], inst['l'], inst['D'],'default'))
        elif model_name == "SAT_Cluster":
            sol.append(SAT_runner(inst['n'], inst['m'], inst['s'], inst['l'], inst['D'],'clustering'))
        elif model_name == "MIP_Normal":
            # Uncomment the following line when MIP_runner is available
            # sol.append(MIP_runner(inst['n'], inst['m'], inst['s'], inst['l'], inst['D'],'default'))
            pass
        elif model_name == "MIP_Cluster":
            # Uncomment the following line when MIP_runner is available
            # sol.append(MIP_runner(inst['n'], inst['m'], inst['s'], inst['l'], inst['D'],'cluster'))
            pass
        elif model_name == "SMT_Normal":
            sol.append(SMT_runner(inst['n'], inst['m'], inst['s'], inst['l'], inst['D'],'default'))
        elif model_name == "SMT_Cluster":
            sol.append(SMT_runner(inst['n'], inst['m'], inst['s'], inst['l'], inst['D'],'clustering'))
        else:
            print(f"Model {model_name} not recognized.")
                # Check model_name
        
    print(sol)
    return