'''
This file will generate the output file for the given model and the given instance.

It will run the model on the instance and save the output in the output file, in res folder
'''

from CP.CP_runner import run_CP_model as CP_runner
from SAT.SAT_Model import SAT_MCP as SAT_runner
from MIP.MIP_runner import MIP_MCP as MIP_runner
from SMT.SMT_Model_Int import SMT_MCP as SMT_runner
from functions.Instances_Reader import inst_read as IR
from functions.create_json import create_json as CJ

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
        #print(type(inst))
        if model_name == "CP_Gecode_1":
            sol_tmp = CP_runner(inst['inst'], inst['n'], inst['m'], "Gecode", "IndomainRandom")
            sol.append(sol_tmp)
            CJ(sol_tmp, res_folder, model_name, inst['inst'])
        elif model_name == "CP_Gecode_2":
            sol_tmp = CP_runner(inst['inst'], inst['n'], inst['m'], "Gecode", "IndomainRandom_RelAndRec")
            sol.append(sol_tmp)
            CJ(sol_tmp, res_folder, model_name, inst['inst'])
        elif model_name == "CP_Gecode_3":
            sol_tmp = CP_runner(inst['inst'], inst['n'], inst['m'], "Gecode", "IndomainMin_RelAndRec")
            sol.append(sol_tmp)
            CJ(sol_tmp, res_folder, model_name, inst['inst'])
        elif model_name == "CP_Chuffed_1":
            sol_tmp = CP_runner(inst['inst'], inst['n'], inst['m'], "Chuffed", "Smallest")
            sol.append(sol_tmp)
            CJ(sol_tmp, res_folder, model_name, inst['inst'])
        elif model_name == "CP_Chuffed_2":
            sol_tmp = CP_runner(inst['inst'], inst['n'], inst['m'], "Chuffed", "InputOrder")
            sol.append(sol_tmp)
            CJ(sol_tmp, res_folder, model_name, inst['inst'])
        elif model_name == "SAT_Normal":
            sol_tmp = SAT_runner(inst['n'], inst['m'], inst['s'], inst['l'], inst['D'],'default')
            sol.append(sol_tmp)
            CJ(sol_tmp, res_folder, model_name, inst['inst'])
        elif model_name == "SAT_Cluster":
            sol_tmp = SAT_runner(inst['n'], inst['m'], inst['s'], inst['l'], inst['D'],'clustering')
            sol.append(sol_tmp)
            CJ(sol_tmp, res_folder, model_name, inst['inst'])
        elif model_name == "MIP_Normal":
            # Uncomment the following line when MIP_runner is available
            sol_tmp = MIP_runner(inst['n'], inst['m'], inst['s'], inst['l'], inst['D'],'default')
            sol.append(sol_tmp)
            CJ(sol_tmp, res_folder, model_name, inst['inst'])
        elif model_name == "MIP_Cluster":
            # Uncomment the following line when MIP_runner is available
            sol_tmp = MIP_runner(inst['n'], inst['m'], inst['s'], inst['l'], inst['D'],'clustering')
            sol.append(sol_tmp)
            CJ(sol_tmp, res_folder, model_name, inst['inst'])
        elif model_name == "SMT_Normal":
            sol_tmp = SMT_runner(inst['n'], inst['m'], inst['s'], inst['l'], inst['D'],'default')
            sol.append(sol_tmp)
            CJ(sol_tmp, res_folder, model_name, inst['inst'])
        elif model_name == "SMT_Cluster":
            sol_tmp = SMT_runner(inst['n'], inst['m'], inst['s'], inst['l'], inst['D'],'clustering')
            sol.append(sol_tmp)
            CJ(sol_tmp, res_folder, model_name, inst['inst'])
        else:
            print(f"Model {model_name} not recognized.")
                # Check model_name
        
        if 'Normal' in model_name:
            d_key = 'default'
        else:
            d_key = 'clustering'

        if model_name in ["SAT_Normal","SAT_Cluster", "MIP_Normal", "MIP_Cluster", "SMT_Normal", "SMT_Cluster"] and len(sol[-1][d_key]["sol"])>0: 
            for i in range(len(sol[-1][d_key]["sol"])):
                for j in range(len(sol[-1][d_key]["sol"][i])):
                    sol[-1][d_key]["sol"][i][j] += 1


    print(sol)
    return
