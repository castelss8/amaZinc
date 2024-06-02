'''
This file will generate the output file for the given model and the given instance.

It will run the model on the instance and save the output in the output file, in res folder
'''

from CP.CP_runner import run_CP_model as CP_runner
from SAT.SAT_Model import SAT_MCP as SAT_runner
from MIP.MIP_Model import MIP_MCP as MIP_model
from MIP.MIP_runner import MIP_MCP as MIP_runner
from SMT.SMT_Model_Int import SMT_MCP as SMT_runner
from functions.Instances_Reader import inst_read as IR
from functions.create_json import create_json as CJ

import pathlib as path

MODELS = ["CP_Gecode_1", "CP_Gecode_2", "CP_Gecode_3",
          "CP_Chuffed_1", "CP_Chuffed_2",
          "SAT_Normal","SAT_Cluster", 
          "MIP_Gurobi_DefaultSetting", "MIP_Gurobi_Feasibility", "MIP_Gurobi_Optimality", "MIP_Gurobi_Bounding",
          "MIP_2_DefaultSetting", "MIP_2_Feasibility", "MIP_2_Optimality",
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
        elif model_name == "MIP_Gurobi_DefaultSetting":
            sol_tmp = MIP_runner(inst['n'], inst['m'], inst['s'], inst['l'], inst['D'],focus=0)
            sol.append(sol_tmp)
            CJ(sol_tmp, res_folder, model_name, inst['inst'])
        elif model_name == "MIP_Gurobi_Feasibility":
            sol_tmp = MIP_runner(inst['n'], inst['m'], inst['s'], inst['l'], inst['D'],focus=1)
            sol.append(sol_tmp)
            CJ(sol_tmp, res_folder, model_name, inst['inst'])
        elif model_name == "MIP_Gurobi_Optimality":
            sol_tmp = MIP_runner(inst['n'], inst['m'], inst['s'], inst['l'], inst['D'],focus=2)
            sol.append(sol_tmp)
            CJ(sol_tmp, res_folder, model_name, inst['inst'])
        elif model_name == "MIP_Gurobi_Buonding":
            sol_tmp = MIP_runner(inst['n'], inst['m'], inst['s'], inst['l'], inst['D'],focus=3)
            sol.append(sol_tmp)
            CJ(sol_tmp, res_folder, model_name, inst['inst'])
        elif model_name == "MIP_2_DefaultSetting":
            sol_tmp = MIP_model(inst['n'], inst['m'], inst['s'], inst['l'], inst['D'],'DefaultSetting')
            sol.append(sol_tmp)
            CJ(sol_tmp, res_folder, model_name, inst['inst'])
        elif model_name == "MIP_2_Feasibility":
            sol_tmp = MIP_model(inst['n'], inst['m'], inst['s'], inst['l'], inst['D'],'Feasibility')
            sol.append(sol_tmp)
            CJ(sol_tmp, res_folder, model_name, inst['inst'])
        elif model_name == "MIP_2_Optimality":
            sol_tmp = MIP_model(inst['n'], inst['m'], inst['s'], inst['l'], inst['D'],'Optimality')
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
        elif 'Cluster' in model_name:
            d_key = 'clustering'
        elif 'DefaultSetting' in model_name:
            d_key = 'DefaultSetting'
        elif 'Feasibility' in model_name:
            d_key = 'Feasibility'
        elif 'Optimality' in model_name:
            d_key = 'Optimality'
        
        '''
        if model_name in ["SAT_Normal","SAT_Cluster", "MIP_DefaultSetting", "MIP_Feasibility", "MIP_Optimality", "SMT_Normal", "SMT_Cluster"] and len(sol[-1][d_key]["sol"])>0: 
            for i in range(len(sol[-1][d_key]["sol"])):
                for j in range(len(sol[-1][d_key]["sol"][i])):
                    sol[-1][d_key]["sol"][i][j] += 1
        '''

    print(sol)
    return
