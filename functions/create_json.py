import json
import os

def create_json(solution, res_folder, model_name, istance_n):

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

    if model_name in ["SAT_Normal","SAT_Cluster", "MIP_DefaultSetting", "MIP_Feasibility", "MIP_Optimality", "SMT_Normal", "SMT_Cluster"]:
        if len(solution[d_key]["sol"])>0: 
            for i in range(len(solution[d_key]["sol"])):
                for j in range(len(solution[d_key]["sol"][i])):
                    solution[d_key]["sol"][i][j] += 1


    #create res folder if doesn't exist
    if not os.path.exists(res_folder):
        os.makedirs(res_folder)

    #control model name
    res_folder = str(res_folder)
    if  'CP' in model_name:
        #check if CP folder exists, create it if not
        if not os.path.exists(res_folder+"/CP"):
            os.makedirs(res_folder+"/CP")
        res_folder = res_folder+"/CP/"
    elif  'SAT' in model_name:
        #check if SAT folder exists, create it if not
        if not os.path.exists(res_folder+"/SAT"):
            os.makedirs(res_folder+"/SAT")
        res_folder = res_folder+"/SAT/"
    elif  'MIP' in model_name:
        #check if MIP folder exists, create it if not
        if not os.path.exists(res_folder+"/MIP"):
            os.makedirs(res_folder+"/MIP")
        res_folder = res_folder+"/MIP/"
    elif  'SMT' in model_name:
        #check if SMT folder exists, create it if not
        if not os.path.exists(res_folder+"/SMT"):
            os.makedirs(res_folder+"/SMT")
        res_folder = res_folder+"/SMT/"
    else:
        res_folder = res_folder+"Unknown"
    
    #create the json file with istance_n as name if it doesn't exist, if it exists append new solution
    if not os.path.exists(res_folder+str(istance_n)+'.json'):
        with open(res_folder+str(istance_n)+'.json', 'w') as f:
            #write the solution in the json file, each key must be a json serializable object
            json.dump(solution, f, indent=4)
    else:
        with open(res_folder+str(istance_n)+'.json', 'r') as f: 
            data = json.load(f)
            data.update(solution)
        with open(res_folder+str(istance_n)+'.json', 'w') as f:
            #write the solution in the json file, each key must be a json serializable object
            json.dump(data, f, indent=4)
    
