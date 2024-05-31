import json
import os

def create_json(solution, res_folder, model_name, istance_n):
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
    
