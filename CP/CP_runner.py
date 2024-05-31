import os
import json
import math
import re
from pathlib import Path

# util function for manipulating strings 

def find_between(s: str, first: str, last: str):
        start = s.index(first) + len(first)
        end = s.index(last, start)
        return s[start:end]


# function running the CP model on a specific instance, whith a specific solver and a specific search strategy

def run_CP_model(instance: int, n: int, m: int, solver: str, search: str):
    
    '''
        input:
            instance: int = number of the instance
            n: int = number of items as defined in instance
            m: int = number of couriers as defined in instance
            solver: str = solver to be used in the run ("Gecode" or "Chuffed")
            search: str = search strategy to be used in the run;
                            for solver = "Gecode":
                                - "IndomainRandom"
                                - "IndomainRandom_RelAndRec"
                                - "IndomainMin_RelAndRec"
                            for solver = "Chuffed":
                                - "Smallest"
                                - "InputOrder"

        output: dictionary to be used in "res\CP\inst.json" for solution for the model, as described in the project description
    '''

    # instance path
    if instance <=9:
        instance_path = Path("./CP/Preprocessed_Instances/inst0"+str(instance)+".dzn")
    else:
        instance_path = Path("./CP/Preprocessed_Instances/inst"+str(instance)+".dzn")
    

    # model path

    if solver == "Gecode" and search == "IndomainRandom":
        model_path = Path("./CP/CP_model_1.mzn")
    elif solver == "Gecode" and search == "IndomainRandom_RelAndRec":
        model_path = Path("./CP/CP_model_2.mzn")
    elif solver == "Gecode" and search == "IndomainMin_RelAndRec":
        model_path =  Path("./CP/CP_model_3.mzn")
    elif solver == "Chuffed" and search == "Smallest":
        model_path = Path("./CP/CP_model_4.mzn")
    elif solver == "Chuffed" and search == "InputOrder":
        model_path =Path("./CP/CP_model_5.mzn")

    # run of the model
    with open(model_path, 'r') as f:
        pass

    args = "minizinc --solver "+solver+" "+str(model_path)+" "+str(instance_path)+" --solver-time-limit 300000 --json-stream --output-time --intermediate"
    print(args)


    # read of json-stream

    results = []

    minizinc_output = os.popen(args).readlines()
    print('popen')

    for i in minizinc_output:
        try:
            results.append(json.loads(i))
        except json.JSONDecodeError:
            pass

    # initializes sol: list of lists = best solution found
    
    solutions = [j for j in results if j["type"] == "solution"]


    # case when no solution is found in 300 seconds - defines all outputs
    
    if len(solutions) == 0:
        time = 300
        optimal = False
        obj = "N/A"
        sol = []

    
    # all other cases - defines obj: int = best found value for objective function; defines sol: list of list = best solution found
    elif len(solutions) > 0:
        sol = [[] for i in range (1,m+1)]

        best_solution = solutions[-1]
        pred = re.split(", ", find_between(best_solution["output"]["default"],"pred = [", str("]\n")))
        obj = int(find_between(best_solution["output"]["default"],"distance = ", str("\n")))

        for i in range(1, m+1):
            k = int(pred[n+m+i-1])
            while int(k)<(n+1):
                sol[i-1]=[k]+sol[i-1]
                k = int(pred[k-1])


    # all other cases - defines time: int = running time (in seconds); defines optimal: bool = false iff time = 300 iff the model couldn't prove that sol is optimal    
        
        optimal_solution = [j for j in results if (j["type"] == "status" and j["status"] == "OPTIMAL_SOLUTION")]
        
        if len(optimal_solution) == 0:
            time = 300
            optimal = False
        else:
            time = math.floor(int(optimal_solution[-1]["time"])/1000)
            optimal = True

    # print of the dictionary
    print({solver+"_"+search: {"time": time, "optimal": optimal, "obj": obj, "sol": sol}})



    return {solver+"_"+search: {"time": time, "optimal": optimal, "obj": obj, "sol": sol}}
