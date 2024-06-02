import numpy as np
import sys

sys.path.append('/Users/mange/Documents/GitHub/Uni/amaZinc')
from functions import sol_functions as sf
from functions import clustering as cl
from mip import Model, minimize, INTEGER, BINARY, xsum, Var, OptimizationStatus, CBC
import time


def MIP_Model(n, m, s, l, D, emphasis):
    
    model = Model(solver_name='CBC')
    model.emphasis = emphasis

    #Decision variables
    '''
    vehicle matrix 
    col = packs
    rows = courriers 
    each rows represent the list of packs foreach courrier
    #create a [m x (n+m)] matrix v. v[i][j]==True if the i-th courier takes the j-th item (or starting/ending point)
        ''' 
    cour_pred = [[[model.add_var(f'cour({c})_pred({i})_is{j}', var_type=BINARY)  for i in range(n+1)] for j in range(n+1)] for c in range(m)]
    
    # 
    # each pack have to be carried by exactly one courrier + each item is predecessor of exactly one item + 
    # + each item has 1 predecessor
    for i in range(n):
        model += xsum(cour_pred[c][i][j] for c in range(m) for j in range(n+1)) == 1        
    for j in range(n):
        model += xsum(cour_pred[c][i][j] for c in range(m) for i in range(n+1)) == 1
    for c in range(m):
        for i in range(n+1):
            for j in range(n+1):
                model += cour_pred[c][i][j] <= xsum(cour_pred[c][h][i] for h in range(n+1))
    

    for c in range(m):
        model += xsum(cour_pred[c][i][n] for i in range(n+1)) == 1
        model += xsum(cour_pred[c][n][j] for j in range(n+1)) == 1
    
    #Constraint fot the capacity of each courrier
    for c in range(m):
        model += xsum(cour_pred[c][i][j]*s[i] for i in range(n) for j in range(n+1)) <= l[c]
    
   
    #array to avoid loops, lenght n
    avoid_loops = [model.add_var(f'avoid_loops_({i})', lb=1, ub=n, var_type=INTEGER) for i in range(n)]

    for i in range(n):
        for j in range(n):
            #if pred[i][j] == 1 then avoid_loops[i] - avoid_loops[j] >= 1
            #if pred[i][j] == 0 then 0 - 0 - 0 >= 0
            model += avoid_loops[i] - avoid_loops[j]  >=  xsum(cour_pred[c][i][j] for c in range(m))*n - n + 1
    
    dist_ub = sum([D[i][i+1] for i in range(n)]) + D[n][0]
    #Vector of distances for each courier
    dist_vector = [model.add_var(f'distance_({c})', var_type=INTEGER, lb=0, ub=dist_ub) for c in range(m)]
    
    #Constraint for the distance of each courier
    for c in range(m):
        #print([cour_pred[c][i][j] for i in range(n+1) for j in range(n+1)])
        model += dist_vector[c] == xsum((cour_pred[c][i][j] * D[j][i]) for i in range(n+1) for j in range(n+1)) 

    # Creare la variabile max_dist e impostarla come massimo di dist_vector
    max_dist = model.add_var('max_dist', var_type=INTEGER, lb=0, ub=dist_ub)

    # Constraint for the max distance: takes the maximum value of dist_vector
    for c in range(m):
        model += dist_vector[c] <= max_dist
    
    model.objective = max_dist
    model.verbose = 0
    
    return model, cour_pred, max_dist 

def MIP_MCP(n, m, s, l, D, approaches: list, total_time=300):
  
    if 'DefaultSetting' in approaches:
        emphasis = 0
    elif 'Feasibility' in approaches:
        emphasis = 1
    elif 'Optimality' in approaches:
        emphasis = 2

    model, cour_pred, max_dist = MIP_Model(n, m, s, l, D, emphasis=emphasis)
    
    solutions = {}
    
    # Time Limit of 5 minutes

    initial_time = time.time()

    status = model.optimize(max_seconds=300)    

    

    final_time = min(300, time.time()-initial_time)

    if model.num_solutions == 0:
            solutions[approaches] = {'time' : 300 , 'optimal' : False , 'obj' : 'N/A' , 'sol' : []}
    else: 
        item_pred_mid  = [(i,j) for j in range(n) for i in range(n) for c in range(m) if cour_pred[c][i][j].x]
        item_pred_start = [(i, n+c) for i in range(n) for c in range(m) if cour_pred[c][i][n].x] 
        item_pred_end = [(n+c, j) for j in range(n) for c in range(m) if cour_pred[c][n][j].x] 
        item_pred_still = [(n+c,n+c) for c in range(m) if cour_pred[c][n][n].x]
        item_pred = item_pred_start + item_pred_mid + item_pred_end + item_pred_still
        print(item_pred)

        tmp  = [(c, j) for i in range(n+1) for c in range(m) for j in range(n+1) if cour_pred[c][i][j].x] 
        cour_item = [(c, i) for i in range(n) for c in range(m) for j in range(n+1) if cour_pred[c][i][j].x] 
        print(cour_item)
        print(tmp)
        solutions[approaches] = {'time' : int(final_time) , 'optimal' : status == OptimizationStatus.OPTIMAL , 'obj' : int(model.objective_value), 'sol' : sf.solution_maker(item_pred, cour_item, n, m)}
    
    return solutions
