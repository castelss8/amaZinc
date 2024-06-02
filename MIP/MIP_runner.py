import gurobipy as gp
from gurobipy import GRB, quicksum
import numpy as np
from functions import sol_functions as sf
from functions import clustering as cl
import copy
import numpy as np


'''
    Defining the creation of the model for the default approach
'''

def MIP_Model(n, m, s, l, D, focus):

    model = gp.Model("MIP_Model_Gurobi")

    #Decision variables

    '''
    vehicle matrix 
    col = packs
    rows = courriers 
    each rows represent the list of packs foreach courrier
    #create a [m x (n+m)] matrix v. v[i][j]==True if the i-th courier takes the j-th item (or starting/ending point)
    ''' 
    cour = model.addMVar(shape=(m, n+m), vtype=GRB.BINARY, name="cour")
    model.update()

    #Constraints for cour

    #Each pack have to be carried by exactly one courrier
    for j in range(n):
        model.addConstr(quicksum(cour[i][j] for i in range(m)) == 1, f"cour_{j}")
    
    #The last m-columns of cour are fixed: the courier c starts and ends at the deposit n+c
    for j in range(m):
        for i in range(m):
            if j == i:
                model.addConstr(cour[i][n+j] == 1, f"cour_{n+j}")
            else:
                model.addConstr(cour[i][n+j] == 0, f"cour_{n+j}")
    
    #Constraint fot the capacity of each courrier
    
    #The sum of the weights of the items carried by the courier c must be less than or equal to the capacity of the courier c
    for c in range(m):
        model.addConstr(quicksum(cour[c][i]*s[i] for i in range(n)) <= l[c], f"peso_{c}")

    '''
    pred matrix
    pred[i][j] = true if the j-th item is the predecessor of the i-th item  
    '''
    pred = model.addMVar(shape=(n+m, n+m), vtype=GRB.BINARY, name="pred")
    model.update()

    #Constraints for the predecessor matrix

    #Each item/ending point has exactly one predecessor and each item/starting point is predecessor of exactly one other item. 
    for i in range(n+m):
        col_i = [pred[j][i] for j in range(n+m)]
        model.addConstr(quicksum(col_i) == 1, f"PC_{i}")
        model.addConstr(quicksum(pred[i]) == 1, f"PR_{i}")

    #if the courier c has the item i and item j is the predecessor of item i then c has also item j
    for c in range(m):
        for i in range(n+m):
            for j in range(n+m):
                model.addConstr(cour[c][j] >= cour[c][i] + pred[i][j] - 1, f"consistency_{c}_{i}_{j}")
    
    #Vector to avoid loops, lenght n
    avoid_loops = [model.addVar(lb=1, ub=n, vtype=GRB.INTEGER, name=f"avoid_loops") for _ in range(n)]
    model.update()
    for i in range(n):
        for j in range(n):
            model.addConstr(avoid_loops[i] - avoid_loops[j] >= n*(pred[i][j]) - n+1, f"avoid_loops_{i}_{j}")
    
    #Vector of distances for each courier
    dist_vector = []
    for courier in range(m):
        dist_vector.append(model.addVar(vtype=GRB.INTEGER, name="dist_vector"))
    
    model.update()

    '''
    #Constraint for the distance of each courier
    for courier in range(m):
        model.addConstr(dist_vector[courier] - (quicksum((pred[item][n+courier]) * D[n][item] for item in range(n))+
                                                quicksum((pred[n+courier][item]) * D[item][n] for item in range(n))+
                                                quicksum((cour[courier][i] * pred[i][j]) * D[j][i] for i in range(n) for j in range(n))) == 0, 
                                                f"dist_vector_{courier}")
    '''
    # nuova variabile decisionale
    prod = {}
    for courier in range(m):
        for i in range(n):
            for j in range(n):
                prod[courier, i, j] = model.addVar(vtype=GRB.BINARY, name=f"prod_{courier}_{i}_{j}")
    model.update()

    '''
    prod[courier, i, j] rappresenta il prodotto delle variabili decisionali cour[courier][i] e pred[i][j] => linearization
    '''

    # nuovi vincoli
    for courier in range(m):
        for i in range(n):
            for j in range(n):
                model.addConstr(prod[courier, i, j] <= cour[courier][i], f"prod_courier_{courier}_i_{i}_j_{j}_1")
                model.addConstr(prod[courier, i, j] <= pred[i][j], f"prod_courier_{courier}_i_{i}_j_{j}_2")
                model.addConstr(prod[courier, i, j] >= cour[courier][i] + pred[i][j] - 1, f"prod_courier_{courier}_i_{i}_j_{j}_3")
                model.addConstr(prod[courier, i, j] >= 0, f"prod_courier_{courier}_i_{i}_j_{j}_4")

    '''
    prod[courier, i, j] <= cour[courier][i]: Questo vincolo assicura che prod[courier, i, j] non possa essere 1 a meno che cour[courier][i] non sia 1. Se cour[courier][i] è 0, allora prod[courier, i, j] deve essere 0.

    prod[courier, i, j] <= pred[i][j]: Analogamente, questo vincolo assicura che prod[courier, i, j] non possa essere 1 a meno che pred[i][j] non sia 1. Se pred[i][j] è 0, allora prod[courier, i, j] deve essere 0.

    prod[courier, i, j] >= cour[courier][i] + pred[i][j] - 1: Questo vincolo assicura che prod[courier, i, j] debba essere 1 se sia cour[courier][i] che pred[i][j] sono 1. Se entrambe queste variabili sono 1, allora la somma cour[courier][i] + pred[i][j] sarà 2, e quindi prod[courier, i, j] deve essere almeno 1.

    prod[courier, i, j] >= 0: Questo vincolo assicura semplicemente che prod[courier, i, j] non possa essere negativo.

    Insieme, questi vincoli assicurano che prod[courier, i, j] sia 1 se e solo se sia cour[courier][i] che pred[i][j] sono 1, che è esattamente il comportamento che ci aspettiamo dal prodotto di queste due variabili.   
    '''
    
    # Modifica il vincolo originale
    for courier in range(m):
        model.addConstr(dist_vector[courier] - (quicksum((pred[item][n+courier]) * D[n][item] for item in range(n))+
                                                quicksum((pred[n+courier][item]) * D[item][n] for item in range(n))+
                                                quicksum((prod[courier, i, j]) * D[j][i] for i in range(n) for j in range(n))) == 0, 
                                                f"dist_vector_{courier}")
    

    #Constraint for the max distance: takes the maximum value of dist_vector
    max_dist = model.addVar(vtype=GRB.INTEGER, name="max_dist")
    model.update()
    
    model.addGenConstrMax(max_dist, dist_vector, name="max_dist_constraint")

    #Objective function: minimize the maximum distance
    model.setObjective(max_dist, GRB.MINIMIZE)

    return model, pred, cour



def MIP_MCP(n, m, s, l, D, approaches: list = 'default', focus=0, total_time=300):
    
    solutions = {}
    
    '''
    Optimizing the problem with the default approach: no clustering, using Gurobi model optimization and MIPFocus parameter 
    The MIPFocus parameter in Gurobi is used to guide the MIP solver in different directions depending on the value set:
    
    MIPFocus=0 (default): The solver balances between finding feasible solutions and proving optimality. 
    It's the default behavior of the solver.

    MIPFocus=1: The solver focuses more on finding feasible solutions. 
    This is useful in finding a feasible solution quickly rather than proving optimality.

    MIPFocus=2: The solver focuses more on proving optimality. 
    This is useful when proving that the solution found is the best possible one.

    MIPFocus=3: The solver focuses on improving the bound. 
    This is useful when its difficult proving that the optimal solution has been found.

    ''' 
    if focus == 0:
        json_name = "Gurobi_Default"
    elif focus == 1:
        json_name = "Gurobi_Feasible"
    elif focus == 2:
        json_name = "Gurobi_Optimal"
    elif focus == 3:
        json_name = "Gurobi_Bound"


    gp.setParam("TimeLimit", total_time) #Time Limit of 5 minutes
        
    model, pred, cour = MIP_Model(n,m,s,l,D, focus) 

    solutions = {}

    #MIPFocus parameter
    model.setParam(GRB.Param.MIPFocus, focus)
    
    model.optimize()

    #Solutions Creation
    if model.SolCount == 0: #no solution found
        solutions[json_name] = {'time' : 300 , 'optimal' : False , 'obj' : 'N/A' , 'sol' : []}
    else: 
        item_pred, cour_item = [(i,j) for j in range(n+m) for i in range(n+m) if pred.X[i][j]], [(i, j) for j in range(n) for i in range(m) if cour.X[i][j]] 
        solutions[json_name] = {'time' : int(model.Runtime) , 'optimal' : model.status == GRB.OPTIMAL , 'obj' : int(model.objVal), 'sol' : sf.solution_maker(item_pred, cour_item, n, m)}

    return solutions
