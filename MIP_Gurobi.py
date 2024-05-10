import gurobipy as gp
from gurobipy import GRB, quicksum
import numpy as np

import numpy as np

#Read instance from file 
f_name = "Instances/inst11"
with open(f_name + ".dat", "r") as f:
    lines = f.readlines()

#variables
m = int(lines[0]) #number of couriers
n = int(lines[1]) #number of items
l = [int(x) for x in lines[2].split()] #%array of couriers' capacities
s = [int(x) for x in lines[3].split()] #%array of items' sizes
D = [[int(x) for x in line.split()] for line in lines[4:]] #%matrix of distances
#lower_bound = prim(D) #lower bound

# Time Limit of 5 minutes
#gp.setParam("TimeLimit", 300) # Time Limit of 5 minutes

try:
    #Model Initialization
    model = gp.Model("MIP_Model_Gurobi")

    #Constraints
    #model.addConstr(obj >= lower_bound)

    #Decision variables
    '''
      vehicle matrix 
      col = packs
      rows = courriers 
      each rows represent the list of packs foreach courrier
      #create a [m x (n+m)] matrix v. v[i][j]==True if the i-th courier takes the j-th item (or starting/ending point)
    ''' 
    v = model.addMVar(shape=(m, n+m), vtype=GRB.BINARY, name="v")
    model.update()
    #print("matrice dei veicoli ", v)

    #constraints for v 
    # each pack have to be carried by exactly one courrier
    for j in range(n):
        model.addConstr(quicksum(v[i][j] for i in range(m)) == 1, f"v_{j}")
    
    for j in range(m):
        for i in range(m):
            if j == i:
                model.addConstr(v[i][n+j] == 1, f"v_{n+j}")
            else:
                model.addConstr(v[i][n+j] == 0, f"v_{n+j}")
    
    #print("l: ", l)
    #Constraint fot the capacity of each courrier (peso)
    for c in range(m):
        #print("constraint peso ", quicksum(v[c][i]*s[i] for i in range(n)) <= l[c])
        model.addConstr(quicksum(v[c][i]*s[i] for i in range(n)) <= l[c], f"peso_{c}")

    '''
      pred matrix
      pred[i][j] = true if the j-th item is the predecessor of the i-th item  
      #Gurobi matrix of predecessors, boolean
   '''
    pred = model.addMVar(shape=(n+m, n+m), vtype=GRB.BINARY, name="pred")
    #print("matrice dei pred ", pred)
    model.update()

    #Each item/ending point has exactly one predecessor and each item/starting point is predecessor of exactly one other item. 
    #i=0 to i=n+m-1
    #j=0 to j=n+m-1 
    for i in range(n+m):
        col_i = [pred[j][i] for j in range(n+m)]
        model.addConstr(quicksum(col_i) == 1, f"PC_{i}")
        model.addConstr(quicksum(pred[i]) == 1, f"PR_{i}")

    #Constraint coerenza courier - predecessore/successore
    for c in range(m):
        for i in range(n+m):
            for j in range(n+m):
                model.addConstr(v[c][j] >= v[c][i] + pred[i][j] - 1, f"coerenza_{c}_{i}_{j}")
                #se v[c][i] v pred[i][j] => v[c][j]
                #se v[c][i] e pred[i][j] sono entrambi 1, allora v[c][j] deve essere almeno 2, ma poiché è una variabile binaria, deve essere 1. Se uno o entrambi sono 0, allora v[c][j] può essere 0 o 1.
    
    #vector to avoid loops, lenght n
    no_loops = [model.addVar(lb=1, ub=n, vtype=GRB.INTEGER, name=f"no_loops") for _ in range(n)]
    model.update()
    for i in range(n):
        for j in range(n):
            #if pred[i][j] == 1 then no_loops[i] - no_loops[j] >= 1
            #if pred[i][j] == 0 then 0 - 0 - 0 >= 0
            model.addConstr(no_loops[i]*pred[i][j] - no_loops[j]*pred[i][j] - pred[i][j] >= 0, f"no_loops_{i}_{j}")
    
    #vector of distances for each courier
    dist_vector = []
    for courier in range(m):
        dist_vector.append(model.addVar(vtype=GRB.INTEGER, name="dist_vector"))
    
    model.update()

    # Creare le espressioni per i punti di partenza, di arrivo e intermedi
    for courier in range(m):
        #starting_point = quicksum((pred[item][n+courier]) * D[n][item] for item in range(n))
        #ending_point = quicksum((pred[n+courier][item]) * D[item][n] for item in range(n))
        #mid_points = quicksum((v[courier][i] * pred[i][j]) * D[j][i] for i in range(n) for j in range(n))
        #distance_of_this_path = quicksum(starting_point + mid_points + ending_point)
        model.addConstr(dist_vector[courier] - (quicksum((pred[item][n+courier]) * D[n][item] for item in range(n))+
                                                quicksum((pred[n+courier][item]) * D[item][n] for item in range(n))+
                                                quicksum((v[courier][i] * pred[i][j]) * D[j][i] for i in range(n) for j in range(n))) == 0, 
                                                f"dist_vector_{courier}")

    # Creare la variabile max_dist e impostarla come massimo di dist_vector
    max_dist = model.addVar(vtype=GRB.INTEGER, name="max_dist")
    model.update()
    # Creare il vincolo per max_dist, prende il massimo valore di dist_vector
    model.addGenConstrMax(max_dist, dist_vector, name="max_dist_constraint")


    # Impostare l'obiettivo del modello per minimizzare max_dist
    model.setObjective(max_dist, GRB.MINIMIZE)

    #Solver
    model.optimize()

    #model.computeIIS()
    #model.write("model.ilp")
    
except gp.GurobiError as e:

    print('Error code ' + str(e.errno) + ": " + str(e))

'''
def prim(G):
    INF = 9999999
    N = len(G[0])
    sum = 0

    selected_node = np.zeros(N)
    no_edge = 0
    selected_node[0] = True

    while(no_edge < N-1):
        minimum = INF
        a = 0
        b = 0

        for i in range(N):
            if selected_node[i]:
                for j in range(N):
                    if ((not selected_node[j]) and G[i][j]):
                        if minimum > G[i][j]:
                            minimum = G[i][j]
                            a = i
                            b = j
        sum+=minimum
       # print(str(a) + "-" + str(b) + " : " + str(G[a][b]))
        selected_node[b] = True
        no_edge += 1
    #print(sum)
    return sum'''
