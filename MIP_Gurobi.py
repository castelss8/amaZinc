import gurobipy as gp
from gurobipy import GRB, quicksum
import numpy as np
from functions import sol_functions as sf
from functions import clustering as cl
import copy

import numpy as np

#Read instance from file 
f_name = "Instances/inst10"
with open(f_name + ".dat", "r") as f:
    lines = f.readlines()

#variables
m = int(lines[0]) #number of couriers
n = int(lines[1]) #number of items
l = [int(x) for x in lines[2].split()] #%array of couriers' capacities
s = [int(x) for x in lines[3].split()] #%array of items' sizes
D = [[int(x) for x in line.split()] for line in lines[4:]] #%matrix of distances
#lower_bound = prim(D) #lower bound


"""
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


    #val_pred = pred.X
    #print(" val pred: ", val_pred)
    

    #model.computeIIS()
    #model.write("model.ilp")
    
except gp.GurobiError as e:

    print('Error code ' + str(e.errno) + ": " + str(e))
"""

def MIP_Model(n, m, s, l, D):
    
    #Model Initialization
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

    #constraints for cour
    # each pack have to be carried by exactly one courrier
    for j in range(n):
        model.addConstr(quicksum(cour[i][j] for i in range(m)) == 1, f"cour_{j}")
    
    #the last m-columns of cour are fixed: the courier c starts and ends at the deposit n+c
    for j in range(m):
        for i in range(m):
            if j == i:
                model.addConstr(cour[i][n+j] == 1, f"cour_{n+j}")
            else:
                model.addConstr(cour[i][n+j] == 0, f"cour_{n+j}")
    
    #Constraint fot the capacity of each courrier (peso)
    for c in range(m):
        model.addConstr(quicksum(cour[c][i]*s[i] for i in range(n)) <= l[c], f"peso_{c}")

    '''
      pred matrix
      pred[i][j] = true if the j-th item is the predecessor of the i-th item  
      #Gurobi matrix of predecessors, boolean
   '''
    pred = model.addMVar(shape=(n+m, n+m), vtype=GRB.BINARY, name="pred")
    model.update()

    #Each item/ending point has exactly one predecessor and each item/starting point is predecessor of exactly one other item. 
    for i in range(n+m):
        col_i = [pred[j][i] for j in range(n+m)]
        model.addConstr(quicksum(col_i) == 1, f"PC_{i}")
        model.addConstr(quicksum(pred[i]) == 1, f"PR_{i}")

    #Constraint coerenza courier - predecessore/successore
    #if the courier c has the item i and item j is the predecessor of item i then c has also item j
    for c in range(m):
        for i in range(n+m):
            for j in range(n+m):
                model.addConstr(cour[c][j] >= cour[c][i] + pred[i][j] - 1, f"coerenza_{c}_{i}_{j}")
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
                                                quicksum((cour[courier][i] * pred[i][j]) * D[j][i] for i in range(n) for j in range(n))) == 0, 
                                                f"dist_vector_{courier}")

    # Creare la variabile max_dist e impostarla come massimo di dist_vector
    max_dist = model.addVar(vtype=GRB.INTEGER, name="max_dist")
    model.update()
    # Creare il vincolo per max_dist, prende il massimo valore di dist_vector
    model.addGenConstrMax(max_dist, dist_vector, name="max_dist_constraint")

    # Impostare l'obiettivo del modello per minimizzare max_dist
    model.setObjective(max_dist, GRB.MINIMIZE)
    
    return model, pred, cour

def MIP_Model_Clustering(n, D):
    
    #Model Initialization
    model = gp.Model("MIP_Model_Gurobi_Clustering")
    
    '''
      pred matrix
      pred[i][j] = true if the j-th item is the predecessor of the i-th item  
      #Gurobi matrix of predecessors, boolean
   '''
    pred = model.addMVar(shape=(n+1, n+1), vtype=GRB.BINARY, name="pred")
    model.update()
    
    #Each item/ending point has exactly one predecessor and each item/starting point is predecessor of exactly one other item. 
    for i in range(n+1):
        col_i = [pred[j][i] for j in range(n+1)]
        model.addConstr(quicksum(col_i) == 1, f"PC_{i}")
        model.addConstr(quicksum(pred[i]) == 1, f"PR_{i}")

    #vector to avoid loops, lenght n
    no_loops = [model.addVar(lb=1, ub=n, vtype=GRB.INTEGER, name=f"no_loops") for _ in range(n)]
    model.update()
    for i in range(n):
        for j in range(n):
            #if pred[i][j] == 1 then no_loops[i] - no_loops[j] >= 1
            #if pred[i][j] == 0 then 0 - 0 - 0 >= 0
            model.addConstr(no_loops[i]*pred[i][j] - no_loops[j]*pred[i][j] - pred[i][j] >= 0, f"no_loops_{i}_{j}")

    #vector of distances for each courier
    dist = model.addVar(vtype=GRB.INTEGER, name="dist")
    model.update()

    model.addConstr(dist - quicksum((pred[i][j] * D[j][i] for i in range(n) for j in range(n))) == 0)

    # Impostare l'obiettivo del modello per minimizzare max_dist
    model.setObjective(dist, GRB.MINIMIZE)
    
    return model, pred



def MIP_MCP(n, m, s, l, D, approaches: list, total_time=300):
    
    solutions = {}
    
    if 'default' in approaches:
        # Time Limit of 5 minutes
        gp.setParam("TimeLimit", total_time) # Time Limit of 5 minutes

        model, pred, cour = MIP_Model(n,m,s,l,D)
        model.optimize()

        if model.SolCount == 0:
            solutions['default'] = {'time' : 300 , 'optimal' : False , 'obj' : 'N/A' , 'sol' : []}
        else: 
            item_pred, cour_item = [(i,j) for j in range(n+m) for i in range(n+m) if pred.X[i][j]], [(i, j) for j in range(n) for i in range(m) if cour.X[i][j]] 
            solutions['default'] = {'time' : int(model.Runtime) , 'optimal' : model.status == GRB.OPTIMAL , 'obj' : int(model.objVal), 'sol' : sf.solution_maker(item_pred, cour_item, n, m)}

    if 'clustering' in approaches:

        clusters, s_clusters = cl.complete_clustering(D, s, n, m)

        real_clusters = [cluster for cluster in clusters if len(cluster)>1]
        clusters_paths=[]

        time_used = 0

        for it in range(len(real_clusters)):
            cluster = real_clusters[it]
            print('working on' , cluster)
            n_cluster = len(cluster)

            gp.setParam("TimeLimit", 60*1*((it+1)/len(real_clusters))) #1 minute for all clusters
            
            D_clus=[]
            for i in cluster:
                D_clus.append([])
                for j in cluster:
                    D_clus[-1].append(D[i][j])
                D_clus[-1].append(0)
            D_clus.append([0 for k in range(n_cluster+1)])

            model, pred = MIP_Model_Clustering(n_cluster, D_clus)
            model.optimize()

            time_used += model.Runtime

            if model.SolCount == 0:
                cluster_copy=copy.deepcopy(cluster)
                cluster_copy.append(-1)
                cluster_copy[:0] = [-1]
                clusters_paths.append([(cluster_copy[i],cluster_copy[i+1]) for i in range(n_cluster+1)])
            else: 
                cluster_copy=copy.deepcopy(cluster)
                cluster_copy.append(-1)
                clusters_paths.append([(cluster_copy[i],cluster_copy[j]) for i in range(n_cluster+1) for j in range(n_cluster+1) if pred.X[i][j]])

        n_new = len(clusters)-1

        first_items_for_clusters=[]
        last_item_for_clusters=[]

        i=0
        for clus in clusters:
            if len(clus)>1:
                path=clusters_paths[i]
                for couple in path:
                    if couple[0]==-1:
                        last_item_for_clusters.append(couple[1])
                    elif couple[1]==-1:
                        first_items_for_clusters.append(couple[0])
                i+=1
            else:
                last_item_for_clusters.append(clus[0])
                first_items_for_clusters.append(clus[0])

        D_new=[]
        for i in range(n_new+1):
            D_new.append([])
            for j in range(n_new+1):
                if j==i:
                    D_new[-1].append(0)
                else:
                    D_new[-1].append( D[first_items_for_clusters[i]][last_item_for_clusters[j]])
        
        big_sol = MIP_MCP(n_new,m,s,l,D_new, ['default'], total_time=300-time_used)
        
        if big_sol['default']['sol'] != []:
            sol = sf.solution_maker_cluster(clusters, clusters_paths, first_items_for_clusters, big_sol['default']['sol'], m)
            solutions['clustering'] = {'time' : big_sol['default']['time']+time_used, 'optimal' : False, 'obj' : sf.obj_fun_from_solution(sol, n, D), 'sol' : sol} 
        else:
            solutions['clustering'] = {'time' : 300 , 'optimal' : False , 'obj' : 'N/A' , 'sol' : []}

    return solutions

print(MIP_MCP(n,m,s,l,D, ['default']))