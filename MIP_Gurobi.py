import gurobipy as gp
from gurobipy import GRB, quicksum
import numpy as np

import numpy as np

#Read instance from file 
f_name = "Instances/inst01"
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
    model = gp.Model("MIP_Model")

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
                model.addConstr(v[i][n+j] == 1)
            else:
                model.addConstr(v[i][n+j] == 0)
    
    #Constraint fot the capacity of each courrier (peso)
    for c in range(m):
        print("constraint peso ", quicksum(v[c][i]*s[i] for i in range(n)) <= l[c])
        model.addConstr(quicksum(v[c][i]*s[i] for i in range(n)) <= l[c])

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
        model.addConstr(sum(col_i) == 1, f"PC_{i}")
        model.addConstr(sum(pred[i]) == 1, f"PR_{i}")

    #Constraint coerenza courier - predecessore/successore
    for c in range(m):
        for i in range(n+m):
            for j in range(n+m):
                model.addConstr(v[c][j] >= v[c][i] + pred[i][j] - 1)
                #se v[c][i] e pred[i][j] sono entrambi 1, allora v[c][j] deve essere almeno 2, ma poiché è una variabile binaria, deve essere 1. Se uno o entrambi sono 0, allora v[c][j] può essere 0 o 1.
   
    #vector to avoid loops, lenght n
    no_loops = [model.addVar(lb=1, ub=n, vtype=GRB.INTEGER, name=f"no_loops") for _ in range(n)]
    for i in range(n):
        for j in range(n):
            if pred[i][j]: #if pred[i][j] is true 
                model.addConstr(no_loops[i] > no_loops[j])
    
    #vector of distances
    dist_vector = []
    for courier in range(m):
        dist_vector.append(model.addVar(vtype=GRB.INTEGER, name="dist_vector"))

    # Creare le espressioni per i punti di partenza, di arrivo e intermedi
    for courier in range(m):
        starting_point = [quicksum((pred[item][n+courier] == 1) * D[n][item] for item in range(n))]
        ending_point = [quicksum((pred[n+courier][item] == 1) * D[item][n] for item in range(n))]
        mid_points = [quicksum((v[courier][i] == 1 and pred[i][j] == 1) * D[j][i] for i in range(n) for j in range(n))]
        distance_of_this_path = quicksum(starting_point + mid_points + ending_point)
        model.addConstr(dist_vector[courier] == distance_of_this_path)

    # Creare la variabile max_dist e impostarla come massimo di dist_vector
    max_dist = model.addVar(vtype=GRB.INTEGER, name="max_dist")
    model.addGenConstrMax(max_dist, dist_vector, name="max_dist")

    # Impostare l'obiettivo del modello per minimizzare max_dist
    model.setObjective(max_dist, GRB.MINIMIZE)

     
    #Solver
    model.optimize()

    #model.computeIIS()
    #model.write("model.ilp")
    
    '''
    # The deposit has at most n predecessor and it is predecessor of at most n items
    col_n = []
    for j in range(n+m):
        col_n += [pred[j][n]]
    PR_n = model.addVar(vtype=GRB.BINARY, name=f"PR_{n}") #pred row (predecessors of the deposit -- last item of each courier)
    PC_n = model.addVar(vtype=GRB.BINARY, name=f"PC_{n}") #pred col (items that have deposit as predecessor -- first items of each courier)
    model.addConstr(sum(PR_n for i in pred[n+m]) <= m, f"PR_{n}") #constraint for the deposit that has at most n predecessor
    model.addConstr(sum(PC_n for i in col_n) <= m, f"PC_{n}") #constraint for the deposit that is predecessor of at most n items
    #v[i[m+i]] = true e da lì in poi falso 
    
    # Each item has exactly one predecessor and each item is predecessor of exactly one other item
    for i in range(n):
        col_i = []
        for j in range(n+1):
            col_i += [pred[j][i]]
        PC_i = model.addVar(vtype=GRB.BINARY, name=f"PC_{i}") #var for each item that has exactly one predecessor
        PR_i = model.addVar(vtype=GRB.BINARY, name=f"PR_{i}") #var for each item that is predecessor of exactly one other item
        model.addConstr(sum(PC_i for i in col_i) == 1, f"PC_{i}") #constraint for each item that has exactly one predecessor
        model.addConstr(sum(PR_i for i in pred[i]) == 1, f"PR_{i}") #constraint for each item that is predecessor of exactly one other item

    
    
    
    #Se due items hanno come predecessore il deposito allora devono essere presi da due veicoli diversi
    for i in range(n):
        for j in range(i+1, n):
            for k in range(m):
                #genconstrindicator constraint che si applica solo quando var=True
                model.addGenConstrIndicator(pred[i][n], True, v[k][i] + v[k][j] <= 1) #constraint for each pack that is carried by exactly one courrier
                #Per ogni combinazione di i, j (con j sempre maggiore di i) e k: se pred[i][n] è True, allora la somma di v[k][i] e v[k][j] deve essere minore o uguale a 1
                #Cioè se un certo elemento i è un predecessore di n, allora non può essere che sia v[k][i] che v[k][j] siano True (o, in termini di programmazione lineare intera, uguale a 1) allo stesso tempo

    #loop avoidance
    for i in range(n):
        for j in range(n):
            model.addGenConstrIndicator(pred[i][j], True, no_loops[i] - no_loops[j] >= 1) #constraint for each pack that is carried by exactly one courrier
            #se pred[i][j] è True, allora il valore di no_loops[i] deve essere almeno uno più grande del valore di no_loops[j].
    '''

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
