import gurobipy as gp
from gurobipy import GRB
import numpy as np

import numpy as np

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
        print(str(a) + "-" + str(b) + " : " + str(G[a][b]))
        selected_node[b] = True
        no_edge += 1
    print(sum)
    return sum

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
lower_bound = prim(D) #lower bound

# Time Limit of 5 minutes
gp.setParam("TimeLimit", 300) # Time Limit of 5 minutes

try:
    #Model Initialization
    model = gp.Model("LP_Model")

    #Decision variables
    '''
      pred matrix
      pred[i][j] = true if the j-th item is the predecessor of the i-th item  
      #Gurobi matrix of predecessors, boolean
   '''
    #pred = model.addMVar(shape=(n+1, n+1), vtype=GRB.BINARY, name="pred")
    pred = [[model.addVar(vtype=GRB.BINARY) for _ in range(n+m)] for _ in range(n+m)]
    
    '''
      vehicle matrix 
      col = packs
      rows = courriers 
      each rows represent the list of packs foreach courrier
      #create a [m x (n+m)] matrix v. v[i][j]==True if the i-th courier takes the j-th item (or starting/ending point)
    ''' 
    #v = model.addMVar(shape=(m, n+m), vtype=GRB.BINARY, name="v")
    v = [[model.addVar(vtype=GRB.BINARY) for _ in range(n+m)] for _ in range(m)]

    #vector to avoid loops, lenght n
    #no_loops = model.addMVar(shape=(n), lb=1, ub=n, vtype=GRB.INTEGER, name="no_loops")
    no_loops = [model.addVar(vtype=GRB.INTEGER, name=f"no_loops_{i}") for i in range(n)]
    
    #Obj definition
    #dobbiamo otimizzare il corriere che fa il percorso più lungo
    #obj = 
   # model.setObjective(obj, GRB.MINIMIZE)

    #Constraints
    #model.addConstr(obj >= lower_bound)

    # The deposit has at most n predecessor and it is predecessor of at most n items
    col_n = []
    for j in range(n):
        col_n += [pred[j][n]]
    PR_n = model.addVar(vtype=GRB.BINARY, name=f"PR_{n}") #pred row (predecessors of the deposit -- last item of each courier)
    PC_n = model.addVar(vtype=GRB.BINARY, name=f"PC_{n}") #pred col (items that have deposit as predecessor -- first items of each courier)
    model.addConstr(sum(PR_n for i in pred[n]) <= m, f"PR_{n}") #constraint for the deposit that has at most n predecessor
    model.addConstr(sum(PC_n for i in col_n) <= m, f"PC_{n}") #constraint for the deposit that is predecessor of at most n items

    # Each item has exactly one predecessor and each item is predecessor of exactly one other item
    for i in range(n):
        col_i = []
        for j in range(n+1):
            col_i += [pred[j][i]]
        PC_i = model.addVar(vtype=GRB.BINARY, name=f"PC_{i}") #var for each item that has exactly one predecessor
        PR_i = model.addVar(vtype=GRB.BINARY, name=f"PR_{i}") #var for each item that is predecessor of exactly one other item
        model.addConstr(sum(PC_i for i in col_i) == 1, f"PC_{i}") #constraint for each item that has exactly one predecessor
        model.addConstr(sum(PR_i for i in pred[i]) == 1, f"PR_{i}") #constraint for each item that is predecessor of exactly one other item

    # each pack have to be carried by exactly one courrier
    for j in range(n):
        v_j = model.addVar(vtype=GRB.BINARY, name=f"v_{j}") #var for each pack that is carried by exactly one courrier
        model.addConstr(sum(v[i][j] for i in range(m)) == 1, f"v_{j}") #constraint for each pack that is carried by exactly one courrier
    
    
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
    
    #Solver
    model.optimize()

    model.computeIIS()
    model.write("model.ilp")

except gp.GurobiError as e:

    print('Error code ' + str(e.errno) + ": " + str(e))