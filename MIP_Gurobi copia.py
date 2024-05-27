import gurobipy as gp
from gurobipy import GRB
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

# Time Limit of 5 minutes
gp.setParam("TimeLimit", 300) # Time Limit of 5 minutes

try:
    # Model Initialization
    model = gp.Model("LP_Model")

    #Decision variables
 
    #Gurobi matrix of predecessors boolean 
    pred = model.addMVar(shape=(n+1, n+1), vtype=GRB.BINARY, name="pred")

    '''
      vehicle matrix 
      col = packs
      rows = courriers 
      each rows represent the list of packs foreach courrier
      #create a [m x (n+m)] matrix v. v[i][j]==True if the i-th courier takes the j-th item (or starting/ending point)
    ''' 
    v = model.addMVar(shape=(m, n), vtype=GRB.BINARY, name="v")

    #vector to avoid loops, lenght n
    no_loops = model.addMVar(shape=(n), lb=1, ub=n, vtype=GRB.INTEGER, name="no_loops")

     
except gp.GurobiError as e:

    print('Error code ' + str(e.errno) + ": " + str(e))