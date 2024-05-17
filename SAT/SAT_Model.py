from z3 import *

import numpy as np
from itertools import combinations
from utils import *
import math
import time


# Def of constraints

def at_least_one_seq(bool_vars):
    return Or(bool_vars)

def at_most_one_seq(bool_vars, name):
    constraints = []
    n = len(bool_vars)
    s = [Bool(f"s_{name}_{i}") for i in range(n - 1)]
    constraints.append(Or(Not(bool_vars[0]), s[0]))
    constraints.append(Or(Not(bool_vars[n-1]), Not(s[n-2])))
    for i in range(1, n - 1):
        constraints.append(Or(Not(bool_vars[i]), s[i]))
        constraints.append(Or(Not(bool_vars[i]), Not(s[i-1])))
        constraints.append(Or(Not(s[i-1]), s[i]))
    return And(constraints)

def exactly_one_seq(bool_vars, name):
    return And(at_least_one_seq(bool_vars), at_most_one_seq(bool_vars, name))

def at_most_k_seq(bool_vars, k, name):
    constraints = []
    n = len(bool_vars)
    s = [[Bool(f"s_{name}_{i}_{j}") for j in range(k)] for i in range(n - 1)]
    constraints.append(Or(Not(bool_vars[0]), s[0][0]))
    constraints += [Not(s[0][j]) for j in range(1, k)]
    for i in range(1, n-1):
        constraints.append(Or(Not(bool_vars[i]), s[i][0]))
        constraints.append(Or(Not(s[i-1][0]), s[i][0]))
        constraints.append(Or(Not(bool_vars[i]), Not(s[i-1][k-1])))
        for j in range(1, k):
            constraints.append(Or(Not(bool_vars[i]), Not(s[i-1][j-1]), s[i][j]))
            constraints.append(Or(Not(s[i-1][j]), s[i][j]))
    constraints.append(Or(Not(bool_vars[n-1]), Not(s[n-2][k-1])))   
    return And(constraints)

def at_least_k_seq(bool_vars, k, name):
    return at_most_k_seq([Not(var) for var in bool_vars], len(bool_vars)-k, name)

def exactly_k_seq(bool_vars, k, name):
    return And(at_most_k_seq(bool_vars, k, name+'1'), at_least_k_seq(bool_vars, k, name))



def SAT_MCP(n, m, sz, cpt):
    solv= Solver()
    max_w = np.max(cpt)

    #Create a m x (n+m) matrix c: c[i][j] ==True iff the i-th courier takes the j-th item (or starting/ending point).
    c=[[Bool(f"c({i})_{j})") for j in range(n+m)]for i in range(m)]

    #Each pack have to be carried by exactly one courrier
    for j in range(n):
        solv.add(exactly_one_seq([c[i][j] for i in range(m)], f"one_c_for_item({j})"))

    #The last m coulmns of c are fixed: the courier cour starts and ends at the deposit n+cour
    for j in range(m):
        for i in range(m):
            if j==i:
                solv.add(c[i][n+j])
            else:
                solv.add(Not(c[i][n+j]))
    
    #Weight constraint
    for cour in range(m):
        cour_weight = [c[cour][item] for item in range(n) for _ in range(sz[item])]
        solv.add(at_most_k_seq(cour_weight, cpt[cour]))
   
    #Create a (n+m)x(n+m) matrix: pred[i][j] = true if the j-th item is the predecessor of the i-th item. 
    #The n+c column in the matrix is the starting point of the c-th courier. The n+c row in the matrix is the ending point of the c-th courier
    pred = [[Bool(f"pred({i})_{j}")for j in range(n+m)]for i in range(n+m)]

    #Each item/ending point has exactly one predecessor and each item/starting point is predecessor of exactly one other item. 
    for i in range(n+m):
        col_i = []
        for j in range(n+m):
            col_i += [pred[j][i]]

        solv.add(exactly_one_seq(col_i, f"PC_{i}"))
        solv.add(exactly_one_seq(pred[i], f"PR_{i}"))

    #If the courier cour has the item i and the item j is the predecessor of the item i then cour has the item j
    for cour in range(m):
        solv.add(And(  [Implies(And([ c[cour][i], pred[i][j]] ) , c[cour][j]) for i in range(n+m) for j in range(n+m)]  ))
   
    #Create a list of length n no_loop: if the item j is predecessor of the item i then no_loop(i) as an integer is > than no_loop(j) as an integer 
    nl_max = n//(m-1)+1
    no_loops = [[Bool(f"no_loops({item})_{j}")for j in range(nl_max)]for item in range(n)]

    for item in range(n):
        for k in range(n//(m-1)+1):
            solv.add(Implies(Not(no_loops[item][k]), Not(Or(no_loops[item][:k])))) #k=False then all the above are false --> if j<k is true then all the below are true
    #Example: no_loops[j] = [1 1 1 1 1 1 1 0 0 0]

    # If item_j is predecessor of item_j then no_loops[j]<no_loops[i] as integers    
    for item_i in range(n):
        for item_j in range(n):
            solv.add(Implies(pred[item_i][item_j], Or([  And(no_loops[item_i][k], Not(no_loops[item_j][k])) for k in range(nl_max)  ])  ))
   