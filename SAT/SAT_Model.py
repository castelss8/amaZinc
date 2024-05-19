from z3 import *

import numpy as np
from itertools import combinations
from utils import *
import math
import time
from objective_function import objective_function

# Def of constraints
def exactly_k(var: list[BoolRef], k: int):
    return PbEq([(v,1) for v in var], k)

def exactly_one(var: list[BoolRef]):
    return PbEq([(v,1) for v in var], 1)

def at_most_k(var: list[BoolRef], k: int):
    """
    Return constraint that at most k of the variables in vars are true
    :param variables: List of variables
    :param k: Maximum number of variables that can be true
    :return:
    """

    return PbLe([(v, 1) for v in var], k)


def SAT_MCP(n, m, s, l, D):
    solv= Solver()
    max_w = np.max(l)

    #Create a m x (n+m) matrix cour: cour[i][j] ==True iff the i-th courier takes the j-th item (or starting/ending point).
    cour=[[Bool(f"c({i})_{j})") for j in range(n+m)]for i in range(m)]

    #Each pack have to be carried by exactly one courrier
    for j in range(n):
        solv.add(exactly_one([cour[i][j] for i in range(m)]))

    #The last m coulmns of c are fixed: the courier cour starts and ends at the deposit n+cour
    for j in range(m):
        for i in range(m):
            if j==i:
                solv.add(cour[i][n+j])
            else:
                solv.add(Not(cour[i][n+j]))
    
    #Weight constraint
    for courier in range(m):
        cour_weight = [cour[courier][item] for item in range(n) for _ in range(s[item])]
        solv.add(at_most_k(cour_weight, l[courier]))
   
    #Create a (n+m)x(n+m) matrix: pred[i][j] = true if the j-th item is the predecessor of the i-th item. 
    #The n+c column in the matrix is the starting point of the c-th courier. The n+c row in the matrix is the ending point of the c-th courier
    pred = [[Bool(f"pred({i})_{j}")for j in range(n+m)]for i in range(n+m)]

    #Each item/ending point has exactly one predecessor and each item/starting point is predecessor of exactly one other item. 
    for i in range(n+m):
        col_i = []
        for j in range(n+m):
            col_i += [pred[j][i]]

        solv.add(exactly_one(col_i))
        solv.add(exactly_one(pred[i]))

    #If the courier cour has the item i and the item j is the predecessor of the item i then cour has the item j
    for courier in range(m):
        solv.add(And(  [Implies(And([ cour[courier][i], pred[i][j]] ) , cour[courier][j]) for i in range(n+m) for j in range(n+m)]  ))
   
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


    # Time
    starting_time = time.time()
    timeout = starting_time + 60*5 #Ending time
    check_timeout = timeout-time.time() #Time left
    best_obj = 10000

    #print('Start...')

    while time.time() < timeout:
        solv.set('timeout', int(check_timeout*1000)) #time left in millisec 
        solv.push()

        for courier in range(m):
            tmp_dist = [And(cour[courier][item], pred[item][item_2]) for item in range(n) for item_2 in range(n) for _ in range(D[item_2][item])]
            tmp_dist += [And(cour[courier][item], pred[item][n+courier]) for item in range(n) for _ in range(D[n][item])]
            tmp_dist += [And(cour[courier][item], pred[n+courier][item]) for item in range(n) for _ in range(D[item][n])]
            solv.add(at_most_k(tmp_dist, best_obj-1))
        
        if solv.check()==sat:
            tmp_model = solv.model()
            item_pred, cour_item = [(i,j) for j in range(n+m) for i in range(n+m) if tmp_model.evaluate(pred[i][j])], [(i, j) for j in range(n) for i in range(m) if tmp_model.evaluate(cour[i][j])] #tmp_model.evaluate(c[i][j][0])
            tmp_obj = objective_function(item_pred, cour_item, n, m, D)
            if tmp_obj<best_obj:
                best_solution=tmp_model
                best_obj=tmp_obj
                #print(best_obj)
            check_timeout = timeout-time.time() #Time left
            solv.pop()
        else:
            return best_solution, best_obj



#TEMPORANEO:
instance_n=2 #from 1 to 21
#"C:\Users\mange\Documents\GitHub\Uni\amaZinc\SAT\Instances\inst01.dat"
if instance_n<10:
    file_name='inst0'+str(instance_n)+'.dat'
else:
    file_name='inst'+str(instance_n)+'.dat'
file = open('/Users/mange/Documents/GitHub/Uni/amaZinc/SAT/Instances/inst02.dat', 'r')
#file = open('./Instances/'+file_name, 'r')

splitted_file = file.read().split('\n')

m = int(splitted_file[0])
n = int(splitted_file[1])
cpt_tmp = list(map(int, splitted_file[2].split(' ')))
tmp_sz=splitted_file[3].split(' ')
if '' in tmp_sz:
    sz_tmp=list(map(int, [tmp_sz[i] for i in range(len(tmp_sz)) if tmp_sz[i]!='']))
else:
    sz_tmp = list(map(int, splitted_file[3].split(' ')))
D = [list(map(int, line.strip().split(' '))) for line in splitted_file[4:(n+5)]]

print('Instance number '+str(instance_n)+': '+str(n)+' items and '+str(m)+' couriers.')
SAT_MCP(n, m, sz_tmp, cpt_tmp, D)

