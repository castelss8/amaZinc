from z3 import *

import numpy as np
from itertools import combinations
from utils import *
import math
import time
import sys
from importlib import reload

sys.path.append('/Users/mange/Documents/GitHub/Uni/amaZinc')
from functions import sol_functions as sf
from functions import clustering as cl 

# Def of constraints
def exactly_k(var: list[BoolRef], k: int):
    return PbEq([(v,1) for v in var], k)

def exactly_one(var: list[BoolRef]):
    return PbEq([(v,1) for v in var], 1)

def at_most_k(var: list[BoolRef], k: int):
    return PbLe([(v, 1) for v in var], k)

def big_SAT_Solver(n, m, s, l):

    solv= Solver()

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
    
    return solv, pred, cour


def small_SAT_Solver(n):

   solv= Solver()
   
   '''
      pred matrix  
   '''
   #pred[i][j] = true if the j-th item is the predecessor of the i-th item. 
   #The n+i column in the matrix is the starting point of the i-th courier. The n+i row in the matrix is the ending point of the i-th courier
   pred = [[Bool(f"pred({i})_{j}")for j in range(n+1)]for i in range(n+1)]

   #Each item/ending point has exactly one predecessor and each item/starting point is predecessor of exactly one other item. 
   #i=0 to i=n+m-1
   #j=0 to j=n+m-1 
   for i in range(n+1):
      col_i = []
      for j in range(n+1):
         col_i += [pred[j][i]]

      solv.add(exactly_one(col_i))
      solv.add(exactly_one(pred[i]))
   
   no_loops = [[Bool(f"no_loops({item})_{j}")for j in range(n)]for item in range(n)]

   for item in range(n):
      for k in range(n):
         solv.add(Implies(Not(no_loops[item][k]), Not(Or(no_loops[item][:k])))) #k=False then all the above are false --> if j<k is true then all the below are true
   
   for item_i in range(n):
      for item_j in range(n):
         solv.add(Implies(pred[item_i][item_j], Or([And(no_loops[item_i][k], Not(no_loops[item_j][k]))for k in range(n)])  ))
   
   return solv, pred


def SAT_MCP(n:int, m:int, s:list, l:list, D:list, approaches:list, tot_time = 300):
    """
    input
    - n : int = number of items
    - m : int = number of couriers
    - s : list of ints = items' weight
    - l : list of ints = couriers' capacity
    - D : list of lists = distance matrix
    - approaches : list of strings = approach to use ('default' or 'clustering')
    - tot_time : int = time's upper bound

    output
    - solutions : dict = it has approaches as keys and dictionaries containing the solution as items
    """

    solutions = {}

    if 'default' in approaches:

        solv, pred, cour = big_SAT_Solver(n, m, s, l)

        # Time
        starting_time = time.time()
        timeout = starting_time + tot_time #Ending time
        check_timeout = timeout-time.time() #Time left
        best_obj = sf.up_bound(n,D)

        opt = False
        while time.time() < timeout and not opt:
            solv.set('timeout', int(check_timeout*1000)) #time left in millisec 

            solv.push()
            for courier in range(m):
                tmp_dist = [And(cour[courier][item], pred[item][item_2]) for item in range(n) for item_2 in range(n) for _ in range(D[item_2][item])]
                tmp_dist += [And(cour[courier][item], pred[item][n+courier]) for item in range(n) for _ in range(D[n][item])]
                tmp_dist += [And(cour[courier][item], pred[n+courier][item]) for item in range(n) for _ in range(D[item][n])]
                solv.add(at_most_k(tmp_dist, best_obj-1))

            if solv.check()==sat:
                tmp_model = solv.model()
                item_pred, cour_item = [(i,j) for j in range(n+m) for i in range(n+m) if tmp_model.evaluate(pred[i][j])], [(i, j) for j in range(n) for i in range(m) if tmp_model.evaluate(cour[i][j])] 
                tmp_obj = sf.obj_fun(item_pred, cour_item, n, m, D)
                if tmp_obj<best_obj:
                    #best_solution=tmp_model
                    best_obj=tmp_obj
                check_timeout = timeout-time.time() #Time left
                solv.pop()
            else:
                solutions['default'] = {'time' : int(time.time() - starting_time) , 'optimal' : True , 'obj' : best_obj , 'sol' : sf.solution_maker(item_pred, cour_item, n, m)}
                opt = True

        if not opt:
            if best_obj == sf.up_bound(n,D):
                solutions['default'] = {'time' : 300 , 'optimal' : False , 'obj' : 'N/A' , 'sol' : []}
            else:
                solutions['default'] = {'time' : 300 , 'optimal' : False , 'obj' : best_obj , 'sol' : sf.solution_maker(item_pred, cour_item, n, m)}
        
    
    if 'clustering' in approaches:

        clusters, s_clusters = cl.complete_clustering(D, s, n, m)

        real_clusters = [cluster for cluster in clusters if len(cluster)>1]
        clusters_paths=[]

        starting_time = time.time()
        timeout = starting_time + 60*5

        for it in range(len(real_clusters)):
            cluster = real_clusters[it]
            print('working on' , cluster)
            n_cluster = len(cluster)
    
            timeout_for_clustering = starting_time + 60*1*((it+1)/len(real_clusters))
            check_timeout_for_clustering = timeout_for_clustering-time.time() #Time left for this cluster

            D_clus=[]
            for i in cluster:
                D_clus.append([])
                for j in cluster:
                    D_clus[-1].append(D[i][j])
                D_clus[-1].append(0)
            D_clus.append([0 for k in range(n_cluster+1)])

            solv, pred = small_SAT_Solver(n_cluster)

            best_obj = sf.up_bound(n_cluster, D_clus)

            opt = False
            while time.time() < timeout_for_clustering and not opt:
                solv.set('timeout', int(check_timeout_for_clustering*1000)) #time left in millisec 
                solv.push()

                tmp_dist = [pred[item][item_2] for item in range(n_cluster) for item_2 in range(n_cluster) for _ in range(D_clus[item_2][item])]
                solv.add(at_most_k(tmp_dist, best_obj-1))
                
                if solv.check()==sat:
                    tmp_model = solv.model()
                    item_pred = [(i,j) for j in range(n_cluster+1) for i in range(n_cluster+1) if tmp_model.evaluate(pred[i][j])]
                    tmp_obj = sf.obj_fun_clus(item_pred, n_cluster, D_clus)
                    if tmp_obj<best_obj:
                        best_model=tmp_model
                        best_obj=tmp_obj
                    check_timeout_for_clustering = timeout_for_clustering-time.time() #Time left
                    solv.pop()
                else:
                    cluster_copy=copy.deepcopy(cluster)
                    cluster_copy.append(-1)
                    clusters_paths.append([(cluster_copy[i],cluster_copy[j]) for i in range(n_cluster+1) for j in range(n_cluster+1) if best_model.evaluate(pred[i][j])])
                    opt = True
            if not opt:
                cluster_copy=copy.deepcopy(cluster)
                cluster_copy.append(-1)
                clusters_paths.append([(cluster_copy[i],cluster_copy[j]) for i in range(n_cluster+1) for j in range(n_cluster+1) if best_model.evaluate(pred[i][j])])
        
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
                    D_new[-1].append( D[first_items_for_clusters[i]][last_item_for_clusters[j]] )

        big_sol = SAT_MCP(n_new, m, s_clusters, l, D_new, 'default', timeout)

        if big_sol['default']['optimal'] == True:
            sol = sf.solution_maker_cluster(clusters, clusters_paths, first_items_for_clusters, big_sol['default']['sol'], m)
            solutions['clustering'] = {'time' : int(time.time() - starting_time) , 'optimal' : False , 'obj' : sf.obj_fun_from_solution(sol, n, D) , 'sol' : sol} #Da cambiare!!!
        
        else:
            solutions['clustering'] = {'time' : 300 , 'optimal' : False , 'obj' : 'N/A' , 'sol' : []}

    return solutions


        


instance_n=5 #from 1 to 21

if instance_n<10:
    file_name='inst0'+str(instance_n)+'.dat'
else:
    file_name='inst'+str(instance_n)+'.dat'

file = open('Documents/GitHub/Uni/amaZinc/SAT/Instances/inst07.dat')

splitted_file = file.read().split('\n')

m = int(splitted_file[0])
n = int(splitted_file[1])
l = list(map(int, splitted_file[2].split(' ')))
tmp_sz=splitted_file[3].split(' ')
if '' in tmp_sz:
    s=list(map(int, [tmp_sz[i] for i in range(len(tmp_sz)) if tmp_sz[i]!='']))
else:
    s = list(map(int, splitted_file[3].split(' ')))
D = [list(map(int, line.strip().split(' '))) for line in splitted_file[4:(n+5)]]

print('Instance number '+str(instance_n)+': '+str(n)+' items and '+str(m)+' couriers.')

print(SAT_MCP(n, m, s, l, D, ['clustering']))