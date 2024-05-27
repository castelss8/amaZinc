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

def max_z3(vec):
  m = vec[0]
  for value in vec[1:]:
    m = If(value > m, value, m)
  return m

def big_SMT_Solver(n, m, s, l):

    solv= Solver()

    #Create a (n+m)-long array cour: cour[i] == j iff the i-th item (or starting/ending point) is taken by the j-th courier
    cour = [Int(f"c{i}") for i in range(n+m)]

    #Set cour range
    for i in range(n):
        solv.add(cour[i] < m )
        solv.add(cour[i] >= 0 )

    #The last m elements of cour are fixed: the courier cour starts and ends at the deposit n+cour
    for i in range(m):
        solv.add(cour[n+i] == i)
        #solv.add(cour[n+m+i] == i)

    #Weight constraint
    for c in range(m):
        solv.add(Sum([If(cour[i] == c, s[i], 0) for i in range(n)]) <= l[c])

    #Create a (n+m)-long array: pred[i] == j if the j-th item is the predecessor of the i-th item. 
    #The n+c element in the matrix is the starting/ending point of the c-th courier.
    pred = [Int(f"pred({i})")for i in range(n+m)]

    #Set pred range
    for i in range(n+m):
        solv.add(pred[i] < n+m)
        solv.add(pred[i] >= 0)

    #Each item/ending point has a different predecessor
    solv.add(Distinct([pred[i] for i in range(n+m)]))

    #If the courier cour has the item i and the item j is the predecessor of the item i then cour has the item j
    for c in range(m):
        solv.add(And(  [Implies(And([ cour[i] == c, pred[i] == j] ) , cour[j] == c) for i in range(n+m) for j in range(n+m)]  ))

    #Create a list of length n no_loop: if the item j is predecessor of the item i then no_loop(i) as an integer is > than no_loop(j) as an integer 
    no_loops = [BitVec(f'no_loop{i}', 16) for i in range(n)]
    solv.add(And([Implies(pred[i] == j, no_loops[i]>no_loops[j]) for i in range(n) for j in range(n)]))

    return solv, pred, cour


def small_SMT_Solver(n):
    solv= Solver()

    #Create a (n+1)-long array: pred[i] == j if the j-th item is the predecessor of the i-th item. 
    #The n-th element in the matrix is the starting/ending point.
    pred = [Int(f"pred({i})")for i in range(n+1)]

    #Set pred range
    for i in range(n+1):
        solv.add(pred[i] < n+1)
        solv.add(pred[i] >= 0)

    #Each item/ending point has a different predecessor
    solv.add(Distinct([pred[i] for i in range(n+1)]))

    #Create a list of length n no_loop: if the item j is predecessor of the item i then no_loop(i) as an integer is > than no_loop(j) as an integer 
    no_loops = [BitVec(f'no_loop{i}', 16) for i in range(n)]
    solv.add(And([Implies(pred[i] == j, no_loops[i]>no_loops[j]) for i in range(n) for j in range(n)]))
    
    return solv, pred


def SMT_MCP(n:int, m:int, s:list, l:list, D:list, approaches:list, tot_time = 300):
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

        solv, pred, cour = big_SMT_Solver(n, m, s, l)

        # Time
        starting_time = time.time()
        timeout = starting_time + tot_time #Ending time
        check_timeout = timeout-time.time() #Time left
        best_obj = sf.up_bound(n,D)

        stop = False
        while time.time() < timeout and not stop:

            solv.set('timeout', int(check_timeout*1000)) #time left in millisec 

            solv.push()
           
            for c in range(m):
                starting_point = [pred[item] == n+c for item in range(n)]
                starting_point_distances = [D[n][item] for item in range(n)]
                ending_point = [pred[n+c] == item for item in range(n)]
                ending_point_distances = [D[item][n] for item in range(n)]
                mid_points = [And(cour[i] == c, pred[i] == j) for i in range(n) for j in range(n)]
                mid_points_distances = [D[j][i] for i in range(n) for j in range(n)]

                points  = starting_point + ending_point + mid_points
                distances = starting_point_distances + ending_point_distances + mid_points_distances
                solv.add(PbLe([ (points[i], distances[i]) for i in range(len(points)) ], best_obj-1))

            if solv.check()==sat:
                tmp_model = solv.model()
                item_pred, cour_item = [(i,j) for j in range(n+m) for i in range(n+m) if tmp_model.evaluate(pred[i]).as_long() == j], [(c, i) for i in range(n) for c in range(m) if tmp_model.evaluate(cour[i]).as_long() == c]
                tmp_obj = sf.obj_fun(item_pred, cour_item, n, m, D)
                if tmp_obj<best_obj:
                    best_obj=tmp_obj
                check_timeout = timeout-time.time() #Time left
                solv.pop()

            elif best_obj != sf.up_bound(n,D) and time.time() < timeout: #else (no new solutions are found), if it found at least one solution and there is still time then it has found the optimal solution!
                solutions['default'] = {'time' : int(time.time() - starting_time) , 'optimal' : True , 'obj' : best_obj , 'sol' : sf.solution_maker(item_pred, cour_item, n, m)}
                stop = True

            elif best_obj != sf.up_bound(n,D): #else (no more solution and not optimal reached and no more time), if at least one solution was found:
                solutions['default'] = {'time' : 300 , 'optimal' : False , 'obj' : best_obj, 'sol' : sf.solution_maker(item_pred, cour_item, n, m)}
            
            else: #no more solution and not optimal reached and no solution found at all
                solutions['default'] = {'time' : 300 , 'optimal' : False , 'obj' : 'N/A' , 'sol' : []}
                stop = True   
    
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

            solv, pred = small_SMT_Solver(n_cluster)

            best_obj = sf.up_bound(n_cluster, D_clus)

            stop = False

            while time.time() < timeout_for_clustering and not stop:
                solv.set('timeout', int(check_timeout_for_clustering*1000)) #time left in millisec 
                solv.push()

                points = [pred[i] == j for i in range(n) for j in range(n)]
                distances = [D[j][i] for i in range(n) for j in range(n)]

                points  = starting_point + ending_point + mid_points
                distances = starting_point_distances + ending_point_distances + mid_points_distances
                solv.add(PbLe([ (points[i], distances[i]) for i in range(len(points)) ], best_obj-1))
                
                if solv.check()==sat: #If a new solution for this cluster is found:
                    tmp_model = solv.model()
                    item_pred = [(i,j) for j in range(n_cluster+1) for i in range(n_cluster+1) if tmp_model.evaluate(pred[i][j])]
                    tmp_obj = sf.obj_fun_clus(item_pred, n_cluster, D_clus)
                    if tmp_obj<best_obj:
                        best_model=tmp_model
                        best_obj=tmp_obj
                    check_timeout_for_clustering = timeout_for_clustering-time.time() #Time left
                    solv.pop()
                elif best_obj != sf.up_bound(n_cluster, D_clus): #else (no new sol for this cluster) if at least one solutios was found save it
                    cluster_copy=copy.deepcopy(cluster)
                    cluster_copy.append(-1)
                    clusters_paths.append([(cluster_copy[i],cluster_copy[j]) for i in range(n_cluster+1) for j in range(n_cluster+1) if best_model.evaluate(pred[i][j])])
                    stop = True
                else: #else (no new solution and it didn't found any solution at all) save a standard solution
                    cluster_copy=copy.deepcopy(cluster)
                    cluster_copy.append(-1)
                    cluster_copy[:0] = [-1]
                    clusters_paths.append([(cluster_copy[i],cluster_copy[i+1]) for i in range(n_cluster+1)])
                    stop = True


                
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

file = open('Documents/GitHub/Uni/amaZinc/SAT/Instances/inst21.dat')

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

print(SMT_MCP(n, m, s, l, D, ['default']))