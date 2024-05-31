""" Smt Model for MCP

This script contains the definition of two solvers (big_SMT_Solver and small_SAM_Solver) and the definition of a function (SMT_MCP) that uses the two solvers
to find solutions for a given istances of the Multiple Courier Planning problem. Two approaches have been implemented: the first one ('default') search the
optimal solution while the second one ('clustering') tries to reduce the dimension of the istance, hoping to find solutions quicklier and for larger istances.

"""

from z3 import *

#import numpy as np
#from itertools import combinations
from utils import *
#import math
import time
import sys
#from importlib import reload

sys.path.append('/Users/mange/Documents/GitHub/Uni/amaZinc')
from functions import sol_functions as sf
from functions import clustering as cl 

""" TODO: da cancellare!
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
"""

def big_SMT_Solver(n, m, s, l):
    """ This function creates and returns a SMT solver that can find a correct solution of the MCP problem.

    Parameters
    ----------
    n : int
        The number of items
    m : int
        The number of couriers
    s : list of ints
        The items' sizes
    l : list of ints
        The couriers' capacities

    Returns
    -------
    solv : z3.Solver()
        The solver with all the constraints added
    pred : list z3.Int
        The variables assigning to each item its predecessor
    cour : list of z3.Int
        The variables assigning to each item the courier that is carrying it.
    
    """

    solv= Solver()

    # cour is (n+m)-long array: cour[i] == j iff the i-th item (or starting/ending point) is taken by the j-th courier
    cour = [Int(f"c{i}") for i in range(n+m)]

    # Set cour range
    for i in range(n):
        solv.add(cour[i] < m )
        solv.add(cour[i] >= 0 )

    # # The courier c starts and ends at the deposit n+c <-> The last m elements of cour are setted to the corresponding courier.
    for i in range(m):
        solv.add(cour[n+i] == i)

    # Each courier can't carry more weight than its capacity <-> for every courier c, we add the upper bound to the sum of n Ifs: each If returns the weight of the
    # item if the item is carried, 0 otherwise.
    for c in range(m):
        solv.add(Sum([If(cour[i] == c, s[i], 0) for i in range(n)]) <= l[c])

    # pred is an (n+m)-long array: pred[i] == j if the j-th item is the predecessor of the i-th item. 
    # pred[i] = n+c if the item i is the first item taken by the c-th courier.
    # pred[n+c] = i if the item i is the last item taken by the c-th courier.
    pred = [Int(f"pred({i})")for i in range(n+m)]
    # Set pred range
    for i in range(n+m):
        solv.add(pred[i] < n+m)
        solv.add(pred[i] >= 0)

    # Each item/ending point has a different predecessor
    solv.add(Distinct([pred[i] for i in range(n+m)]))

    #In one route all the items have to be carried by the same courier <-> If the courier c carries the item i and the item j is the predecessor of the item i then 
    # c carries the item j
    for c in range(m):
        solv.add(And(  [Implies(And([ cour[i] == c, pred[i] == j] ) , cour[j] == c) for i in range(n+m) for j in range(n+m)]  ))

    # Each courier should start and finish his route in the origin and can't take an item twice. In other words, no internal loops are admitted.
    # avoid_loops is a n-long array: if the item j is predecessor of the item i then avoid_loop(i) is greater than > than avoid_loop(j) as an integer 
    avoid_loops = [BitVec(f'avoid_loop{i}', 16) for i in range(n)]
    solv.add(And([Implies(pred[i] == j, avoid_loops[i]>avoid_loops[j]) for i in range(n) for j in range(n)]))

    return solv, pred, cour


def small_SMT_Solver(n):
    """ This function creates and returns a SMT solver that can find a correct solution of a simplified MCP problem, with only one courier and no capacity bounds.

    Parameters
    ----------
    n : int
        The number of items

    Returns
    -------
    solv : z3.Solver()
        The solver with all the constraints added
    pred : list of z3.Int
        The variables assigning to each item its predecessor

    """
    solv= Solver()

    # pred is an (n+1)-long array: pred[i] == j if the j-th item is the predecessor of the i-th item. 
    # pred[i] = n if the item i is the first item of the route.
    # pred[n] = i if the item i is the last item of the route.
    pred = [Int(f"pred({i})")for i in range(n+1)]

    # Set pred range
    for i in range(n+1):
        solv.add(pred[i] < n+1)
        solv.add(pred[i] >= 0)

    # Each item/ending point has a different predecessor
    solv.add(Distinct([pred[i] for i in range(n+1)]))

    # The avoid_loops array is equal to the avoid_loops array described in big_SAT_Solver
    avoid_loops = [BitVec(f'avoid_loop{i}', 16) for i in range(n)]
    solv.add(And([Implies(pred[i] == j, avoid_loops[i]>avoid_loops[j]) for i in range(n) for j in range(n)]))
    
    return solv, pred


def SMT_MCP(n:int, m:int, s:list, l:list, D:list, approaches:list, tot_time = 300):
    """ SMT_MCP function, given an istance and a list of approaches, perform a search and returns the solutions found
    
    Parameters
    ----------
    n : int 
        The number of items
    m : int
        The number of couriers
    s : list of ints
        The items' sizes
    l : list of ints
        The couriers' capacities
    D : list of lists of ints
        The distance matrix
    approaches : list of strings 
        The approaches to use ('default' or 'clustering')
    tot_time : int, optional
        Time's upper bound (equal to 300 by default)

    Returns
    -------
    solutions : dict 
        The dictionary containing the solutions. It has the approaches as keys and dictionaries containing the solution as items

    """

    solutions = {}

    #'default' approach searches the optimal solution using big_SMT_Solver
    if 'default' in approaches:

        solv, pred, cour = big_SMT_Solver(n, m, s, l)

        # Time managing
        starting_time = time.time()
        timeout = starting_time + tot_time #Ending time
        check_timeout = timeout-time.time() #Time left
        best_obj = sf.up_bound(n,D)

        stop = False
        while time.time() < timeout and not stop:
            solv.set('timeout', int(check_timeout*1000)) #time left in millisec 
            solv.push()
            
            #Add the upper bound to the objective function
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
                # the problem is Satisfiable -> A new solution has been found. Update the best objective function's value, the corresponding solution and the time left.
                tmp_model = solv.model()
                item_pred = [(i, tmp_model.evaluate(pred[i]).as_long())  for i in range(n+m)]
                cour_item = [(tmp_model.evaluate(cour[i]).as_long(), i) for i in range(n)]
                
                tmp_obj = sf.obj_fun(item_pred, cour_item, n, m, D)
                if tmp_obj<best_obj:
                    best_obj=tmp_obj
                check_timeout = timeout-time.time() #Time left
                solv.pop()

            elif best_obj != sf.up_bound(n,D) and time.time() < timeout:
                # No new solutions are found, one solution stored and there is still time -> it has found the optimal solution. Save the optimal solution.                
                solutions['default'] = {'time' : int(time.time() - starting_time) , 'optimal' : True , 'obj' : best_obj , 'sol' : sf.solution_maker(item_pred, cour_item, n, m)}
                stop = True

            elif best_obj != sf.up_bound(n,D): 
                # No new solutions, one solution stored and no more time -> the time endend and we didn't reach the optimal solution. Save the best solution found.
                solutions['default'] = {'time' : 300 , 'optimal' : False , 'obj' : best_obj, 'sol' : sf.solution_maker(item_pred, cour_item, n, m)}
            
            else: 
                # No new solutions, no solutions found at all and no more time -> it didn't find any solution in the given time. Save an empty solution
                solutions['default'] = {'time' : 300 , 'optimal' : False , 'obj' : 'N/A' , 'sol' : []}
                stop = True   
    
    #'clustering' approach creates clusters, orders them using small_SMT_Solver, creates a new distance matrix and a new items' weights array and perform a 'default'
    # search on the reduced problem. 
    if 'clustering' in approaches:
        #Find some clusters
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

            #Create D_clus: the distance matrix with only the items in this cluster
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

                #Add the upper bound to the objective function
                points = [pred[i] == j for i in range(n_cluster) for j in range(n_cluster)]
                distances = [D_clus[j][i] for i in range(n_cluster) for j in range(n_cluster)]
                solv.add(PbLe([ (points[i], distances[i]) for i in range(len(points)) ], best_obj-1))
                
                if solv.check()==sat: 
                    # the problem is Satisfiable -> A new solution has been found. Update the best objective function's value, the corresponding solution and the time left.
                    tmp_model = solv.model()
                    item_pred = [(i, tmp_model.evaluate(pred[i]).as_long())  for i in range(n_cluster+1)]
                    tmp_obj = sf.obj_fun_clus(item_pred, n_cluster, D_clus)
                    if tmp_obj<best_obj:
                        best_model=tmp_model
                        best_obj=tmp_obj
                    check_timeout_for_clustering = timeout_for_clustering-time.time() #Time left
                    solv.pop()

                elif best_obj != sf.up_bound(n_cluster, D_clus): 
                    #No new solutions, one solution stored. Save the best solution found.
                    cluster_copy=copy.deepcopy(cluster)
                    cluster_copy.append(-1)
                    clusters_paths.append([(cluster_copy[i],cluster_copy[j]) for i in range(n_cluster+1) for j in range(n_cluster+1) if best_model.evaluate(pred[i]).as_long() == j])
                    stop = True
                else: 
                    # No new solutions and  no solutions found at all. save a standard solution.
                    cluster_copy=copy.deepcopy(cluster)
                    cluster_copy.append(-1)
                    cluster_copy[:0] = [-1]
                    clusters_paths.append([(cluster_copy[i],cluster_copy[i+1]) for i in range(n_cluster+1)])
                    stop = True

        # Given all the single solutions for the clusters, save the first and last item of each cluster
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

        # Build D_new: D[item][cluster] = D[item][x] where x is the first item of the cluster and D[cluster][item] = D[x][item] where x is the last item of the cluster.
        D_new=[]
        for i in range(n_new+1):
            D_new.append([])
            for j in range(n_new+1):
                if j==i:
                    D_new[-1].append(0)
                else:
                    D_new[-1].append( D[first_items_for_clusters[i]][last_item_for_clusters[j]] )

        # Recursively call SMT_MCP with default approach to solve the simplified istance
        big_sol = SMT_MCP(n_new, m, s_clusters, l, D_new, ['default'], timeout)

        # Save the complete solution
        if big_sol['default']['sol'] != []:
            sol = sf.solution_maker_cluster(clusters, clusters_paths, first_items_for_clusters, big_sol['default']['sol'], m)
            solutions['clustering'] = {'time' : int(time.time() - starting_time) , 'optimal' : False , 'obj' : sf.obj_fun_from_solution(sol, n, D) , 'sol' : sol} 
        
        else:
            solutions['clustering'] = {'time' : 300 , 'optimal' : False , 'obj' : 'N/A' , 'sol' : []}

    return solutions


        

'''
instance_n=5 #from 1 to 21

if instance_n<10:
    file_name='inst0'+str(instance_n)+'.dat'
else:
    file_name='inst'+str(instance_n)+'.dat'

file = open('Documents/GitHub/Uni/amaZinc/SAT/Instances/inst04.dat')

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

print(SMT_MCP(n, m, s, l, D, ['default', 'clustering']))
'''