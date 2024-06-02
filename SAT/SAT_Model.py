""" Sat Model for MCP

This script contains the definition of two solvers (big_SAT_Solver and small_SAT_Solver) and the definition of a function (SAT_MCP) that uses the two solvers
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

# Definition of standard constraints
def exactly_k(var: list[BoolRef], k: int):
    return PbEq([(v,1) for v in var], k)

def exactly_one(var: list[BoolRef]):
    return PbEq([(v,1) for v in var], 1)

def at_most_k(var: list[BoolRef], k: int):
    return PbLe([(v, 1) for v in var], k)


def big_SAT_Solver(n, m, s, l):
    """ This function creates and returns a SAT solver that can find a correct solution of the MCP problem.

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
    pred : list of lists of z3.Bool
        The variables assigning to each item its predecessor
    cour : list of lists of z3.Bool
        The variables assigning to each courier the items he is carrying
    
    """
    solv= Solver()

    # cour is an m x (n+m) matrix: cour[c][j] ==True iff the c-th courier takes the j-th item (or starting/ending point).
    cour=[[Bool(f"cour({c})_{i})") for i in range(n+m)]for c in range(m)]

    # Each pack has to be carried by exactly one courrier <-> each column of cour has exactly one True
    for i in range(n):
        solv.add(exactly_one([cour[c][i] for c in range(m)]))

    # The courier c starts and ends at the deposit n+c <-> for every i, the last m columns of the i-th row are all False except the m+i-th one that is True.
    for c in range(m):
        for i in range(m):
            if i==c:
                solv.add(cour[c][n+i])
            else:
                solv.add(Not(cour[c][n+i]))

        print('75')

    # Each courier can't carry more weight than its capacity <-> for every courier c we build an array (cour_weight) that, for every item, contains copies of 
    # cour[c][item] in number equal to the weight of item. Therefore if cour[c][item] is True, in cour_weight there are s[item] True (among the others). Then
    # we impose that the number of True in cour_weight for a coruier c is at most the equal to the capacity of c.
    for c in range(m):
        cour_weight = [cour[c][item] for item in range(n) for _ in range(s[item])]
        solv.add(at_most_k(cour_weight, l[c]))

    print('84')

    # pred is an (n+m)x(n+m) matrix: pred[i][j] = true if the j-th item is the predecessor of the i-th item. 
    # pred[i][n+c] = True if the item i is the first item taken by the c-th courier. 
    # pred[n+c][i] = True if the item i is the last item taken by the c-th courier.
    pred = [[Bool(f"pred({i})_{j}")for j in range(n+m)]for i in range(n+m)]

    # Each item/ending point has exactly one predecessor and each item/starting point is predecessor of exactly one other item <-> each column and each row of 
    # pred has exactly one True. 
    for i in range(n+m):
        col_i = []
        for j in range(n+m):
            col_i += [pred[j][i]]
        solv.add(exactly_one(col_i))
        solv.add(exactly_one(pred[i]))

    print(100)

    #In one route all the items have to be carried by the same courier <-> If the courier c carries the item i and the item j is the predecessor of the item i then 
    # c carries the item j
    for courier in range(m):
        solv.add(And(  [Implies(And([ cour[courier][i], pred[i][j]] ) , cour[courier][j]) for i in range(n+m) for j in range(n+m)]  ))

    print(107)

    # Each courier should start and finish his route in the origin and can't take an item twice. In other words, no internal loops are admitted.
    # avoid_loops is a n x al_max matrix: for every item, avoi_loops[item][:] is an array such that if one index is False then also all the previous indexes are False
    # For example avoid_loops[j] = [0 0 0 0 0 0 0 0 0 1 1 1 1 1]
    al_max = n//(m-1)+1
    avoid_loops = [[Bool(f"avoid_loops({item})_{j}")for j in range(al_max)]for item in range(n)]
    for item in range(n):
        for k in range(al_max):
            solv.add(Implies(Not(avoid_loops[item][k]), Not(Or(avoid_loops[item][:k])))) #k=False then all the previous ones are False (-> if j>k is True then all the following ones are True)
    print(117)
    # If item_j is predecessor of item_i (pred[item_i][item_j] = True) then avoid_loops[item_i][:] has more True than avoid_loops[item_j][:]. Since avoid loops has
    # the Falses before the Trues, this is equivalent to say that for at least one k avoid_loops[item_i][k] = True and avoid_loops[item_j][k] = False
    for item_i in range(n):
        for item_j in range(n):
            solv.add(Implies(pred[item_i][item_j], Or([  And(avoid_loops[item_i][k], Not(avoid_loops[item_j][k])) for k in range(al_max)  ])  ))
    print(123)
    return solv, pred, cour


def small_SAT_Solver(n):
    """ This function creates and returns a SAT solver that can find a correct solution of a simplified MCP problem, with only one courier and no capacity bounds.

    Parameters
    ----------
    n : int
        The number of items

    Returns
    -------
    solv : z3.Solver()
        The solver with all the constraints added
    pred : list of lists of z3.Bool
        The variables assigning to each item its predecessor
    
    """
    solv= Solver()
   
    # pred is an (n+1)x(n+1) matrix: pred[i][j] = true if the j-th item is the predecessor of the i-th item. 
    # pred[i][n] = True if the item i is the first item of the route. 
    # pred[n][i] = True if the item i is the last item of the route.
    pred = [[Bool(f"pred({i})_{j}")for j in range(n+1)]for i in range(n+1)]

    # Each item/ending point has exactly one predecessor and each item/starting point is predecessor of exactly one other item <-> each column and each row of 
    # pred has exactly one True.
    for i in range(n+1):
       col_i = []
       for j in range(n+1):
          col_i += [pred[j][i]]
       solv.add(exactly_one(col_i))
       solv.add(exactly_one(pred[i]))
    
    # The avoid_loops matrix is equal to the avoid_loops matrix described in big_SAT_Solver
    avoid_loops = [[Bool(f"avoid_loops({item})_{j}")for j in range(n)]for item in range(n)]
    for item in range(n):
       for k in range(n):
          solv.add(Implies(Not(avoid_loops[item][k]), Not(Or(avoid_loops[item][:k]))))
    for item_i in range(n):
       for item_j in range(n):
          solv.add(Implies(pred[item_i][item_j], Or([And(avoid_loops[item_i][k], Not(avoid_loops[item_j][k]))for k in range(n)])  ))
   
    return solv, pred


def SAT_MCP(n:int, m:int, s:list, l:list, D:list, approaches:list, tot_time = 300):
    """ SAT_MCP function, given an istance and a list of approaches, perform a search and returns the solutions found
    
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

    #'default' approach searches the optimal solution using big_SAT_Solver
    if 'default' in approaches:
        solv, pred, cour = big_SAT_Solver(n, m, s, l)
        print(203)
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
            for courier in range(m):
                print(courier)
                tmp_dist = [And(cour[courier][item], pred[item][item_2]) for item in range(n) for item_2 in range(n) for _ in range(D[item_2][item])]
                tmp_dist += [pred[item][n+courier] for item in range(n) for _ in range(D[n][item])]
                tmp_dist += [pred[n+courier][item] for item in range(n) for _ in range(D[item][n])]
                solv.add(at_most_k(tmp_dist, best_obj-1))
            print(221)
            if solv.check()==sat and time.time() < timeout:
                print('check!')
                # the problem is Satisfiable -> A new solution has been found. Update the best objective function's value, the corresponding solution and the time left.
                tmp_model = solv.model()
                item_pred, cour_item = [(i,j) for j in range(n+m) for i in range(n+m) if tmp_model.evaluate(pred[i][j])], [(i, j) for j in range(n) for i in range(m) if tmp_model.evaluate(cour[i][j])] 
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
                # No new solutions, no solutions found at all -> it didn't find any solution. Save an empty solution
                solutions['default'] = {'time' : 300 , 'optimal' : False , 'obj' : 'N/A' , 'sol' : []}
                stop = True   

        if time.time() > timeout and best_obj != sf.up_bound(n,D) and not 'default' in list(solutions.keys()):
            solutions['default'] = {'time' : 300 , 'optimal' : False , 'obj' : best_obj, 'sol' : sf.solution_maker(item_pred, cour_item, n, m)}



    #'clustering' approach creates clusters, orders them using small_SAT_Solver, creates a new distance matrix and a new items' weights array and perform a 'default'
    # search on the reduced problem. 
    if 'clustering' in approaches:
        #Find some clusters
        clusters, s_clusters = cl.complete_clustering(D, s, n, m)
        real_clusters = [cluster for cluster in clusters if len(cluster)>1]
        clusters_paths=[]

        starting_time = time.time()
        timeout = starting_time + 60*5
        print(real_clusters)
        for it in range(len(real_clusters)):
            cluster = real_clusters[it]
            print(real_clusters[it])
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

            solv, pred = small_SAT_Solver(n_cluster)
            best_obj = sf.up_bound(n_cluster, D_clus)
            stop = False

            while time.time() < timeout_for_clustering and not stop:
                solv.set('timeout', int(check_timeout_for_clustering*1000)) #time left in millisec 
                solv.push()

                #Add the upper bound to the objective function
                tmp_dist = [pred[item][item_2] for item in range(n_cluster) for item_2 in range(n_cluster) for _ in range(D_clus[item_2][item])]
                if len(tmp_dist)>0:
                    solv.add(at_most_k(tmp_dist, best_obj-1))
                
                if solv.check()==sat:
                # the problem is Satisfiable -> A new solution has been found. Update the best objective function's value, the corresponding solution and the time left.
                    tmp_model = solv.model()
                    item_pred = [(i,j) for j in range(n_cluster+1) for i in range(n_cluster+1) if tmp_model.evaluate(pred[i][j])]
                    tmp_obj = sf.obj_fun_clus(item_pred, n_cluster, D_clus)
                    if tmp_obj<best_obj:
                        best_model=tmp_model
                        best_obj=tmp_obj
                    check_timeout_for_clustering = timeout_for_clustering-time.time() #Time left
                    solv.pop()
                    if best_obj==0:
                        cluster_copy=copy.deepcopy(cluster)
                        cluster_copy.append(-1)
                        clusters_paths.append([(cluster_copy[i],cluster_copy[j]) for i in range(n_cluster+1) for j in range(n_cluster+1) if best_model.evaluate(pred[i][j])])
                        stop = True
                elif best_obj != sf.up_bound(n_cluster, D_clus): 
                    #No new solutions, one solution stored. Save the best solution found.
                    cluster_copy=copy.deepcopy(cluster)
                    cluster_copy.append(-1)
                    clusters_paths.append([(cluster_copy[i],cluster_copy[j]) for i in range(n_cluster+1) for j in range(n_cluster+1) if best_model.evaluate(pred[i][j])])
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

        # Recursively call SAT_MCP with default approach to solve the simplified istance
        time_for_main_search = int(timeout-time.time())
        big_sol = SAT_MCP(n_new, m, s_clusters, l, D_new, ['default'], time_for_main_search)

        #Save the complete solution
        if big_sol['default']['sol'] != []:
            sol = sf.solution_maker_cluster(clusters, clusters_paths, first_items_for_clusters, big_sol['default']['sol'], m) 
            solutions['clustering'] = {'time' : min(int(time.time() - starting_time), 300) , 'optimal' : False , 'obj' : sf.obj_fun_from_solution(sol, n, D) , 'sol' : sol}
        
        else:
            solutions['clustering'] = {'time' : 300 , 'optimal' : False , 'obj' : 'N/A' , 'sol' : []}

    return solutions