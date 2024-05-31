def obj_fun(item_pred, cour_item, n, m, D):
    '''
    input
    - item_pred: list of tuples = list of couples (i,j) such that j is predecessor of i
    - cour_item: list of tuples = list of couples (c,i) such that c is the courier that takes i
    - n : number of items
    - m : number of couriers
    - D : list of lists = distance matrix

    output
    - max_distance : int = max distance done by couriers

    It returns the value of the objective function given an insance and a solution
    '''
    distances=[0 for _ in range(m)]

    # (i,j) in item_pred => j is predecessor of i
    for i, j in item_pred:

        if i>=n and j<n:
            c=i-n
            distances[c]+=D[j][n]
        elif j>=n and i<n:
            c=j-n
            distances[c]+=D[n][i]
        elif j<n and i<n:
            found=False
            c=0
            while not found:
                if (c,i) in cour_item:
                    distances[c]+=D[j][i]
                    found=True
                c+=1
    max_distance = max(distances)
    return max_distance

def obj_fun_clus(item_pred, n, D):
    distance=0
    for i, j in item_pred:
        if i>=n:
            distance+=D[j][n]
        elif j>=n:
            distance+=D[n][i]
        else:
            distance+=D[j][i]
    return distance

def obj_fun_from_solution(sol, n, D):
    distances = []
    for path in sol:
        if len(path)!=0:
            dis = D[n][path[0]] + sum([D[path[i]][path[i+1]] for i in range(len(path)-1)]) + D[path[-1]][n]
            distances.append(dis)
    return max(distances)

def solution_maker(item_pred, cour_item, n, m):
    cour_item_pred = [[i_p for i_p in item_pred if (cour, i_p[0]) in cour_item] for cour in range(m)]
    solution = [[] for _  in range(m)]
    for cour in range(m):
        current_item = n+cour
        while len(solution[cour])!=len(cour_item_pred[cour]):
            found = False
            i=0
            while not found:
                if (i, current_item) in cour_item_pred[cour]:
                    solution[cour].append(i)
                    current_item = i
                    found = True
                else:
                    i+=1
    return solution

def solution_maker_cluster(clusters, clusters_paths, first_items_for_clusters, solution_big, m):
    solution = [[] for _  in range(m)]

    cluster_path_semplified = []
    for i in clusters_paths:
        cluster_path_semplified += i

    for c in range(m):

        for clus in solution_big[c]:

            if len(clusters[clus]) == 1:
                 solution[c].append(clusters[clus][0])
                 
            elif len(clusters[clus]) > 1 :
                ordered_cluster = []
                current_item = first_items_for_clusters[clus]

                while current_item!=-1:
                    ordered_cluster.append(current_item)
                    found = False
                    i=-1
                    while not found:
                        if (i, current_item) in cluster_path_semplified:
                            current_item = i
                            found = True

                        else:
                            i+=1

                solution[c] = solution[c] + ordered_cluster

    return solution


def up_bound(n, D):
    res = sum([D[i][i+1] for i in range(n)])
    res += D[n][0] + 1
    return res