import copy
import numpy as np

def one_clustering(D, s):
    '''
    input: D = list of lists;
    output:
    It create clusters of 2 items each by finding items that are closest to each other.
    '''

    n=len(D)-1
    Closest=[] #Closest[i] will contain the item that is closest to the i-th item
    for item in range(n): 
        tmp = copy.deepcopy(D[item]) #tmp is a list that contains the distances from item to all the other items
        tmp[item]=np.inf
        Closest.append(np.argmin(tmp[:-1]))

    #Create clusters
    clusters=[]
    already_in_cluster=[]
    for item in range(n):
        if item==Closest[Closest[item]] and (item not in already_in_cluster):
            clusters.append([item, Closest[item]])
            already_in_cluster.append(item)
            already_in_cluster.append(Closest[item])
        elif item!=Closest[Closest[item]]:
            clusters.append([item])
    clusters.append([n])

    #Create D_new
    D_new=[]
    n_new=len(clusters)-1
    for row in range(n_new): 
        D_new.append([]) 
        for column in range(n_new): 
            if row==column:
                D_new[row].append(0)
            else:
                tmp_dist=[]
                for i in clusters[row]:
                    for j in clusters[column]:
                        tmp_dist.append(D[i][j])
                D_new[row].append(np.mean(tmp_dist))
        tmp_dist=[]
        for i in clusters[row]:
            tmp_dist.append(D[i][n])
        D_new[row].append(np.mean(tmp_dist))
    D_new.append([])

    for column in range(n_new):
        tmp_dist=[]
        for i in clusters[column]:
            tmp_dist.append(D[n][i])
        D_new[n_new].append(np.mean(tmp_dist))
    D_new[n_new].append(0)

    #Create s_new
    s_new=[]
    for item in range(n_new):
        s_new.append(sum([s[i] for i in clusters[item]]))

    return D_new, clusters, s_new

def k_clustering(D, s, k):
    D_new=D
    s_new=s
    n=len(D)-1
    clusters=[ [i] for i in range(n+1) ]
    for _ in range(k):
        D_new, new_clusters, s_new = one_clustering(D_new, s_new)
        old_clusters=clusters
        clusters=[]
        for clus in new_clusters:
            clusters.append([])
            for item in clus:
                for old_item in old_clusters[item]:
                    clusters[-1].append(old_item)
    return clusters, s_new
    

def complete_clustering(n, m, s, D):
    mean_num_of_items = (n // m)
    D_new = D
    s_new = s
    clusters = [ [i] for i in range(n+1) ]
    old_clusters = None
    while np.max( [len(clusters[i]) for i in range(len(clusters))] ) < mean_num_of_items and clusters != old_clusters:
        s_old = s_new
        old_clusters=clusters
        D_new, new_clusters, s_new = one_clustering(D_new, s_new)
        clusters=[]
        for clus in new_clusters:
            clusters.append([])
            for item in clus:
                for old_item in old_clusters[item]:
                    clusters[-1].append(old_item)
    return old_clusters, s_old

    

