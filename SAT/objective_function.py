def objective_function(item_pred, cour_item, n, m, D):
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
    return max(distances)