include "globals.mzn";
include "exactly_int.mzn";

int: n; %number of items
int: k; %number of clusters
array [1..k] of int: c; %number of elements in each cluster
array [1..k, 1..max(c)] of -1..n: C; %matrix of clusters: in the i-th row there are the items of the i-th cluster (NOT ORDERED)
array[1..n+1,1..n+1] of int: D; %Matrix of distances

array [1..k] of var int: d; %d[i] is the total distance inside the cluster i 

array [1..k, 1..max(c)] of var 0..n+1: C_opt; %matrix of clusters: in the i-th row there are the items of the i-th cluster (ORDERED)
int: one = 1;

constraint forall(i in 1..k,j in (c[i]+1)..max(c))(C_opt[i,j] = n+1); %if the cluster contains less elements then the largest cluester, fill with n+1

constraint forall(i in 1..k) (forall(j in 1..c[i])(exactly_int(one,C_opt[i,1..c[i]], C[i,j])==true)); %for every cluster, exactly one element in C_opt takes the value of each element of C (WHY 2 FORALL??)


constraint forall(i in 1..k)(d[i] = D[n+1,C_opt[i,1]+1] + sum(j in 2..c[i])(D[C_opt[i,j-1]+1,C_opt[i,j]+1]) + D[C_opt[i,c[i]],n+1]+1); %sum of distances in each cluster NB IL +1 SERVE PERCHè C HA I PACCHI TUTTI TRASLATI INDIETRO DI 1 :D

solve minimize max(i in 1..k)(d[i]);






