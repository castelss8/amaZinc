include "globals.mzn";
include "fzn_at_least_int.mzn";

int: n; %number of items
int: m; %number of couriers

array[1..n] of int: s; %array of items' sizes

array[1..m] of int: l; %array of couriers' capacities

array[1..n+1,1..n+1] of int: D; %Matrix of the distances

array[1..n+(2*m)] of var 1..n+(2*m): p; %array of predecessors of each item and of each starting point and of each ending point

array[1..n+(2*m)] of var 1..n+(2*m): d; %array of successors of each item and of each starting point and of each ending point

array[1..n+(2*m)] of var 1..m: v; %vehicles at each items

array[1..n+(2*m)] of var 0..max(i in 1..m)(l[i]) :q; %quantity of goods of courier that arrives at location i

%CONSTRAINTS:

%constraints defining predecessor

constraint forall(i in 1..m) (
                    p[n+i]=n+m+i)::domain;  %predecessors of starting points

constraint alldifferent(p)::domain_propagation; %all different predecessors

%array[1..n] of var 0..n: no_loops;

%constraint forall(i in 1..n, j in 1..n)(if p[i] == j then no_loops[i] > no_loops[j] endif); VERIFICARE SE CON QUESTO ARRAY MIGLIORA LA PERFORMANCE


%constraints on vehicle
int: one = 1;

constraint forall(i in 1..m)(fzn_at_least_int(one,v[1..n],i));

constraint forall(i in 1..m) (
                    ( v[n+i]=v[n+m+i] ) /\ ( v[n+i] = i ) )::domain; 

constraint forall(i in 1..n) (
                    v[p[i]]=v[i])::domain;
                    
constraint forall(i in n+m+1..n+(2*m)) (
                    v[p[i]]=v[i])::domain;
                    
%constraints on capacity                

constraint forall(i in n+1..n+m) (
                    q[i]=0)::domain;

constraint forall(i in 1..n) (
                    q[i]=q[p[i]]+s[i])::domain;
                    
constraint forall(i in n+m+1..n+(2*m)) (
                    q[i]=q[p[i]])::domain;

constraint forall(i in 1..n+(2*m)) (
                    q[i]<=l[v[i]] );

%COST FUNCTION

array[1..n] of int: max_D=[max([D[i,j] | j in 1..i]) | i in 1..n];

int: up_bound=sum(i in 1..n)(max_D[i]);

array[1..n+(2*m)] of var 0..up_bound: dists; %Alternative upper bound. No real improvement :(

constraint forall(i in n+1..n+m) (
                    dists[i]=0)::domain;


%FUNCTION TO SOLVE

constraint forall(i in 1..n) (
                    if p[i]>n then dists[i]=D[n+1,i] else dists[i]=dists[p[i]]+D[p[i],i] endif)::domain;
                    
constraint forall(i in n+m+1..n+(2*m)) (
                    if p[i]>n then dists[i]=0 else dists[i]=dists[p[i]]+D[p[i],n+1] endif
                    )::domain;

var int: distance = max (i in n+m+1..n+(2*m)) ( dists[i] );

solve minimize distance ;