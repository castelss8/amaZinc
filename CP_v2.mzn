include "globals.mzn";

int: n; %number of items
int: m; %number of couriers

array[1..n] of int: s; %array of items' sizes

%array[1..n+1] of int: s2= s++[0];

array[1..m] of int: l; %array of couriers' capacities

array[1..n+1,1..n+1] of int: D; %Matrix of the distances

array[1..n+(2*m)] of var 1..n+(2*m): p; %array of predecessors of each item and of each starting point and of each ending point

array[1..n+(2*m)] of var 1..n+(2*m): d; %array of successors of each item and of each starting point and of each ending point

%Why p and d are so long?? 

array[1..n+(2*m)] of var 1..m: v; %vehicles at each items

array[1..n+(2*m)] of var 0..sum(i in 1..n)(s[i]) :q; %quantity of goods of courier that arrives at location i

%CONSTRAINTS:

constraint forall(i in 1..m) (
                    p[n+i]=n+m+i);  %predecessors of starting points
constraint forall(i in 1..m) (
                    d[n+m+i]=n+i); %successors of ending point
                    
constraint alldifferent(p)::domain_propagation; %all different predecessors
constraint alldifferent(d)::domain_propagation; %all different successors

% domain_propagation??

constraint forall(i in 1..n) (
                    d[p[i]]=i);
constraint forall(i in n+m+1..n+(2*m)) (
                    d[p[i]]=i);
                    
constraint forall(i in 1..n+m) (
                    p[d[i]]=i);  %coherence constraints
                    
constraint forall(i in 1..m) (
                    ( v[n+i]=v[n+m+i] ) /\ ( v[n+i] = i ) ); % /\ ( v[n+m+i] = i ) ??? %couriers for each starting and ending point
                    
constraint forall(i in 1..n) (
                    v[p[i]]=v[i]);
constraint forall(i in n+m+1..n+(2*m)) (
                    v[p[i]]=v[i]);
constraint forall(i in 1..n+m) (
                    v[d[i]]=v[i]); %routes constraints

constraint forall(i in n+1..n+m) (
                    q[i]=0);

constraint forall(i in 1..n) (
                    q[i]=q[p[i]]+s[i]);
constraint forall(i in n+m+1..n+(2*m)) (
                    q[i]=q[p[i]]);
constraint forall(i in 1..n) (
                    q[i]=q[d[i]]-s[i]); %"quantity of goods" computation + BOUND CONSISTENCY??

constraint forall(i in 1..n+(2*m)) (
                    q[i]<=l[v[i]] );

%COST FUNCTION

array[1..n+(2*m)] of var 0..(sum(i,j in 1..n)(D[i,j])): dists;

constraint forall(i in n+1..n+m) (
                    dist[i]=0);


%!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
constraint forall(i in 1..n) (
                    dist[i]=dist[p[i]]+D[p[i],i]);
constraint forall(i in n+m+1..n+(2*m)) (
                    dist[i]=dist[p[i]]);
constraint forall(i in 1..n) (
                    dist[i]=q[d[i]]-s[i]); %"quantity of goods" computation + BOUND CONSISTENCY??





var int: distance = max (i in 1..m) ( D[n+1,d[n+i]] + sum(j in 1..n-1)(D[C[i,j],C[i,j+1]]) + D[C[i,n],n+1] );
