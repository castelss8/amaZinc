%% Best Performances: Random Seed, Free Search, Optimizer: -O5, Solver Configuration: OR Tools

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

array[1..n+(2*m)] of var 0..max(i in 1..m)(l[i]) :q; %quantity of goods of courier that arrives at location i


%CONSTRAINTS:

constraint forall(i in 1..m) (
                    p[n+i]=n+m+i)::domain;  %predecessors of starting points are corresponding ending point
                    
constraint redundant_constraint(forall(i in 1..m) (
                    d[n+m+i]=n+i))::domain; %successors of ending point
                    
                    
                    
constraint alldifferent(p)::domain; %all different predecessors

constraint redundant_constraint(alldifferent(d)); %::domain; %all different successors



% domain_propagation??

%no_loops constraint (TODO)
array[1..n] of var 1..n: no_loops

constraint forall(i in 1..n) ()

%coherence constraints:
constraint redundant_constraint(forall(i in 1..n) (
                    d[p[i]]=i)); %::domain; %successor of the predecessor of i is i itself (for items)
constraint redundant_constraint(forall(i in n+m+1..n+(2*m)) (
                    d[p[i]]=i)); %::domain; % same (for ending points)
                    
constraint redundant_constraint(forall(i in 1..n+m) (
                    p[d[i]]=i)); %::domain;  
                    
                    
                    

%Vehicles constraints:
constraint forall(i in 1..m) (
                    ( v[n+i]=v[n+m+i] ))::domain; % /\ ( v[n+m+i] = i ) ??? %couriers for each starting and ending point

constraint forall(i in 1..m) (           
                    ( v[n+i] = i ) )::domain;
                    
constraint redundant_constraint(forall(i in 1..m) (v[n+m+i] = i));
                    
constraint forall(i in 1..n) (
                    v[p[i]]=v[i])::domain;
                    
constraint forall(i in n+m+1..n+(2*m)) (
                    v[p[i]]=v[i])::domain;
                    
constraint redundant_constraint(forall(i in 1..n+m) (
                    v[d[i]]=v[i])); %::domain; %routes constraints



%Capacity constraints:
constraint forall(i in n+1..n+m) (
                    q[i]=0)::domain;

constraint forall(i in 1..n) (
                    q[i]=q[p[i]]+s[i])::domain;
                    
constraint forall(i in n+m+1..n+(2*m)) (
                    q[i]=q[p[i]])::domain;  
                   
                    
constraint redundant_constraint(forall(i in 1..n) (
                    if d[i]<=n then q[i]=q[d[i]]-s[d[i]] else q[i] = q[d[i]] endif ))::bounds; %"quantity of goods" computation + BOUND CONSISTENCY??


constraint forall(i in 1..n+(2*m)) (
                    q[i]<=l[v[i]] )::bounds;


%COST FUNCTION

%dists
%array[1..n+(2*m)] of var 0..(sum(i,j in 1..n where i<j)(D[i,j])): dists; %upper and lower limit not optimal!

array[1..n] of int: max_D=[max([D[i,j] | j in 1..i]) | i in 1..n];
int: up_bound=sum(i in 1..n)(max_D[i]);
array[1..n+(2*m)] of var 0..up_bound: dists; %Alternative upper bound. No real improvement :(

constraint forall(i in n+1..n+m) (
                    dists[i]=0)::domain;


%!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
%Computation of dists
constraint forall(i in 1..n) (
                    if p[i]>n then dists[i]=D[n+1,i] else dists[i]=dists[p[i]]+D[p[i],i] endif)::domain;
                    
constraint forall(i in n+m+1..n+(2*m)) (
                    if p[i]>n then dists[i]=0 else dists[i]=dists[p[i]]+D[p[i],n+1] endif
                    )::domain;
                    
constraint redundant_constraint(forall(i in 1..n) (
                    if d[i]>n then dists[i]=dists[d[i]]-D[i,n+1] else dists[i]=dists[d[i]]-D[i,d[i]] endif ))::domain; %starting points should be considered here??

constraint redundant_constraint(forall(i in n+1..n+m) (
                    if d[i]<=n then dists[d[i]]-D[n+1,d[i]]=0 else true endif
                    ))::domain;


var int: distance = max (i in n+m+1..n+(2*m)) ( dists[i] );



solve minimize distance ;
