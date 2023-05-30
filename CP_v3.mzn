%Version with shorter p and d arrays

include "globals.mzn";

int: n; %number of items
int: m; %number of couriers

array[1..n] of int: s; %array of items' sizes

%array[1..n+1] of int: s2= s++[0];

array[1..m] of int: l; %array of couriers' capacities

array[1..n+1,1..n+1] of int: D; %Matrix of the distances

array[1..n+m] of var 1..n+m: p; %array of predecessors of each item and of each ending point

array[1..n+m] of var 1..n+m: d; %array of successors of each item and of each starting point 

array[1..n+m] of var 1..m: v;

array[1..n+m] of var 1..sum(i in 1..n)(s[i]) :q;


%CONSTRAINTS:

constraint alldifferent(p)::domain_propagation; %all different predecessors
constraint alldifferent(d)::domain_propagation; %all different successors


constraint forall(i in 1..n+m) (
                    d[p[i]]=i);
                    
constraint forall(i in 1..n+m) (
                    p[d[i]]=i);  %coherence constraints

constraint forall(i in 1..m) (
                    v[n+i]=i );
                    
constraint forall(i in 1..n+m) (
                    v[p[i]]=v[i]);

constraint forall(i in 1..n+m) (
                    v[p[i]]=v[i]); %routes constraint
                    
constraint forall(i in 1..n) (
                    q[i]=q[p[i]]+s[i]);
constraint forall(i in n+1..n+m) (
                    q[i]=q[p[i]]);
constraint forall(i in 1..n) (
                    q[i]=q[d[i]]-s[i]); %add bound consistency??
                    
constraint forall(i in 1..n) (
                    q[i]<=l[v[i]] ); %!!!
