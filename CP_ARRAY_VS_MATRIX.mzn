%% Best Performances: Random Seed, Free Search, Optimizer: -O5, Solver Configuration: OR Tools

include "globals.mzn";

int: n; %number of items
int: m; %number of couriers

array[1..n] of int: s; %array of items' sizes

%array[1..n+1] of int: s2= s++[0];

array[1..m] of int: l; %array of couriers' capacities

array[1..n+1,1..n+1] of int: D; %Matrix of the distances





%MATRIX ----------------------------------------------------------------------------

%/*

array[1..m, 1..n] of var 0..1 : V_new;

array[1..m] of var 0..sum(i in 1..n)(s[i]): q_new;

%Ogni pacco preso da un solo corriere
constraint forall(i in 1..n) (sum(j in 1..m)(V_new[j,i])==1);

constraint forall(i in 1..m) (q_new[i] = sum(j in 1..n)(V_new[i, j]*s[j]));

constraint forall(i in 1..m) (
                    q_new[i]<=l[i] )::bounds;

solve satisfy;

%*/

%ARRAY -----------------------------------------------------------------------------

/*

array[1..n] of var 1..m: v;

array[1..m, 1..n] of var 0..(sum(i in 1..n)(s[i])): q;

constraint forall(i in 1..m) (if i==v[1] then q[i,1] = s[1] else q[i,1] = 0 endif);

constraint forall(i in 1..m, j in 2..n) (if v[j]==i then q[i,j] = q[i,j-1]+s[j] else q[i,j] = q[i,j-1] endif);

constraint forall(i in 1..m) (q[i,n]<=l[i] )::bounds;

solve satisfy;

*/

