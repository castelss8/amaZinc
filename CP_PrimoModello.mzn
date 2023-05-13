include "globals.mzn";

int: n;
int: m;

array[1..n] of int: s;
%array[1] of int: t;
array[1..n+1] of int: s2= s++[0];
array[1..m] of int: l;

array[1..n+1,1..n+1] of int: D;

array[1..m, 1..n] of var 1..n+1: C; %matrix of the couriers. C[i,j]= the j-th item taken by the i-th courier

%each item must be taken exactly once
constraint global_cardinality([C[i,j] | i in 1..m, j in 1..n], [i | i in 1..n], [1 | i in 1..n]); 

%each courier has a max capacity
constraint forall(i in 1..m) (
                  sum(k in [C[i,j] | j in 1..n])(s2[k])<=l[i]
                  );
                  
%after a courier has returned to the origin his tour ends.                  
constraint forall(i in 1..m) (
                  forall(j in 1..n) (
                          (C[i,j]<n+1) \/ (all_equal([C[i,k] | k in j..n]))
                          )
                  );
                  
%objective function to minimize
var int: distance = max (i in 1..m) ( D[n+1,C[i,1]] + sum(j in 1..n-1)(D[C[i,j],C[i,j+1]]) + D[C[i,n],n+1] );


solve minimize distance;

output [
    
    "distance = ", show(distance), "\n"
];

