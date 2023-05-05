include "globals.mzn";

int: n;
int: m;

array[1..n] of int: s;
%array[1] of int: t;
array[1..n+1] of int: s2= s++[0];
array[1..m] of int: l;

array[1..n+1,1..n+1] of int: D;

array[1..m, 1..n] of var 1..n+1: C; %matrix of the couriers. C[i,j]= the j-th item taken by the i-th courier

constraint global_cardinality([C[i,j] | i in 1..m, j in 1..n], [i | i in 1..n], [1 | i in 1..n]); 

constraint forall(i in 1..m) (
                  sum(k in [C[i,j] | j in 1..n])(s2[k])<=l[i]
                  );