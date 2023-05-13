%This generate a n+1 x n+1 matrix D that follows all the rules of the distance matrix. To change the matrix at each generation is necessary to change the "Random Seed" in the "Configuration Editor". 


include "globals.mzn";

int: n;
int: max_D;

array[1..n+1, 1..n+1] of var 0..max_D: D;



constraint global_cardinality( [D[i,i] | i in 1..n+1], [0], [n+1]);
constraint global_cardinality( [D[i,j] | i,j in 1..n+1], [0], [n+1]);

constraint forall(i,j,k in 1..n+1) (
                  D[i,k]<=D[i,j]+D[j,k]
                  );
                  
                  
solve :: int_search(D, first_fail, indomain_random)
      satisfy;