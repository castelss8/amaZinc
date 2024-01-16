include "globals.mzn";

% a. ARRAYS FOR THE FIRST PART OF SEARCH - NON OPTIMAL SOLUTIONS

int: n; %number of items
int: m; %number of couriers

array[1..n] of int: s; %array of items' sizes

array[1..m] of int: l; %array of couriers' capacities

array[1..n+1,1..n+1] of int: D; %Matrix of distances

array[1..n+(2*m)] of var 1..n+(2*m): p; %array of predecessors of each item and of each starting point and of each ending point

array[1..n] of var 1..m: v; %array describing which courier carries each object

array[1..m, 1..n] of var 0..(sum(i in 1..n)(s[i])): q; %matrix describing the total volume carried by each courier at each point 


%a. CONSTRAINTS

%a. Construction of randomized function

array[1..((n*100)+x)] of var (1/((n*100)+x))..(1/((n*100)+x)): weights; %weights function for discrete distribution (CHANGE X IN ACTUAL INPUT VARIABLE TO WHICH THE RANDOMIZED PROCEDURE CAN BE LINKED TO)

function array[1..n] of var 1..((n*100)+x): f(var 1..100: x) = %CHANGE THE UPPER BOUND ON x (ACTUALLY, CHANGE PARAMETERS OF f)
  let {
        array[1..n] of var 0..((n*100)+x): g; %array of n random numbers
        constraint forall(i in 1..n) (g[i] = discrete_distribution(weights));
        }
        in g;

function array[1..n] of var 1..n: perm = arg_sort(f(x)); %permutation that reorders g

%a. Predecessor constraints

constraint alldifferent(p)::domain; %all different predecessors

constraint forall(i in 1..m) (p[n+i]=n+m+i)::domain;  %predecessors of starting points are corresponding ending point
                    
constraint forall(i in 1..n, j in 1..n) (if j == p[i] then perm[j] <= perm[i] endif); %Symmetry breaking - predecessors are stable with the order defined by perm
                    
%a. Vehicle constraints:

constraint forall(i in 1..m) (
                    v[n+i] = i );
                    
constraint forall(i in 1..n+m) (
                    v[p[i]] = v[i]);

constraint forall(i in 1..n+m) (
                    v[p[i]] = v[i]); %routes constraint

%a. Courier capacity constraints
constraint forall(i in 1..m) (if i == v[1] then q[i,1] = s[1] else q[i,1] = 0 endif);

constraint forall(i in 1..m, j in 2..n) (if v[j] == i then q[i,j] = q[i,j-1]+s[j] else q[i,j] = q[i,j-1] endif);

constraint forall(i in 1..m) (q[i,n]<=l[i])::bounds;


%a. INTERMEDIATE FUNCTION TO MINIMIZE

array[1..n] of int: max_D=[max([D[i,j] | j in 1..i]) | i in 1..n];
int: up_bound=sum(i in 1..n)(max_D[i]); %create upper bound for the array of distances traveled by each courier

function array[1..m] of 1..up_bound: interm_dist(p,D) = %CHANGE UPPER BOUND OF FUNCTION
  let {
       array[1..(n+(2*m))] of var 0..up_bound: id;
       constraint forall(i in n+1..n+m) (id[i]=0)::domain;
       constraint forall(i in 1..n) (if p[i]>n then id[i] = D[n+1,i] else id[i] = id[p[i]]+D[p[i],i] endif)::domain;
       constraint forall(i in n+m+1..n+(2*m)) (if p[i]>n then id[i] = 0 else id[i] = id[p[i]]+D[p[i],n+1] endif)::domain;
       } in id; %function computing distance traveled at each object
       
var int: intermediate_distance = max (i in n+m+1..n+(2*m)) (interm_dist(p,D)); %variable to minimize: max of distances traveled (only endpoints are considered)

solve minimize intermediate_distance;


