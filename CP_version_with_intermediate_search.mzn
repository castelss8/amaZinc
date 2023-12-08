%Version with shorter p and d arrays

include "globals.mzn";

% a. ARRAYS FOR THE FIRST PART OF SEARCH - NON OPTIMAL SOLUTIONS

int: n; %number of items
int: m; %number of couriers

array[1..n] of int: s; %array of items' sizes

%array[1..n+1] of int: s2= s++[0];

array[1..m] of int: l; %array of couriers' capacities

array[1..n+1,1..n+1] of int: D; %Matrix of the distances

array[1..n] of var 1..m: v; %array describing which courier carries each object

array[1..n,1..m] of var bool: H; %Matrix describing wether object i is carried or not by vehicle j



% a. CONSTRAINTS:


constraint forall(i in 1..n) (
                    s[i]<=l[v[i]] - (sum(j in 1..i-1 where v[i]=v[j])(s[j])))::domain_propagation;
                    
%alternative version: 
% constraint forall(i in 1..n) (
%                    s[i]<=l[v[i]] - (sum(j in 1..i-1)(s[j]*boolToInt(H[v[i],j]))));



% a. FUNCTION TO BE MINIMIZED


array[1..m] of var (min(i in 1..n+1, j in 1..n+1)(D[i,j])..(sum(i in 1..n+1,j in 1..n+1)(D[i,j]))): k; %array of distance run by each courier

constraint forall(i in 1..m) (k[i]=(D[(min(o in 1..n)(i=v[o])),n+1])+sum(j in (min(o in 1..n)(i=v[o]))+1..n where i=v[j], z in 1..n where z=max(u in 1..j-1)(i==v[u]))(D[z,j]))::domain_propagation; 

%alternative version for this constraint: same as last alternative version

solve minimize sum(i in 1..m)(k[i]);
