include "globals.mzn";
include "fzn_at_least_int.mzn";


%INPUT VARIABLES 

int: n; %number of items
int: m; %number of couriers
array[1..n] of int: s; %array of items' sizes
array[1..m] of int: l; %array of couriers' capacities
array[1..n+1,1..n+1] of int: D; %matrix of the distances between items


%DEFINED VARIABLES 

array[1..n+(2*m)] of var 1..n+(2*m): pred; %array of predecessors of each item, starting point and ending point

array[1..n+(2*m)] of var 1..m: cour; %courier visiting each item, starting point and ending point location

array[1..n+(2*m)] of var 0..max(i in 1..m)(l[i]) :q; %total size of the items carried by courier on arrival at location i


%CONSTRAINTS

%predecessor defining constraints

constraint forall(i in 1..m) (pred[n+i]=n+m+i)::domain;  %predecessors of starting points

constraint alldifferent(pred)::domain_propagation; %all different predecessors

constraint forall(i,j in 1..m, k,z in 1..m) (if (i < j /\ l[i] = l[j] /\ pred[k] = n+i /\ pred[z] = n+j) then k < z endif) :: domain; %SYMMETRY BREAKING CONSTRAINT 1: if two couriers have the same capacity, force the first courier of the two to carry the item with smaller number


%constraints on vehicle

int: one = 1;

constraint forall(i in 1..m)(fzn_at_least_int(one, cour[1..n],i)); %each courier must carry at least one item

constraint forall(i in 1..m) (
                    (cour[n+i]=cour[n+m+i]) /\ (cour[n+i] = i))::domain; %courier i starts from point n+i and ends at point n+m+i

constraint forall(i in 1..n) (
                    cour[pred[i]]=cour[i])::domain; %item i and its predecessor are carried by the same courier (definition for items)
                    
constraint forall(i in n+m+1..n+(2*m)) (
                    cour[pred[i]]=cour[i])::domain; %item i and its predecessor are carried by the same courier (definition for ending points)
                    
                    
%constraints on capacity   
             
constraint forall(i in n+1..n+m) (
                    q[i]=0)::domain; %couriers are empty at starting points

constraint forall(i in 1..n) (
                    q[i]=q[pred[i]]+s[i])::domain; %iterative calculation of the total size of items carried on arrival at location i
                    
constraint forall(i in n+m+1..n+(2*m)) (
                    q[i]=q[pred[i]])::domain; %at ending points, the courier does not change the total size of items carried

constraint forall(i in 1..n+(2*m)) (
                    q[i]<=l[cour[i]] ); %courier i cannot carry more than his capacity


%COST FUNCTION

int: up_bound=D[n+1,1] + sum(i in 1..n)(D[i,i+1]); %upper bound

int: low_bound=min(i in 1..n)(D[n+1,i]+D[i,n+1]); %lower bound

array[1..n+(2*m)] of var 0..up_bound: dists; %array of distance traveled by the relative courier on arrival at location i

constraint forall(i in n+1..n+m) (
                    dists[i]=0)::domain; %the distance traveled at starting points is 0


constraint forall(i in 1..n) (
                    if pred[i]>n then dists[i]=D[n+1,i] else dists[i]=dists[pred[i]]+D[pred[i],i]           
                    endif)::domain; %iterative calculation of the total distance traveled by the courier on arrival at location i
                    
constraint forall(i in n+m+1..n+(2*m)) (
                    if pred[i]>n then dists[i]=0 else dists[i]=dists[pred[i]]+D[pred[i],n+1] endif
                    )::domain; %final step of calculation of the total distance traveled by the courier on arrival at location i
                    
          
%OBJECTIVE FUNCTION

var low_bound..up_bound: distance = max (i in n+m+1..n+(2*m)) (dists[i]);

solve  
 :: int_search(pred, smallest, indomain_min)
 :: int_search(cour, smallest, indomain_min)
 :: restart_luby(250)
minimize distance;

output["pred = \(pred)\n"];
output["cour = \(cour)\n"];
output["distance = \(distance)\n"];
