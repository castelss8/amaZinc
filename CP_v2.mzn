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
                    p[n+i]=n+m+i);  %predecessors of starting points are corresponding ending point
constraint forall(i in 1..m) (
                    d[n+m+i]=n+i); %successors of ending point
                    
                    
                    
constraint alldifferent(p)::domain_propagation; %all different predecessors
constraint alldifferent(d)::domain_propagation; %all different successors



% domain_propagation??

%coherence constraints:
constraint forall(i in 1..n) (
                    d[p[i]]=i); %successor of the predecessor of i is i itself (for items)
constraint forall(i in n+m+1..n+(2*m)) (
                    d[p[i]]=i); % same (for ending points)
                    
constraint forall(i in 1..n+m) (
                    p[d[i]]=i);  
                    
                    
                    

%Vehicles constraints:
constraint forall(i in 1..m) (
                    ( v[n+i]=v[n+m+i] )); % /\ ( v[n+m+i] = i ) ??? %couriers for each starting and ending point

constraint forall(i in 1..m) (           
                    ( v[n+i] = i ) )
                    
constraint forall(i in 1..n) (
                    v[p[i]]=v[i]);
constraint forall(i in n+m+1..n+(2*m)) (
                    v[p[i]]=v[i]);
constraint forall(i in 1..n+m) (
                    v[d[i]]=v[i]); %routes constraints



%Capacity constraints:
constraint forall(i in n+1..n+m) (
                    q[i]=0);

constraint forall(i in 1..n) (
                    q[i]=q[p[i]]+s[i]);
constraint forall(i in n+m+1..n+(2*m)) (
                    q[i]=q[p[i]]);
                     
                    
constraint forall(i in 1..n) (
                    if d[i]<=n then q[i]=q[d[i]]-s[d[i]] else q[i] = q[d[i]] endif ); %"quantity of goods" computation + BOUND CONSISTENCY??


constraint forall(i in 1..n+(2*m)) (
                    q[i]<=l[v[i]] );


%COST FUNCTION

%dists
array[1..n+(2*m)] of var 0..(sum(i,j in 1..n)(D[i,j])): dists; %upper limit not optimal!

constraint forall(i in n+1..n+m) (
                    dists[i]=0);




%!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
%Computation of dists
constraint forall(i in 1..n) (
                    if p[i]>n then dists[i]=D[n+1,i] else dists[i]=dists[p[i]]+D[p[i],i] endif);
                    
constraint forall(i in n+m+1..n+(2*m)) (
                    if p[i]>n then dists[i]=0 else dists[i]=dists[p[i]]+D[p[i],n+1] endif
                    );
                    
constraint forall(i in 1..n) (
                    if d[i]>n then dists[i]=dists[d[i]]-D[i,n+1] else dists[i]=dists[d[i]]-D[i,d[i]] endif ); %starting points should be considered here??

constraint forall(i in n+1..n+m) (
                    if d[i]<=n then dists[d[i]]-D[n+1,d[i]]=0 else true endif
                    );




var int: distance = max (i in n+m+1..n+(2*m)) ( dists[i] );

solve minimize distance ;
