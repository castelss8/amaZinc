import matplotlib as mplt
import numpy as np
import pandas as pd
from sympy import symbols, Eq, solve, sqrt
import matplotlib.pyplot as plt



#read istances
f_name = "Instances/inst16"
with open(f_name + ".dat", "r") as f:
    lines = f.readlines()

#variables
m = int(lines[0]) #number of couriers
n = int(lines[1]) #number of items
l = [int(x) for x in lines[2].split()] #%array of couriers' capacities
s = [int(x) for x in lines[3].split()] #%array of items' sizes
D = [[int(x) for x in line.split()] for line in lines[4:]] #%matrix of distances
#lower_bound = prim(D) #lower bound
#this is the origin point where each currier will start
x0, y0 = 0 , 0
P0 = [x0, y0]
#assign another arbitrary point from D matrix, it will be placed on te x-axis
#the other coordinate is caculate using the distance between the origin point and the first pack

D_P0_P1 = D[0][n]
x1 = sqrt(D_P0_P1**2)
y1 =0
P1 =[x1,y1]
P = [P0, P1]


#calculate third point using O and P1
for i in range(1, n):
    D_P1_Pi = D[0][i]
    D_P0_Pi = D[n][i]

    x,y = symbols('x y')
    eq1 = Eq(sqrt((x - x1)**2 + (y - y1)**2), D_P1_Pi)
    eq2 = Eq(sqrt(x1**2 + y1**2), D_P0_P1)
    eq3 = Eq(sqrt(x**2 + y**2), D_P0_Pi)

    solution = solve((eq1, eq2, eq3), (x, y))

    x,y= solution[0]
    #print(x,y)
    Pi = [float(x), float(y)]
    print(Pi)
    P.append(Pi)


#print(P)

#represent the points in a cartesian plane

fig, ax = plt.subplots()
#plt.xlim(-max(l), max(l))
#plt.ylim(-max(l), max(l))
plt.grid()
plt.axhline(0, color='black')
plt.axvline(0, color='black')
ax.plot(x0, y0, 'bo')
ax.plot(x1, y1, 'ro')
for i in range(2, n):
    ax.plot(P[i][0], P[i][1], 'go')
plt.show()



#CONNECTING POINTS
#connect the points with lines
fig, ax = plt.subplots()
#plt.grid()
#plt.axhline(0, color='black')
#plt.axvline(0, color='black')
ax.plot(x0, y0, 'bo')
ax.plot(x1, y1, 'ro')
for i in range(2, n):
    ax.plot(P[i][0], P[i][1], 'yo')
    ax.plot([x0, P[i][0]], [y0, P[i][1]], 'b-')
plt.show()
