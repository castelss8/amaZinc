# amaZinc
Combinatorial and Decision Making Optimization - Project Work
<br> 
---
The problem presented is the Multiple Couriers Programming.<br>
Different strategies have been used:
- Constraint Programming with MiniZinc
- SAT-Solvers with Z3
- SMT-Solvers with Z3
- Mixed Integer Programming with MIP (Gurobi)
---
To run the project with docker:
- ```docker build -t castelss .``` to build the docker image.
- ```docker run -t castelss``` to run the docker image.

Then on the bash terminal it is possible to run the script with the command:
- ```python main_solutions.py``` to run a command line interface and choose the solver and the instance to solve.

There are eight different solvers to choose

---
By:
- Davide Bombardi davide.bombardi@studio.unibo.it
- Giorgia Castelli giorgia.castelli2@studio.unibo.it
- Alice Fratini alice.fratini2@studio.unibo.it
- Madalina Ionela Mone madalina.mone@studio.unibo.it
