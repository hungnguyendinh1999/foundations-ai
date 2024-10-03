import numpy as np
import sys
import time
from ASTAR import ASTAR, H_score
from BFS import BFS
from GENERATOR import generator, generator_pathcheck
from HILLDESCENT import HILLDESCENT, HILLDESCENT_RANDOM_RESTART, HILLDESCENT_RANDOM_UPHILL, energyfunction
from SIMULATED_ANNEALING import SIMULATED_ANNEALING


# Set maze dimension.
k = 10

# maze, start_state, goal_state = generator(int(k))


'''
If mazes generated with the above line throw errors in future steps, it may be due to
there being no path between start and goal states. After implementing the 
BFS algorithm and completing the generator_pathcheck function in the GENERATOR.py file,
you can use the following line to generate a maze with a guaranteed path between
start and goal states by uncommenting it.
'''
# maze, start_state, goal_state = generator_pathcheck(int(k))

maze, start_state, goal_state = np.array([[3, 3, 1, 2],
       [0, 3, 3, 2],
       [1, 3, 2, 3],
       [2, 3, 2, 1]]), (0, 2), (1, 0)

'''
Ensure the following line prints a 2-D numpy array, and two tuples with two integers each.
Example output:
[[1 3 1 3 3]
 [1 1 0 1 2]
 [2 4 3 3 1]
 [3 2 2 4 1]
 [4 4 4 1 2]] (0, 0) (1, 2)
'''
print(maze, start_state, goal_state) 

# Check if start state is the same as goal state.
print(start_state==goal_state)


'''
Ensure the following lines print a 2-dimensional numpy array with 0 in the start state
and -1 for any state which is not reachable from the start state. Example output:
[[ 2  1  0  1  2]
 [ 3  4  1 -1  6]
 [ 4  5  5  6  5]
 [-1  2 -1  2  3]
 [ 5 -1  5 -1  4]]
'''
path_matrix=BFS(maze, start_state)
print(path_matrix) 


'''
Ensure that the following lines print the length of the shortest path between start 
and goal, as well as the complete shortest path, contained in a tuple. Example output:
(1, ((0, 2), (1, 2)))
'''
# shortest_path = ASTAR(maze,start_state,goal_state)
# print(shortest_path)


'''
Ensure that the following lines print whether the energy of the original maze is 
greater than or equal to the energy of the maze after hill descent and its variants.

Example output:
True
True
True
True

If your methods work correctly, these outputs should always be True, since if no 
better solution is found, the original maze is returned.
'''
# start_time = time.time()
energy = energyfunction(maze, start_state, goal_state)
print("Energy", energy)
# print("Energy(maze) time:\t", time.time()-start_time)
print("HILLDESCENT:",HILLDESCENT(maze,start_state,goal_state,3)[1])
# print("HILLDESCENT(maze) time:\t", time.time()-start_time)
# print(HILLDESCENT_RANDOM_RESTART(maze,start_state,goal_state,100,5)[1])
# print("HILLDESCENT_RANDOM_RESTART(maze) time:\t", time.time()-start_time)
# print(energy >= HILLDESCENT_RANDOM_UPHILL(maze,start_state,goal_state,00,0.03)[1])
# print("HILLDESCENT_RANDOM_UPHILL(maze) time:\t", time.time()-start_time)
# print(energy >= SIMULATED_ANNEALING(maze,start_state,goal_state,100,1000,0.98)[1])
# print("SIMULATED_ANNEALING(maze) time:\t", time.time()-start_time)











