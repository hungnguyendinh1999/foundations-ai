import numpy as np
import sys

sys.path.append('./')
sys.path.append('../')

from BFS import BFS
from GENERATOR import generator, generator_pathcheck


'''
MAKE ABSOLUTELY NO CHANGES TO THIS FILE OR YOUR CODE WILL FAIL THE AUTOGRADER.

Evaluates random maze generation, and validates formatting of the output matrix from BFS.
'''

for k in range(10,101,10):
	maze, start_state, goal_state = generator(int(k))
	print(maze.shape, len(start_state), len(goal_state), start_state==goal_state, np.sum(maze==0), k/2-2 <= np.mean(maze) <= k/2+2)
	path_matrix=BFS(maze, start_state)
	print(path_matrix[start_state], path_matrix[goal_state].dtype)

