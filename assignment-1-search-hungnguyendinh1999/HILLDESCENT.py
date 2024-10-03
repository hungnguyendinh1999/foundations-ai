import sys
import numpy as np
from BFS import BFS
from ASTAR import ASTAR


def energyfunction(maze, start, goal):
	'''
	Compute the energy as the sum of the shortest path length 
	from the start state to the goal state (computed using A*)
	and the number of cells that are not reachable from the 
	start state (computed using BFS).

	If using print statements to debug, please make sure
	to remove them before your final submisison.
	'''
	path_matrix = BFS(maze, start)
	if path_matrix[goal] == -1:
		return None
	
	shortest_path = ASTAR(maze, start, goal)

	count_nonreachable = np.count_nonzero(path_matrix == -1)
	energy = shortest_path[0] + count_nonreachable
	
	return energy



def HILLDESCENT(maze, start_cell, goal_state, iterations):
	'''
	Fill in this function to implement Hill Descent local search.

	Your function should return the best solution found, 
	which should be a tuple containing 2 elements:

	1. The best maze found, which is a 2-dimensional numpy array.
	2. The energy of the best maze found.

	Note that you should make a local copy of the maze 
	before making any changes to it.

	If using print statements to debug, please make sure
	to remove them before your final submisison.
	'''
	# Initialize
	best_energy = energyfunction(maze, start_cell, goal_state)
	best_maze = maze
	n = len(maze)
	for _ in range(iterations):
		# pick random non-goal state
		random_local = goal_state
		while random_local == goal_state:
			random_local = (np.random.randint(0, n-1), np.random.randint(0, n-1))
		print("Changing random: ", random_local)
		# change its jump value to a random value
		new_maze = best_maze.copy()
		new_maze[random_local] = np.random.randint(1, n-1)
		new_energy = energyfunction(new_maze, start_cell, goal_state)

		# new maze CAN be unsolvable
		print("new_energy for ", new_maze[random_local], "is ",new_energy)
		if (new_energy is not None) and (new_energy < best_energy):
			best_maze = new_maze
			best_energy = new_energy
	
	best_solution = (best_maze, best_energy)

	return best_solution



def HILLDESCENT_RANDOM_RESTART(maze, start_cell, goal_state, iterations, num_searches):
	'''
	Fill in this function to implement Hill Descent local search with Random Restarts.

	For a given number of searches (num_searches), run hill descent search.

	Keep track of the best solution through all restarts, and return that.

	Your function should return the best solution found, 
	which should be a tuple containing 2 elements:

	1. The best maze found, which is a 2-dimensional numpy array.
	2. The energy of the best maze found.

	Note that you should make a local copy of the maze 
	before making any changes to it.

	You will also need to keep a separate copy of the original maze
	to use when restarting the algorithm each time.

	If using print statements to debug, please make sure
	to remove them before your final submisison.
	'''
	best_maze, best_energy = maze, None

	for _ in range(num_searches):
		descended_maze, descended_energy = HILLDESCENT(maze, start_cell, goal_state, iterations)
		if best_energy is None or descended_energy < best_energy:
			best_maze = descended_maze
			best_energy = descended_energy

	best_solution = (best_maze, best_energy)

	return best_solution



def HILLDESCENT_RANDOM_UPHILL(maze, start_cell, goal_state, iterations, probability):
	'''
	Fill in this function to implement Hill Descent local search with Random uphill steps.

	At each iteration, with probability specified by the probability
	argument, allow the algorithm to move to a worse state.

	Your function should return the best solution found, 
	which should be a tuple containing 2 elements:

	1. The best maze found, which is a 2-dimensional numpy array.
	2. The energy of the best maze found.

	Note that you should make a local copy of the maze
	before making any changes to it.

	If using print statements to debug, please make sure
	to remove them before your final submisison.
	'''
	# Initialize
	input_maze_energy = energyfunction(maze, start_cell, goal_state)

	best_energy = input_maze_energy
	best_maze = maze
	n = len(maze)

	if best_energy is None:
		raise Exception("Input Maze is not solvable")

	for _ in range(iterations):
		# pick random non-goal state
		random_local = (np.random.randint(0, n-1), np.random.randint(0, n-1))
		while random_local == goal_state:
			random_local = (np.random.randint(0, n-1), np.random.randint(0, n-1))

		# change its jump value to a random value
		new_maze = best_maze.copy()
		new_maze[random_local] = np.random.randint(1, n-1)
		new_energy = energyfunction(new_maze, start_cell, goal_state) # beware of None case
		# biased coin flip for [0, 1] with probabilities (respectively) [1-prob, prob]
		# e.g prob = 0.05,then prob for 0 = 0.95, prob for 1 = 0.05

		
		if (new_energy is not None):
			bias_coin_flip = np.random.choice(2, p=[1-probability, probability])
			if ((new_energy < best_energy) or 
			((new_energy > best_energy) and bias_coin_flip)):
				best_maze = new_maze
				best_energy = new_energy
	
	best_solution = (best_maze, best_energy) if best_energy < input_maze_energy else (maze.copy(), input_maze_energy)

	return best_solution







