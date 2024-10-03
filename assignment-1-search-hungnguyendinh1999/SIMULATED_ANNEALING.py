import sys
import numpy as np
from BFS import BFS
from ASTAR import ASTAR
from HILLDESCENT import energyfunction
import math

def annealing_prob_arr(current_energy, new_energy, temperature):
	# annealing (biased coin flip with annealing variable) for [0, 1] 
	# with probabilities (respectively) [1-prob, prob]
	# where prob = e^(best_energy - new energy)/T
	probability = pow(math.e, (current_energy - new_energy) / temperature)
	biased_coin_flip = np.random.choice(2, p=[1-probability, probability])

	return biased_coin_flip

def SIMULATED_ANNEALING(maze, start_cell, goal_state, iterations, T, decay):

	'''
	Fill in this function to implement Simulated Annealing.

	The energy function is the same as used for Hill Descent
	and is already imported here for it to be used directly
	(see the energyfunction() function in HILLDESCENT.py).

	With an input temperature 'T' and a decay rate 'decay',
	you should run the algorithm for 'iterations' steps.

	At each step, you should randomly select a valid move,
	and move to that state with probability 1 if the energy
	of the new state is less than the energy of the current state,
	or with probability exp((current_energy - new_energy)/T)
	if the energy of the new state is greater than the current energy.

	After each step, decrease the temperature by 
	multiplying it by the decay rate.

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
	temperature = T

	for _ in range(iterations):
		# pick random non-goal state
		random_local = (np.random.randint(0, n-1), np.random.randint(0, n-1))
		while random_local == goal_state:
			random_local = (np.random.randint(0, n-1), np.random.randint(0, n-1))

		# change its jump value to a random value
		new_maze = best_maze.copy()
		new_maze[random_local] = np.random.randint(1, n-1)
		new_energy = energyfunction(new_maze, start_cell, goal_state)
		
		# check for valid maze (solvable)
		if (new_energy is not None):

			if ((new_energy < best_energy) or 
			((new_energy > best_energy) and annealing_prob_arr(best_energy, new_energy, temperature))):
				best_maze = new_maze
				best_energy = new_energy
		
		# At the end, decrease the temperature by multiplying it with the decay constant
		temperature = temperature * decay
	
	best_solution = (best_maze, best_energy) if best_energy < input_maze_energy else (maze.copy(), input_maze_energy)

	return best_solution
