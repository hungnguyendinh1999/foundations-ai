import heapq
import numpy as np
from VALID_MOVES import valid_moves

def H_score(node, goal, n):
	'''
	Fill in this function to return the heuristic value of the current node.

	Compute heuristic as the Manhattan distance between the 
	current node and the goal state, divided by 
	the largest possible jump value.

	n is the dimensionality of the maze (n x n).

	If using print statements to debug, please make sure 
	to remove them before your final submisison.
	'''
	nodeX, nodeY = node
	goalX, goalY = goal

	# Manhattan distance
	heuristic = ( abs(nodeX - goalX) + abs(nodeY - goalY) ) / (n-1)

	return heuristic


def ASTAR(maze, start, goal):
	'''
	Fill in this function that uses A* search to find the shortest 
	path using the heuristic function H_score defined above.

	Return the length of the shortest path from the start state 
	to the goal state, and the path itself.

	Your return statement should be of the form:
	return len(path)-1, path

	where path is a list of tuples, corresponding to the 
	path and includes the start state.

	If using print statements to debug, please make sure 
	to remove them before your final submisison.
	'''
	
	n = len(maze) # dimension n x n
	pqueue=[]
	closed_set = set()

	heapq.heappush(pqueue, (H_score(start, goal, n), 0, (start, ())))
	while pqueue:
		heuristic, cost, (curr_node, path) = heapq.heappop(pqueue)
		# print("PQ.pop() :\n|| heuristic = {0}\n||cost = {1}\n||node & path = {2}".format(heuristic, cost, (curr_node, path)))
		path = path + (curr_node,)
		if curr_node == goal:
			# print("Goal Reached!")
			return len(path)-1, path
		for neighbour in valid_moves(maze, curr_node):
			if neighbour in closed_set:
				continue
			G_score = cost + 1
			item = (G_score + H_score(neighbour, goal, n), G_score, (neighbour, path))
			# print("PQ.push() : \t", item)
			heapq.heappush(pqueue, item)

	# default to (-1, []) if no path is found
	return -1, []

