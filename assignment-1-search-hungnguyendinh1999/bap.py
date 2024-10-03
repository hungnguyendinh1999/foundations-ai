import numpy as np
import json

with open(".github/classroom/autograding.json") as f:
    data = json.load(f)

data = data['tests'][1]
in_matrix = data['input']
out_matrix = data['output']

in_s = in_matrix[:69]
print(in_s)
out_s = out_matrix[:out_matrix.index(']]')+2]
print(out_s)

"""
generate()
5
[[ 1  3  1  3  3]
 [ 1  1  0  1  2]
 [ 2  4  3  3  1]
 [ 3  2  2  4  1]
 [ 4  4  4  1  2]]
0,2
1,2

BFS()
[[ 2  1  0  1  2]
 [ 3  4  1 -1  6]
 [ 4  5  5  6  5]
 [-1  2 -1  2  3]
 [ 5 -1  5 -1  4]]
"""

def ASTAR(maze, start, goal):
	open_list = []
	nearest_parent = {} # closed_list + path tracker
	cost_so_far = {} # closed_list
	nearest_parent[start] = None
	cost_so_far[start] = 0

	path = []
	# a node is coordinate tuple (x, y). E.g. start & goal
	# heap stores (cost, node) => (0, start)
     
	while open_list:
		cost, current_node = heapq.heappop(open_list)
          
		if current_node == goal:
            # Goal reached, construct and return the path
			path.append(goal)
			while current_node != start:
				# backtrack to solution
				current_node = nearest_parent[current_node]
				path.append(current_node)
			
			return len(path)-1, path[::-1] # reverse the array
		
		for neighbor in valid_moves(maze, current_node):
			if neighbor in closed_list:
				continue
			
			new_cost = cost + 1
			if neighbor not in open_list:
				heapq.heappush(open_list, (new_cost + H_score(neighbor, goal, len(maze)), neighbor))
			elif new_cost < neighbor.cost:
				neighbor.cost = new_cost
				neighbor.parent = current_node