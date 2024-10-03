def check_inbound(num, k):
	return num >= 0 and num < k

def valid_moves(maze, node):
	'''
	Fill in this function to return a list of "valid" neighbors 
	for the current node in the rook-jumping-maze.
	
	If using print statements to debug, please make sure 
	to remove them before your final submisison.
	'''
	neighbors = []
	size = len(maze)
	value = maze[node]
	potential_moves = [
		(node[0]-value,node[1]), 
		(node[0]+value,node[1]), 
		(node[0], node[1]-value), 
		(node[0], node[1]+value)]
	
	for x,y in potential_moves:
		if check_inbound(x, size) and check_inbound(y, size):
			neighbors.append((x,y))

	return neighbors
