import numpy as np
from VALID_MOVES import valid_moves

def isAvailable(maze, node):
    return maze[node] == -1

def BFS(maze, start):
    
    '''
    Fill in this function that uses Breadth First Search to find the shortest path 
    from the start state to the goal state.
    
    Return the matrix (a 2-dimensional numpy array) of shortest path 
    distances from the start cell to each cell. 
    
    If no path exists from the start state to a given cell, that cell should be assigned -1.
    
    The start state should be assigned a path length of 0.
    
    If using print statements to debug, please make sure 
    to remove them before your final submisison.
    '''
    # init matrix and start and queue
    path_matrix = np.full(shape=(len(maze),len(maze)), fill_value=-1, dtype=int)
    path_matrix[start] = 0
    queue = [start]

    while queue:
        curr_node = queue.pop(0)
        
        # traverse through neighbors
        neighbors = valid_moves(maze, curr_node)
        for n in neighbors:
            if isAvailable(path_matrix, n):
                queue.append(n)
                # do stuff to it
                path_matrix[n] = path_matrix[curr_node] + 1

    return path_matrix
