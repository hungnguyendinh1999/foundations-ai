import numpy as np
import sys

sys.path.append('./')
sys.path.append('../')


from BFS import BFS
from GENERATOR import generator, generator_pathcheck

'''
MAKE ABSOLUTELY NO CHANGES TO THIS FILE OR YOUR CODE WILL FAIL THE AUTOGRADER.

Evaluates BFS against test inputs.
'''

lines = list(sys.stdin)

i=0

while i<len(lines):
	k=int(lines[i].rstrip('\n'))
	maze=np.zeros((k,k),dtype=int)
	for j in range(i+1,i+k+1):
		maze[j-i-1]=np.fromstring(lines[j].rstrip('\n'), sep=",", dtype=int)
	start=tuple(int(x) for x in lines[i+k+1].rstrip('\n').split(','))
	goal=tuple(int(x) for x in lines[i+k+2].rstrip('\n').split(','))
	i=i+k+3
	
	path_matrix = BFS(maze, start)
	print(path_matrix)

