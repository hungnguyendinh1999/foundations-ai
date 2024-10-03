import numpy as np
from tqdm import tqdm
import gymnasium as gym
import matrix_mdp
import sys
import matplotlib.pyplot as plt

nS = 16
nA = 4

slip_prob = 0.1

actions = ['up', 'down', 'left', 'right']

p_0 = np.array([0 for _ in range(nS)])
p_0[12] = 1

P = np.zeros((nS,nS,nA), dtype=float)

def valid_neighbors(i,j):
	neighbors = {}
	if i>0:
		neighbors[0]=(i-1,j)
	if i<3:
		neighbors[1]=(i+1,j)
	if j>0:
		neighbors[2]=(i,j-1)
	if j<3:
		neighbors[3]=(i,j+1)
	return neighbors

for i in range(4):
	for j in range(4):
		if i==0 and j==2:
			continue
		if i==3 and j==1:
			continue

		neighbors = valid_neighbors(i,j)
		for a in range(nA):
			if a in neighbors:
				P[neighbors[a][0]*4+neighbors[a][1], i*4+j, a] = 1-slip_prob
				for b in neighbors:
					if b != a:
						P[neighbors[b][0]*4+neighbors[b][1], i*4+j, a] = slip_prob/float(len(neighbors.items())-1)

R = np.zeros((nS, nS, nA))

R[2,1,3] = 2000
R[2,3,2] = 2000
R[2,6,0] = 2000

R[13,9,1] = 2
R[13,14,2] = 2
R[13,12,3] = 2

R[11,15,0] = -100
R[11,7,1] = -100
R[11,10,3] = -100
R[10,14,0] = -100
R[10,6,1] = -100
R[10,11,2] = -100
R[10,9,3] = -100
R[9,10,2] = -100
R[9,13,0] = -100
R[9,5,1] = -100
R[9,8,3] = -100

env=gym.make('matrix_mdp/MatrixMDP-v0', p_0=p_0, p=P, r=R)


#################################################################
# Helper Functions
#################################################################

#reverse map observations in 0-15 to (i,j)
def reverse_map(observation):
	return observation//4, observation%4

#################################################################
# Q-Learning
#################################################################

'''
In this section, you will implement Q-learning with epsilon-greedy exploration.
The Q-table is initialized to all zeros. The value of eta should be updated as 1/(1 + number of updates) inside the loop.
The value of epsilon should be decayed to (0.9999 * epsilon) each time inside the loop.

Refer to the written assignment for the update equation. Similar to MDPs, use the following code to take an action:

observation, reward, terminated, truncated, info = env.step(action)

Unlike MDPs, your action is now chosen by the epsilon-greedy policy. The action is chosen as follows:

With probability epsilon, choose a random action.
With probability (1 - epsilon), choose the action that maximizes the Q-value (based on the last estimate). 
In case of ties, choose the action with the smallest index.
In case the chosen action is not a legal move, generate a random legal action.

The episode terminates when the agent reaches one of many terminal states. 

After 10, 100, 1000 and 10000 episodes, plot a heatmap of V_opt(s) for all states s. This should be a 4x4 grid, corresponding to our map of Mordor.
Please use plt.savefig() to save the plot, and do not use plt.show().
Add each heatmap (clearly labeled) to your answer to Q9 in the written submission.

'''
def plot_heatmap(V, name):
	# V must be a 4x4 matrix
	plt.imshow(V, cmap='PuBu')
	for y in range(V.shape[0]):
		for x in range(V.shape[1]):
			plt.text(x, y, '%.3f' % V[y, x],
				horizontalalignment='center',
				verticalalignment='center',
				)
	plt.colorbar()
	plt.grid(True)
	plt.axis('off')
	plt.title(name)
	plt.savefig(name)

Q = np.zeros((nS, nA))

gamma = 0.9
epsilon = 0.9
eta = 1

observation, info = env.reset()

for _ in tqdm(range(10000)):
	while True:
		# Choose an action based on the epsilon-greedy strategy
		choice_pi = np.random.choice([False, True], p=[1-epsilon, epsilon])
		action = -1
		i, j = reverse_map(observation)
		neighbors = valid_neighbors(i, j) # [UP, DOWN, LEFT, RIGHT] == [0, 1, 2, 3]
		while action not in neighbors: #Ensure that chosen move is legal
			if choice_pi:
				action = np.random.choice(list(neighbors.keys()))
			else:
				action = np.argmax(Q[observation,:])
				choice_pi = not choice_pi # make sure it picks random legal move instead
				
		prev_observation = observation
		observation, reward, terminated, truncated, info = env.step(action)
		# use obtained reward and new state to update Q-table 
		Q[prev_observation, action] = (1 - eta)* Q[prev_observation, action] + eta * (reward + gamma*(np.max(Q[observation,:])) )
		
		if terminated or truncated: break
		epsilon = (0.9999 * epsilon)
	# update eta, epsilon per code comment
	eta = 1/(1 + _)

	# record data for heatmap
	if (_+1 in [10, 100, 1000, 10**4]):
		V_opt = np.amax(Q, axis=1).reshape((4,4)) # 16 numbers (maximums)
		plt.figure(_+1)
		plot_heatmap(V_opt, f"heatmap_{_+1}")

	observation, info = env.reset()

#save Q table
np.savetxt('qtable.txt', Q, fmt='%f')

policy = np.array([actions[i] for i in np.argmax(Q, axis=1)])
print(policy.reshape(4,4))