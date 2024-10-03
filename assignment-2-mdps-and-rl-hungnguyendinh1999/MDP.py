import numpy as np
from tqdm import tqdm
import gymnasium as gym
import matrix_mdp
import sys
import matplotlib.pyplot as plt

#######################################
# 1. Initialize transition and reward matrices
# 2. Fill transition and reward matrices with correct values
#######################################

num_states = 5
num_actions = 4

# Create transition and reward matrices below:

T = np.zeros(shape=(5,5,4)) # Transition matrix

R = np.zeros(shape=(5,5,4)) # Reward matrix


# Set the entries in T and R below as per the MDP used in the assignment diagram:
# triplets are (s', s, a)
p = q = s = 0.5
r = 1
# S1 = 0, A1 = 0
T[(0, 0, 0)] = p
T[(1, 0, 0)] = 1-p
# S1 = 0, A2 = 1
T[(2, 0, 1)] = q
T[(1, 0, 1)] = 1-q
# S2 = 1, A3 = 2 => S3 = 2; S4 = 3, S5= 4
T[(2, 1, 2)] = r/3
T[(3, 1, 2)] = r/3
T[(4, 1, 2)] = r/3
# S3 = 2, A4 = 3 => S1 = 0, S5= 4
T[(0, 2, 3)] = 1-s
T[(4, 2, 3)] = s

# Reward
R[(4, 1, 2)] = 10
R[(4, 2, 3)] = 10

#######################################
# 3. Map indices to action-labels 
#######################################

A = {  # map each index to an action-label, such as "A1", "A2", etc. 
    x: f"A{x+1}" for x in range(0,5)
}


#######################################
# Initialize the gymnasium environment
#######################################

# No change required in this section

P_0 = np.array([1, 0, 0, 0, 0])    # This is simply the initial probability distribution which denotes where your agent is, i.e. the start state.

env=gym.make('matrix_mdp/MatrixMDP-v0', p_0=P_0, p=T, r=R)


#######################################
# 4. Random exploration
#######################################

'''
First, we reset the environment and get the initial observation.
The observation tells you which state you are in - in this case, indices 0-4 map to states S1 - S5.
Since we set P_0 = [1, 0, 0, 0, 0], the initial state is always S1 after env.reset() is called.
'''
observation, info = env.reset()

'''
Below, write the code for random exploration, i.e. randomly choosing an action at each time-step and executing it.

A random action is simply a random integer between 0 and the number of actions (num_actions not inclusive).
However, you should make sure that the chosen action can actually be taken from the current state.
If it is not a legal move, generate a new random move.

Avoid hardcoding actions even for states where there is only one action available. That way, your
code is more general, and may be easily adapted to a different environment later.

You will use the following line of code to explore at each time step:

observation, reward, terminated, truncated, info = env.step(action)

The above line of code is used to take one step in the environment using the chosen action.
It takes as input the action chosen by the agent, and returns the next observation (i.e., state),
reward, whether the episode terminated (terminal states), whether the episode was 
truncated (max iterations reached), and additional information.

If at any point the episode is terminated (this happens when we reach a terminal state, 
and the env.step() function returns True for terminated), you should
end the episode in order to reset the environment, and start a new one.

Keep track of the total reward in each episode, and reset the environment when the episode terminates.

Print the average reward obtained over 10000 episodes. 

'''
total_reward = 0
episode_count = 10**4
for _ in tqdm(range(episode_count)):

    running_total = 0
    while True:
        legal_indices = T[:, observation, :].any(0).nonzero()[0] # produce a set of LEGAL actions, given state "observation"
        # a) Choose a random LEGAL action random.int
        action = np.random.choice(legal_indices) # random action
        # b) probabilistically move to the next state, and obtain some reward
        observation, reward, terminated, truncated, info = env.step(action)
        
        # c) add the current reward to the running total for this episode, 
        running_total += reward

        # d) check if you are at a terminal state (refer to code comments for how to do this), and if so, start a new episode.
        if terminated or truncated: break

    total_reward += running_total
    observation, info = env.reset() # reset

avg_reward = total_reward / episode_count

print("Average reward obtained: ", avg_reward)


#######################################
# 5. Policy evaluation 
# 6. Plotting V_pi(s)
#######################################

gamma = 0.9

'''
Initialize the value function V(s) = 0 for all states s.
Use the Bellman update to iteratively update the value function, given the following policy:

S1 -> A1, S2 -> A3, S3 -> A4

Plot the value of S1 over time (i.e., at each iteration of Bellman update).
Please use plt.savefig() to save the plot, and do not use plt.show().
Save and insert this plot into the written submission.

Print the value function of S1 and S2 after 100 iterations.

'''
# Set policy pi
policy = {
    0: 0, # "S1": "A1",
    1: 2, # "S2": "A3",
    2: 3  # "S3": "A4"
}

# Initialize V_pi(s) = 0
V = np.zeros(5)

def bellman_util(s_prime, s, a, V):
    return T[s_prime, s, a] * (R[s_prime, s, a] + gamma * V[s_prime] )

'''
UPDATE in-place V
@parameters:
V: array[5] of value at current time
iterations: number of iterations to perform
policy: policy pi for 
'''
def update_bellman_util(V, iterations=100, policy=policy):
    states = range(5)
    V_pi_S1_over_t = []
    for _ in tqdm(range(iterations)):
        newV = V.copy()
        for state in policy:       # for all states with mapping
            action = policy[state] # policy action
            newV[state] = sum([ bellman_util(s_prime, state, action, V) for s_prime in states])
        V = newV
        V_pi_S1_over_t.append((_, V[0]))
    return V, V_pi_S1_over_t

# iterative updates to value function for ALL states
V, plot_data = update_bellman_util(V, 100, policy)

def plot_and_save(data, plot_name):
    plt.plot([x[0] for x in data], [y[1] for y in data])
    plt.title(f"Graphing {plot_name}")
    plt.xlabel("t (steps)",fontsize='13')	#adds a label in the x axis
    plt.ylabel("Value V(S1)",fontsize='13')	#adds a label in the y axis
    plt.grid()
    plt.savefig(f"{plot_name}.png")

plt.figure(0)
plot_and_save(plot_data, "Utility V(S1) over time using Policy 1")

#######################################
# 7. Evaluating a Second Policy
#######################################

'''
Now change the policy to:

S1 -> A2, S2 -> A3, S3 -> A4

Re-run Bellman updates for all states.

Plot the value of S1 over time (i.e., at each iteration of Bellman update). 
Please use plt.savefig() to save the plot, and do not use plt.show().
Save and insert this plot into the written submission.

Print the value function of S1 and S2 after 100 iterations.
'''

print("Value of S1 and S2 after 100 iterations for first policy: ", V[0], V[1])

policy = {
    0: 1, # "S1": "A2",
    1: 2, # "S2": "A3",
    2: 3  # "S3": "A4"
}

# Initialize V_pi(s) = 0
V = np.zeros(5)

V, plot_data = update_bellman_util(V, 100, policy)
plt.figure(1)
plot_and_save(plot_data, "Utility V(S1) over time using Policy 2")

print("Value of S1 and S2 after 100 iterations for second policy: ", V[0], V[1])


#######################################
# 8. Value Iteration for Best Policy
# 9. Output Best Policy
#######################################

'''
Initialize the value function V(s) = 0 for all states s.

Use value iteration to find the optimal policy for the MDP.

Plot V_opt(S1) over time (i.e., at each iteration of Bellman update).
Please use plt.savefig() to save the plot, and do not use plt.show().
Save and insert this plot into the written submission.

Print the optimal policy after 100 iterations.

'''
def optimal_bellman_util(V, iterations=100):
    states = range(5)
    V_pi_S1_over_t = []
    for _ in tqdm(range(iterations)):
        newV = V.copy()
        for state in states:
            # get legal actions from state
            legal_actions = T[:, state, :].any(0).nonzero()[0]
            # calculate utility for each action to determine best ACTION
            if legal_actions.size > 0:
                d = [sum([ bellman_util(s_prime, state, action, V) for s_prime in states]) for action in legal_actions]
                newV[state] = max( d )
        V = newV
        V_pi_S1_over_t.append((_, V[0]))
    return V, V_pi_S1_over_t

V = np.zeros(5)
plt.figure(2)
V, plot_data = optimal_bellman_util(V, 100)
plot_and_save(plot_data, "Utility V(S1) over time using Optimal Policy")

# calculate \pi_opt given V_opt
opt_policy = {}

states = range(5)
for state in states:
    # get legal actions from state
    legal_actions = T[:, state, :].any(0).nonzero()[0]
    # calculate utility for each action to determine best ACTION
    if legal_actions.size > 0:
        d = [(sum([ bellman_util(s_prime, state, action, V) for s_prime in states]), action) for action in legal_actions]
        opt_policy[state] = max( d )[1]

policy = list(opt_policy.values())
print("Optimal policy: ", [A[i] for i in policy])