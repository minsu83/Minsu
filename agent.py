import numpy as np
from visualize_train import draw_value_image, draw_policy_image

# left, right, up, down
ACTIONS = [np.array([0, -1]),
           np.array([0, 1]),
           np.array([-1, 0]),
           np.array([1, 0])]


class AGENT:
    def __init__(self, env, is_upload=False):

        self.ACTIONS = ACTIONS
        self.env = env
        HEIGHT, WIDTH = env.size()
        self.state = [0,0]

        if is_upload:
            dp_results = np.load('./result/dp.npz')
            self.values = dp_results['V']
            self.policy = dp_results['PI']
        else:
            self.values = np.zeros((HEIGHT, WIDTH))
            self.policy = np.zeros((HEIGHT, WIDTH,len(self.ACTIONS)))+1./len(self.ACTIONS)




    def policy_evaluation(self, iter, env, policy, discount=1.0):
        HEIGHT, WIDTH = env.size()
        new_state_values = np.zeros((HEIGHT, WIDTH))
        iteration = 0

        #***************************************************
        while True:
            delta = 0                                   # Delta(maximum state values change) initialization
            old_state_value = self.values.copy()        # Copy old state values from self.values
            for i in range(HEIGHT):                     # For each state(height, width)
                for j in range(WIDTH):
                    state_value = 0                     # Current state value initialization
                    for k, action in enumerate(ACTIONS):# For each action at each state
                        # Get the next state & reward(s',r) from env.interaction using the current state & action(s,a)
                        (next_i, next_j), reward = env.interaction([i, j], action)
                        if not env.is_terminal([i, j]): # If current state is not terminal
                            # Update state values using the 'Bellman equation'
                            state_value += policy[i, j, k] * (reward + discount * old_state_value[next_i, next_j])
                    self.values[i, j] = state_value     # Copy updated state values to self.values
                    new_state_values[i, j] = state_value# Copy updated state values to new state values
                    # Delta is maximum state values change of all states
                    delta = max(delta, np.abs(old_state_value[i, j] - new_state_values[i, j]))

            if delta < 1e-3:                            # If delta is small enough(smaller than a small positive number)
                break                                   # Finish policy evaluation and goes to policy improvement
        #***************************************************

        draw_value_image(iter, np.round(new_state_values, decimals=2), env=env)
        return new_state_values, iteration



    def policy_improvement(self, iter, env, state_values, old_policy, discount=1.0):
        HEIGHT, WIDTH = env.size()
        policy = old_policy.copy()

        #***************************************************
        policy_stable = True                            # Initialize policy stable true
        for i in range(HEIGHT):                         # For each state(height, width)
            for j in range(WIDTH):
                action_value = np.zeros(len(ACTIONS))   # Create action value array(action value for each actions)
                for k, action in enumerate(ACTIONS):    # For each action at each state
                    (next_i, next_j), reward = env.interaction([i, j], action)
                    if not env.is_terminal([i, j]):     # If current state is not terminal
                        # Update action values using the 'Bellman equation'
                        action_value[k] = (reward + discount * state_values[next_i, next_j])
                # Find action that maximize action values (argmax) 1.find one / 2.find all
                # 1. find one - Make policy=1 when the action maximizes action values
                max_actions = np.argmax(action_value)
                policy[i, j] = np.zeros(len(ACTIONS))
                policy[i, j, max_actions] = 1

                # 2. find all - Make policy=1/(# of max actions) when the action maximizes action values
                #max_actions = np.where(action_value == np.max(action_value))[0]
                #policy[i, j] = np.zeros(len(ACTIONS))  # Make policy=0 when the action doesn't maximize action values
                #policy[i, j, max_actions] = 1 / len(max_actions)

        if not np.array_equal(policy, old_policy):      # If Policy changes(new policy != old policy)
            policy_stable = False                       # Finish policy improvement and goes to policy evaluation
        #***************************************************

        print('policy stable {}:'.format(policy_stable))
        draw_policy_image(iter, np.round(policy, decimals=2), env=env)
        return policy, policy_stable

    def policy_iteration(self):
        iter = 1
        while (True):
            self.values, iteration = self.policy_evaluation(iter, env=self.env, policy=self.policy)
            self.policy, policy_stable = self.policy_improvement(iter, env=self.env, state_values=self.values,
                                                       old_policy=self.policy, discount=1.0)
            iter += 1
            if policy_stable == True:
                break
        np.savez('./result/dp.npz', V=self.values, PI=self.policy)
        return self.values, self.policy


    def get_action(self, state):
        i,j = state
        return np.random.choice(len(ACTIONS), 1, p=self.policy[i,j,:].tolist()).item()


    def get_state(self):
        return self.state

