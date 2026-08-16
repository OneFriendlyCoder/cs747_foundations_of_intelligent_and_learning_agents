"""
NOTE: You are only allowed to edit this file between the lines that say:
    # START EDITING HERE
    # END EDITING HERE

This file contains the base Algorithm class that all algorithms should inherit
from. Here are the method details:
    - __init__(self, num_arms, horizon): This method is called when the class
        is instantiated. Here, you can add any other member variables that you
        need in your algorithm.
    
    - give_pull(self): This method is called when the algorithm needs to
        select an arm to pull. The method should return the index of the arm
        that it wants to pull (0-indexed).
    
    - get_reward(self, arm_index, reward): This method is called just after the 
        give_pull method. The method should update the algorithm's internal
        state based on the arm that was pulled and the reward that was received.
        (The value of arm_index is the same as the one returned by give_pull.)

We have implemented the epsilon-greedy algorithm for you. You can use it as a
reference for implementing your own algorithms.
"""

import numpy as np
import math
# Hint: math.log is much faster than math.log for scalars

class Algorithm:
    def __init__(self, num_arms, horizon):
        self.num_arms = num_arms
        self.horizon = horizon
    
    def give_pull(self):
        raise NotImplementedError
    
    def get_reward(self, arm_index, reward):
        raise NotImplementedError

# Example implementation of Epsilon Greedy algorithm
class Eps_Greedy(Algorithm):
    def __init__(self, num_arms, horizon):
        super().__init__(num_arms, horizon)
        # Extra member variables to keep track of the state
        self.eps = 0.1
        self.counts = np.zeros(num_arms)
        self.values = np.zeros(num_arms)
    
    def give_pull(self):
        if np.random.random() < self.eps:
            return np.random.randint(self.num_arms)
        else:
            return np.argmax(self.values)
    
    def get_reward(self, arm_index, reward):
        self.counts[arm_index] += 1
        n = self.counts[arm_index]
        value = self.values[arm_index]
        new_value = ((n - 1) / n) * value + (1 / n) * reward
        self.values[arm_index] = new_value


# START EDITING HERE
# You can use this space to define any helper functions that you need

# END EDITING HERE

class UCB(Algorithm):
    def __init__(self, num_arms, horizon):
        super().__init__(num_arms, horizon)
        # You can add any other variables you need here
        # START EDITING HERE
        self.counts = np.zeros(num_arms)
        self.values = np.zeros(num_arms)
        self.total_pulls = 0
        # END EDITING HERE
    
    def give_pull(self):
        # START EDITING HERE
        self.total_pulls+=1
        for a in range(self.num_arms):
            if(self.counts[a] == 0):
                return a
        ucb_values = self.values + np.sqrt((2*math.log(self.total_pulls))/self.counts)
        return np.argmax(ucb_values)
        # END EDITING HERE  
        
    
    def get_reward(self, arm_index, reward):
        # START EDITING HERE
        self.counts[arm_index]+=1
        n = self.counts[arm_index]
        value = self.values[arm_index]
        self.values[arm_index] = ((n-1)*value+reward)/n
        # END EDITING HERE


class KL_UCB(Algorithm):
    def __init__(self, num_arms, horizon):
        super().__init__(num_arms, horizon)
        # You can add any other variables you need here
        # START EDITING HERE
        self.counts = np.zeros(num_arms)
        self.values = np.zeros(num_arms)
        self.e = 1e-6
        self.c = 3

    def kl_divergence(self,p,q):
        eps = 1e-12
        p = np.clip(p,eps,1-eps)
        q = np.clip(q,eps,1-eps)
        return (p*math.log(p/q)+((1-p)*math.log((1-p)/(1-q))))

        # END EDITING HERE
    
    def give_pull(self):
        # START EDITING HERE
        for a in range(self.num_arms):
            if self.counts[a] == 0:
                return a

        t = int(np.sum(self.counts))
        ucb_values = np.zeros(self.num_arms)
        for a in range(self.num_arms):
            p = self.values[a]
            low, high = p, 1.0
            rhs = (math.log(t)+self.c*math.log(math.log(max(t,2))))/self.counts[a]
            for _ in range(25):
                q = (low+high)/2
                kl = self.kl_divergence(p,q)
                if kl > rhs:
                    high = q
                else:
                    low = q
            ucb_values[a] = low
        return np.argmax(ucb_values)
        # END EDITING HERE
    
    def get_reward(self, arm_index, reward):
        # START EDITING HERE
        self.counts[arm_index]+=1
        n = self.counts[arm_index]
        value = self.values[arm_index]
        self.values[arm_index] = ((n-1)*value+reward)/n
        # END EDITING HERE

class Thompson_Sampling(Algorithm):
    def __init__(self, num_arms, horizon):
        super().__init__(num_arms, horizon)
        # You can add any other variables you need here
        # START EDITING HERE
        self.st_a = np.zeros(num_arms)
        self.ft_a = np.zeros(num_arms)
        # END EDITING HERE
    
    def give_pull(self):
        # START EDITING HERE
        # Sample from Beta distribution for each arm
        samples = np.random.beta(self.st_a+1, self.ft_a+1)
        mask = samples == samples.max()
        return np.random.choice(np.arange(len(samples))[mask])
        # END EDITING HERE
    
    def get_reward(self, arm_index, reward):
        # START EDITING HERE
        if reward == 1:
            self.st_a[arm_index]+=1
        else:
            self.ft_a[arm_index]+=1
        # END EDITING HERE

