"""
Task 3: Optimized KL-UCB Implementation

This file implements both standard and optimized KL-UCB algorithms for multi-armed bandits.
The optimized version aims to reduce computational overhead while maintaining good regret performance.
"""

import math
import numpy as np
import matplotlib.pyplot as plt

# ------------------ Base Algorithm Class ------------------

class Algorithm:
    def __init__(self, num_arms, horizon):
        self.num_arms = num_arms
        self.horizon = horizon
    
    def give_pull(self):
        raise NotImplementedError
    
    def get_reward(self, arm_index, reward):
        raise NotImplementedError

# ------------------ KL-UCB utilities ------------------
## You can define other helper functions here if needed

# ------------------ Optimized KL-UCB Algorithm ------------------

class KL_UCB_Optimized(Algorithm):
    """
    Optimized KL-UCB algorithm that reduces computation while maintaining identical regret.
    This implements a batched KL-UCB with exponential+binary search for safe pulls of the current best arm.
    """
    ## You can define other functions also in the class if needed
    
    def __init__(self, num_arms, horizon):
        super().__init__(num_arms, horizon)
        # can initialize member variables here
        #START EDITING HERE
        self.counts = np.zeros(num_arms)
        self.values = np.zeros(num_arms)
        self.c = 2.3
        self.current_arm = -1
        self.pull_count = 0
        self.ucb_values = np.full(num_arms, float('inf'))
        self.max_batch = 10000
        self.current_batch = 1

    def kl_divergence(self,p,q):
        eps = 1e-12
        p = np.clip(p,eps,1-eps)
        q = np.clip(q,eps,1-eps)
        return (p*math.log(p/q)+((1-p)*math.log((1-p)/(1-q))))
    
    def compute_ucb(self,a,t):
        p = self.values[a]
        rhs = (math.log(t)+self.c*math.log(math.log(max(t,2))))/self.counts[a]
        n = self.counts[a]
        low,high = p,1.0
        base_iter = int(max(3,math.log(self.horizon)))
        num_iterations = min(10,base_iter-int(math.log(n+1)/2)-int(math.log(self.num_arms)/2))
        for _ in range(num_iterations):
            q = (low+high)/2
            if self.kl_divergence(p,q)>rhs:
                high = q
            else:
                low = q
        return low
        #END EDITING HERE
    
    def give_pull(self):
        #START EDITING HERE
        if np.min(self.counts) == 0:
            for a in range(self.num_arms):
                if self.counts[a] == 0:
                    self.current_arm = a
                    self.pull_count = 1
                    self.current_batch = 1
                    return a
        if self.pull_count < self.current_batch:
            self.pull_count+=1
            return self.current_arm
        t = int(np.sum(self.counts))
        self.ucb_values[self.current_arm] = self.compute_ucb(self.current_arm,t)
        self.current_arm = int(np.argmax(self.ucb_values))
        n = int(self.counts[self.current_arm])
        if n <= 0:
            self.current_batch = 1
        else:
            exponent = int(math.floor(math.log2(n+1)))
            self.current_batch = min(self.max_batch,max(1,2**exponent))
        self.pull_count = 1
        return self.current_arm
        #END EDITING HERE

    def get_reward(self, arm_index, reward):
        #START EDITING HERE
        self.counts[arm_index]+=1
        n = self.counts[arm_index]
        self.values[arm_index]+=(reward-self.values[arm_index])/n
        #END EDITING HERE

# ------------------ Bonus KL-UCB Algorithm (Optional - 1 bonus mark) ------------------

class KL_UCB_Bonus(Algorithm):
    """
    BONUS ALGORITHM (Optional - 1 bonus mark)
    
    This algorithm must produce EXACTLY IDENTICAL regret trajectories to KL_UCB_Standard
    while achieving significant speedup. Students implementing this will earn 1 bonus mark.
    
    Requirements for bonus:
    - Must produce identical regret trajectories (checked with strict tolerance)
    - Must achieve specified speedup thresholds on bonus testcases
    - Must include detailed explanation in report
    """
    # You can define other functions also in the class if needed

    def __init__(self, num_arms, horizon):
        super().__init__(num_arms, horizon)
        self.counts = np.zeros(num_arms)
        self.values = np.zeros(num_arms)
        self.c = 3

    def kl_divergence(self,p,q):
        eps = 1e-12
        p = np.clip(p,eps,1-eps)
        q = np.clip(q,eps,1-eps)
        return (p*np.log(p/q)+((1-p)*np.log((1-p)/(1-q))))

    def compute_ucb(self,t):
        pulled_ai = np.where(self.counts>0)[0]
        ucb_values = np.full(self.num_arms,float('inf'))        
        if len(pulled_ai) == 0:
            return ucb_values
        p = self.values[pulled_ai]
        n = self.counts[pulled_ai]
        rhs = (math.log(t)+self.c*math.log(math.log(max(t,2))))/n        
        low = p.copy()
        high = np.ones_like(p)
        for _ in range(25):
            q = (low+high)/2
            kl_div = self.kl_divergence(p,q)
            mask = kl_div>rhs
            high[mask] = q[mask]
            low[~mask] = q[~mask]
        ucb_values[pulled_ai] = low
        return ucb_values
        #END EDITING HERE

    def give_pull(self):
        #START EDITING HERE
        t = int(np.sum(self.counts)+1)
        if t <= self.num_arms:
            return t-1
        ucb_values = self.compute_ucb(t)
        return np.argmax(ucb_values)
        #END EDITING HERE

    def get_reward(self,arm_index,reward):
        #START EDITING HERE
        self.counts[arm_index] += 1
        n = self.counts[arm_index]
        self.values[arm_index]+=(reward-self.values[arm_index])/n
        #END EDITING HERE