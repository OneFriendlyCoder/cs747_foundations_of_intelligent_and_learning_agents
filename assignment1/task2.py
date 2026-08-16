import numpy as np
from typing import List, Optional, Dict, Tuple

# =========================================================
# ===============   ENVIRONMENT (Poisson)   ===============
# =========================================================

class PoissonDoorsEnv:
    """
    This creates a Poisson environment. There are K doors and each has an associated mean.
    In each step you pick an arm i. Damage to a door is drawn from its corresponding
    Poisson Distribution. Initial health of each door is H0 and decreases by damage in each step.
    Game ends when any door's health < 0.
    """
    def __init__(self, mus: List[float], H0: int = 100, rng: Optional[np.random.Generator] = None):
        self.mus = np.array(mus, dtype=float)
        assert np.all(self.mus > 0), "Poisson means must be > 0"
        self.K = len(mus)
        self.H0 = H0
        self.rng = rng if rng is not None else np.random.default_rng()
        self.reset()

    def reset(self):
        self.health = np.full(self.K, self.H0, dtype=float)
        self.t = 0
        return self.health.copy()

    def step(self, arm: int) -> Tuple[float, bool, Dict]:
        reward = float(self.rng.poisson(self.mus[arm]))
        self.health[arm] -= reward
        self.t += 1
        done = np.any(self.health < 0.0)
        return reward, done, {"reward": reward, "health": self.health.copy(), "t": self.t}


# =========================================================
# =====================   POLICIES   ======================
# =========================================================

class Policy:
    """
    Base Policy interface.
    - Implement select_arm(self, t) to return an int in [0, K-1] to choose an arm.
    - Optionally override update(...) for custom learning.
    """
    def __init__(self, K: int, rng: Optional[np.random.Generator] = None):
        self.K = K
        self.rng = rng if rng is not None else np.random.default_rng()
        self.counts = np.zeros(K, dtype=int)
        self.sums   = np.zeros(K, dtype=float)

    def reset_stats(self):
        self.counts[:] = 0
        self.sums[:]   = 0.0

    def update(self, arm: int, reward: float):
        self.counts[arm] += 1
        self.sums[arm]   += reward

    @property
    def means(self) -> np.ndarray:
        with np.errstate(divide="ignore", invalid="ignore"):
            return self.sums / np.maximum(self.counts, 1)

    def select_arm(self, t: int) -> int:
        raise NotImplementedError

## TASK 2: Make changes here to implement your policy ###
class StudentPolicy(Policy):
    """
    Implement your own algorithm here.
    Replace select_arm with your strategy.
    Currently it has a simple implementation of the epsilon greedy strategy.
    Change this to implement your algorithm for the problem.
    """
    def __init__(self, K: int, rng: Optional[np.random.Generator] = None):
        super().__init__(K, rng)
        self.eps = 1e-12
        self.health = np.full(K, 100, dtype=float)   # health of each door is hardcoded
        self.a0 = 1.0
        self.b0 = 1.0
        self.exploit = K
        self.fin_thresh = 0.12
        self.xplr_setsize = max(1, int(np.ceil(K*1/3)))
        self.xplr_set = None
        self.exploitation = False

    def posterior_params(self,a):
        alpha = self.a0+self.sums[a]
        beta = self.b0+max(1,int(self.counts[a]))
        return alpha,beta

    def posterior_mean(self,alpha,beta):
        return alpha/beta

    def poisson_cdf(self,lam,k):
        if k<0:
            return 0.0
        term = np.exp(-lam)
        s = term
        for i in range(1,k+1):
            term*=lam/i
            s+=term
        return min(1.0,s)

    def ensure_xplr_set(self):
        if self.xplr_set is not None:
            return 
        post_means = np.array(list(map(lambda a: self.posterior_mean(*self.posterior_params(a)),range(self.K))))
        expected_steps = self.health / np.maximum(post_means, self.eps)
        if np.all(self.counts == 0) or np.isclose(np.std(expected_steps),0.0):
            self.xplr_set = self.rng.choice(self.K, size=self.xplr_setsize, replace=False)
        else:
            self.xplr_set = np.argsort(expected_steps)[:self.xplr_setsize]
        self.exploitation = False

    def select_arm(self, t: int) -> int:
        self.ensure_xplr_set()
        if not self.exploitation:
            unpulled_arm = [a for a in self.xplr_set if self.counts[a] == 0]
            if unpulled_arm:
                post_means_local = np.array(list(map(lambda a: self.posterior_mean(*self.posterior_params(a)), unpulled_arm)))
                expected_steps = np.array(list(map(lambda a_pm: self.health[a_pm[0]] / max(a_pm[1], self.eps), zip(unpulled_arm, post_means_local))))
                return int(unpulled_arm[int(np.argmin(expected_steps))])
            else:
                self.exploitation = True
        post_means = np.array(list(map(lambda a: self.posterior_mean(*self.posterior_params(a)), range(self.K))))

        finish_probs = []
        for a in range(self.K):
            threshold = int(np.ceil(self.health[a]))-1
            prob = 1.0 - self.poisson_cdf(post_means[a],threshold)
            finish_probs.append(prob)

        finish_probs = np.array(finish_probs)
        finish_thresh = max(0.01,self.fin_thresh*self.exploit/max(1,t))
        best_finish = int(np.argmax(finish_probs))
        if finish_probs[best_finish]>=finish_thresh:
            return best_finish
        pm_indices = self.health/np.maximum(post_means,self.eps)
        return int(np.argmin(pm_indices))

    def update(self, arm: int, reward: float):
        super().update(arm, reward)
        self.health[arm] -= reward
        if self.xplr_set is not None and not self.exploitation:
            all_pulled = True
            for a in self.xplr_set:
                if self.counts[a] <= 0:
                    all_pulled = False
                    break
            if all_pulled:
                self.exploitation = True

