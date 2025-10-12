import math
import numpy as np
import matplotlib.pyplot as plt

class Estimate:
    def __init__(self, init_estimate:float):
        self.time_step = 0
        self.init_estimate = init_estimate

class Action:
    def __init__(self, true_reward_mean: float, true_reward_var: float=1):
        self.true_reward_mean = true_reward_mean
        self.true_reward_var = true_reward_var
    
    def get_reward(self) -> float:
        reward = np.random.normal(self.true_reward_mean, np.sqrt(self.true_reward_var))
        return reward 
    
class Plotter:
    @staticmethod
    def plot_chosen_action_over_time(actions):
        t = np.arange(len(actions))
        plt.figure()
        plt.plot(t, actions)
        plt.title("Chosen Action over Time")
        plt.xlabel("Step")
        plt.ylabel("Action ID")
        plt.yticks(range(exp.num_actions))
        plt.grid(True)
        plt.show()

    @staticmethod
    def plot_reward_over_time(rewards):
        t = np.arange(len(rewards))
        # 2) Reward over time (+ rolling mean)
        plt.figure()
        plt.plot(t, rewards, label="Reward")
        # rolling mean (e.g., window=50)
        win = 50
        if len(rewards) >= win:
            kernel = np.ones(win) / win
            rolling = np.convolve(rewards, kernel, mode="valid")
            plt.plot(np.arange(win - 1, len(rewards)), rolling, label=f"Rolling mean ({win})")
        plt.title("Reward over Time")
        plt.xlabel("Step")
        plt.ylabel("Reward")
        plt.grid(True)
        plt.legend()
        plt.show()

    @staticmethod
    def plot_cum_optimal_action_rate(optimal):
        t = np.arange(len(optimal))
        # 3) Cumulative optimal action rate
        cum_opt_rate = np.cumsum(optimal) / (t + 1)
        plt.figure()
        plt.plot(t, cum_opt_rate)
        plt.title("Cumulative Optimal Action Rate")
        plt.xlabel("Step")
        plt.ylabel("Optimal %")
        plt.ylim(0, 1)
        plt.grid(True)

        plt.show()


class Experiment:
    def __init__(self, num_time_steps: int, num_actions:int, init_estimate:float):
        self.num_time_steps = num_time_steps
        self.num_actions = num_actions
        self.init_estimate = init_estimate
        self._init_actions()
    
    def _init_actions(self):
        self.max_true_reward_mean = -np.inf
        self.actions = []
        for _ in range(self.num_actions):
            true_reward_mean = np.random.normal(0,1)
            action = Action(true_reward_mean=true_reward_mean, true_reward_var=1)
            self.actions.append(action)
            if self.max_true_reward_mean < true_reward_mean:
                self.max_true_reward_mean = true_reward_mean
        
        # Precompute the index of the optimal action (by true mean), for metrics
        self.opt_idx = int(np.argmax([a.true_reward_mean for a in self.actions]))

    def run_greedy(self):
        """
        Pure greedy (no exploration).
        - At each t, choose action with largest current estimate
        - Update that actions's estimate with the new reward (incremental mean).
        - Carry forward other actions's estimates for plotting
        """
        A = self.num_actions
        Q = np.full(self.num_actions, self.init_estimate, dtype=float)
        counts = np.zeros(self.num_actions, dtype=int)

        actions_hist = []
        rewards_hist = []
        optimal_hits = []

        for t in range(self.num_time_steps):
            a = int(np.argmax(Q))

            r = self.actions[a].get_reward()

            # incremental mean update for chosen action
            counts[a] += 1 
            alpha = 1.0 / counts[a]
            Q[a] = Q[a] + alpha * (r - Q[a])

            # Record history
            actions_hist.append(a)
            rewards_hist.append(r)
            optimal_hits.append(1 if a == self.opt_idx else 0)

        return {
            "actions": actions_hist,
            "rewards": rewards_hist,
            "optimal_hits": optimal_hits
        }

    def run_epsilon_greedy(self, epsilon:float):
        A = self.num_actions 
        Q = np.full(self.num_actions, self.init_estimate, dtype=float)
        counts = np.zeros(self.num_actions, dtype=int)

        actions_hist = []
        rewards_hist = []
        optimal_hits = []

        for t in range(self.num_time_steps):
            if np.random.rand() < epsilon: # super interesting
                # explore 
                a = np.random.randint(self.num_actions)
            else:
                # exploit 
                a = int(np.argmax(Q))

            r = self.actions[a].get_reward()

            # incremental mean update for chosen action
            counts[a] += 1 
            alpha = 1.0 / counts[a]
            Q[a] = Q[a] + alpha * (r - Q[a])

            # record history
            actions_hist.append(a)
            rewards_hist.append(r)
            optimal_hits.append(1 if a == self.opt_idx else 0)

        return {
            "actions": actions_hist,
            "rewards": rewards_hist,
            "optimal_hits": optimal_hits
        }


exp = Experiment(num_time_steps=1000, num_actions=10, init_estimate=0)
print(f"Max true reward: {exp.max_true_reward_mean}")
greedy_ret = exp.run_greedy()
eps_greedy_ret = exp.run_epsilon_greedy(epsilon=0.2)


actions = np.array(eps_greedy_ret["actions"])
rewards = np.array(eps_greedy_ret["rewards"])
optimal = np.array(eps_greedy_ret["optimal_hits"])

Plotter.plot_chosen_action_over_time(actions)
Plotter.plot_reward_over_time(rewards)






