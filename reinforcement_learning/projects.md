# Reinforcement Learning (RL) Projects Checklist

## **Beginner Projects**
- [ ] **CartPole with Deep Q-Network (DQN)**
  - Train a DQN agent to balance the pole in the `CartPole-v1` environment.
  - Implement experience replay and target networks.
  - Compare vanilla DQN with Double DQN.

- [ ] **Lunar Lander with PPO**
  - Use **Proximal Policy Optimization (PPO)** to land a spacecraft in `LunarLander-v2`.
  - Experiment with clipped surrogate objectives.
  - Compare PPO with DQN.

- [ ] **Taxi-V2 with Q-Learning**
  - Train a **tabular Q-learning agent** to navigate the `Taxi-v2` environment.
  - Implement **epsilon-greedy exploration** and a **decaying learning rate**.
  - Compare performance with **SARSA**.

- [ ] **FrozenLake with Policy Iteration**
  - Solve the `FrozenLake-v1` environment using **Dynamic Programming** methods.
  - Implement **Policy Iteration** and **Value Iteration**.
  - Compare results with **Monte Carlo methods**.

- [ ] **Self-Play Tic-Tac-Toe**
  - Implement an **RL agent that learns Tic-Tac-Toe through self-play**.
  - Train using **Q-learning**.
  - Compare with a **Minimax-based AI**.

## **Intermediate Projects**
- [ ] **Atari Pong with Deep Q-Networks**
  - Train a **DQN agent to play Atari Pong**.
  - Use **frame stacking** and **convolutional layers**.
  - Experiment with **reward clipping** and **prioritized experience replay**.

- [ ] **Continuous Control with Soft Actor-Critic (SAC)**
  - Train an **SAC agent** on the `Pendulum-v1` environment.
  - Experiment with **entropy tuning**.
  - Compare with **DDPG (Deep Deterministic Policy Gradient)**.

- [ ] **Stock Trading Agent with RL**
  - Train an RL agent to **trade stocks using Deep Q-Learning**.
  - Use OpenAI Gym's **TradingEnv** or a **custom market simulation**.
  - Compare performance with **moving average strategies**.

## **Advanced Projects**
- [ ] **Multi-Agent RL for Traffic Signal Optimization**
  - Train **multiple RL agents** to optimize traffic lights in a simulated city.
  - Use **Multi-Agent Deep Deterministic Policy Gradient (MADDPG)**.
  - Compare with traditional **rule-based traffic systems**.

- [ ] **MuJoCo Humanoid Locomotion with PPO**
  - Train a **humanoid robot to walk using PPO** in MuJoCo’s `Humanoid-v2`.
  - Experiment with different **reward functions** and **hyperparameters**.
  - Compare PPO with **TRPO (Trust Region Policy Optimization)** and **SAC**.
