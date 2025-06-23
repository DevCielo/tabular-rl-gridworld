import argparse
import gym
from envs.learning_path_env import LearningPathEnv
from agents.q_learning_agent import QLearningAgent
import numpy as np

def train(env, agent, episodes=500):
    returns = []
    for ep in range(episodes):
        state = env.reset()
        total_r = 0
        action = agent.select_action(state)
        done = False
        while not done:
            next_state, r, done, _ = env.step(action)
            next_action = agent.select_action(next_state)
            # for Q-learning:
            agent.update(state, action, r, next_state, done)
            # for SARSA (uncomment below and comment above)
            # agent.update(state, action, r, next_state, next_action, done)

            state, action = next_state, next_action
            total_r += r
        returns.append(total_r)
        if (ep+1)%50 == 0:
            print(f"Episode {ep+1}/{episodes}, return={total_r}")
    return returns

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=500)
    args = parser.parse_args()

    # define some prereqs (must visit (0,1) before (1,1))
    prereqs = { (1,1): (0,1), (2,2):(1,2)}

    env = LearningPathEnv(grid_shape=(4,4), start=(0,0), goal=(3,3),
    prerequisites = prereqs)

    agent = QLearningAgent(n_states=env.observation_space.n, n_actions=env.action_space.n, alpha=0.1, gamma=0.99, epsilon=0.1)

    returns = train(env, agent, episodes=args.episodes)

    # save Q-table for visualisation
    np.save("q_table.npy", agent.Q)
    print("Training complete, Q-table saved to q_table.npy")
