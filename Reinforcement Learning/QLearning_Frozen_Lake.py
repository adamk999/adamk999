import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
import pickle

def run(episodes, training, render):
    desc = ["SFFF", "FHFH", "FFFH", "HFFG"]
    env = gym.make("FrozenLake-v1", desc=desc, is_slippery=False, render_mode="human" if render else None)

    alpha = 0.9
    gamma = 0.9
    epsilon = 1
    epsilon_decay_r = 0.0001
    rng = np.random.default_rng()

    if training:
        q = np.zeros(shape=(16, 4))
    else:
        with open("frozen_lake.pkl", "rb") as f:
            q = pickle.load(f)

    reward_arr = np.zeros(episodes)

    for ep in range(episodes):
        if ep % 200 == 0:
            print(ep, epsilon)
        state = env.reset()[0]
        if render:
            env.render()

        done = False
        iters = 0
        rewards = 0

        while not done and iters < 100:
            if training and rng.random() < epsilon:
                action = env.action_space.sample()
            else:
                action = np.argmax(q[state, :])

            new_state, reward, done, _, _ = env.step(action)
            if reward == 1:
                print(ep, "Success")
            if training:
                q[state, action] = q[state, action] + alpha * (reward + gamma * np.max(q[new_state, :]) - q[state, action])
            iters += 1
            rewards += reward
            state = new_state

        epsilon = max(epsilon - epsilon_decay_r, 0.2)
        reward_arr[ep] = rewards

    if training:
        with open("frozen_lake.pkl", "wb") as f:
            pickle.dump(q, f)

    mean_rewards = np.zeros(episodes)
    for t in range(episodes):
        mean_rewards[t] = np.mean(reward_arr[max(0, t - 100):t + 1])
    plt.plot(mean_rewards)
    plt.savefig("FLake_Rewards.png")

run(100, training=False, render=True)
