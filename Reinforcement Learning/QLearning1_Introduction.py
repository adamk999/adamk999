import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
import pickle

def run(episodes, is_training=True, render=False):
    env = gym.make('MountainCar-v0', render_mode="human" if render else None)

    alpha = 0.9
    gamma = 0.9
    bins = 20

    pos_space = np.linspace(env.observation_space.low[0], env.observation_space.high[0], bins)
    vel_space = np.linspace(env.observation_space.low[1], env.observation_space.high[1], bins)

    epsilon = 1
    epsilon_decay_r = 2/episodes
    rng = np.random.default_rng()

    if is_training:
        q = np.random.uniform(low=-2, high=0, size=(bins, bins, env.action_space.n))
    else:
        f = open("mountain_car.pkl", "rb")
        q = pickle.load(f)
        f.close()

    rewards_arr = np.zeros(episodes)

    for ep in range(episodes):
        done = False
        if ep % 200 == 0:
            print(ep)
        rewards = 0
        state = env.reset()[0]
        state_p = np.digitize(state[0], pos_space)
        state_v = np.digitize(state[1], vel_space)

        while (not done and rewards > -1000):
            if is_training and rng.random() < epsilon:
                action = env.action_space.sample()
            else:
                action = np.argmax(q[state_p, state_v, :])

            new_state, reward, done, _, _ = env.step(action)
            new_state_p = np.digitize(new_state[0], pos_space)
            new_state_v = np.digitize(new_state[1], vel_space)

            if is_training:
                q[state_p, state_v, action] = q[state_p, state_v, action] + alpha*(
                        reward + gamma*np.max(q[new_state_p,new_state_v,:]) - q[state_p, state_v, action]
                )
            state = new_state
            state_p = new_state_p
            state_v = new_state_v

            rewards += reward

        epsilon = max(epsilon - epsilon_decay_r, 0)
        rewards_arr[ep] = rewards

    env.close()

    mean_rewards = np.zeros(episodes)
    for t in range(episodes):
        mean_rewards[t] = np.mean(rewards_arr[max(0,t-100):t+1])
    plt.plot(mean_rewards)
    plt.savefig("mean_rew.png")

    if is_training:
        f = open("mountain_car.pkl", "wb")
        pickle.dump(q, f)
        f.close()

run(3, is_training=True, render=True)
