import gymnasium as gym
import numpy as np
import pandas as pd
import time
import matplotlib.pyplot as plt


class SilverLake():
    def __init__(self, render):
        self.render = render
    
    # Dynamic Programming Methodologies.
    def policy_iteration(self, iters):
        env = gym.make("FrozenLake-v1", map_name="4x4", is_slippery=False, render_mode="human" if self.render else None)
        policy = np.zeros(env.observation_space.n)

        def pprint(value):
            for i in range(0, 13, 4):
                print(value[i:i+4])

        def policy_eval(policy, env):
            value_table = np.zeros(shape=env.observation_space.n)
            gamma = 1
            threshold = 1e-10

            # v_\pi(s) =  \sum_a π(a|s)(R_S^a + \gamma\sum P_ss'^a v(s'))
            while True:
                new_value_table = np.copy(value_table)
                for state in range(env.observation_space.n):
                    action = policy[state]
                    value_table[state] = sum(
                        [trans_prob*(reward + gamma*new_value_table[next_state])
                        for  trans_prob, next_state, reward, _ in env.P[state][action]]
                    )
                
                if np.sum((np.fabs(new_value_table - value_table))) <= threshold:
                    break

            return new_value_table
        

        def update_policy(value_table, env, gamma = 1.0):
            policy = np.zeros(env.observation_space.n)

            for state in range(env.observation_space.n):
                Q_table = np.zeros(env.action_space.n)
                for action in range(env.action_space.n):
                    for trans_prob, next_state, reward_prob, _ in env.P[state][action]:
                        Q_table[action] += (trans_prob * (reward_prob + gamma*value_table[next_state]))
            
                policy[state] = np.argmax(Q_table)

            return policy

        for i in range(iters):
            policy_old = policy.copy()
            value = policy_eval(policy, env)
            print("VALUE ––––––––")
            pprint(value)
            policy = update_policy(value, env)
            print("POLICY –––––––––")
            pprint(policy)

            if np.all(policy_old == policy):
                print(f"Policy converged at {i}th iteration.")
                break
    

        self.policy_map = policy
        print(self.policy_map)

    def value_iteration(self, n_iter, gamma, return_policy=True, print_val=False):                                                    
        env = gym.make("FrozenLake-v1", map_name="4x4", is_slippery=False, render_mode="human" if self.render else None) 

        value_function = np.zeros(shape=env.observation_space.n)
        threshold = 1e-20

        Q_vals = np.zeros(shape=env.action_space.n)

        for i in range(n_iter):
            value_function_ = value_function.copy()

            for state in range(env.observation_space.n):
                for action in range(env.action_space.n):
                    temp_Q_table = [] # Q_table for Q val of each probbaility weighted action.
                    for next_state_rew in env.P[state][action]:
                        trans_prob, next_state, reward, _ = next_state_rew
                        temp_Q_table.append(trans_prob*(reward + gamma*(value_function_[next_state])))
                    
                    Q_vals[action] = sum(temp_Q_table)
                        
                value_function[state] = max(Q_vals) # Update val func with max(Q_val).
    
            if print_val:
                for i in range(0,16,4):
                    print(np.round(value_function[i:i+4],2))
                print("–––––––––––––")

            if np.sum(np.fabs(value_function_ - value_function)) <= threshold:
                print('Value-iteration converged at iteration# %d.' %(i+1))
                
                break
            
        if return_policy:
            policy = np.zeros(env.observation_space.n)
            for state in range(env.observation_space.n):
               Q_table = np.zeros(env.action_space.n)
               for action in range(env.action_space.n):
                   for next_sr in env.P[state][action]:
                       trans_prob, next_state, reward_prob, _ = next_sr
                       Q_table[action] += (trans_prob * (reward_prob + gamma*value_function[next_state]))
               policy[state] = np.argmax(Q_table)
            self.policy_map = policy
            return policy

        return value_function

    # On-Policy Model-Free Control Algorithms.
    def Sarsa(self, episodes):
        # Int Q(S,A)
        # repeat(eps):
            # Int S
            # Int A from ep-greedy w.r.t. Q
            # repeat(steps):
                # A @ S ---> R, S'
                # use policy get A'
                # Q = Q + a(R + g*Q(S',A') - Q)*E
                # A = A'
                # S = S'

        env = gym.make("FrozenLake-v1", map_name="4x4", is_slippery=False, render_mode="human" if self.render else None)
        Q = np.zeros(shape=(env.observation_space.n, env.action_space.n))
        alpha, gamma, epsilon = 0.3, 0.9, 1

        reward_arr = np.zeros(episodes)
        epsilon_arr = np.zeros(episodes)

        for i in range(episodes):
            state = env.reset()[0]
            action = np.argmax(Q[state]) if np.random.uniform() > epsilon else env.action_space.sample() 
            terminated, truncated = False, False
            
            while (not terminated and not truncated):
                next_state, reward, terminated, truncated, _ = env.step(action)
                next_action = np.argmax(Q[next_state]) if np.random.uniform() > epsilon else env.action_space.sample() 

                # Sarsa update on Q.
                Q[state, action] = Q[state, action] + alpha*(
                                    reward + gamma*Q[next_state, next_action] - Q[state, action]
                                    )
                action = next_action
                state = next_state
            
            epsilon = max(epsilon - 1/episodes, 0.15)
            reward_arr[i] = reward
            epsilon_arr[i] = epsilon

        print(reward_arr.sum()) 
        plt.plot(pd.Series(reward_arr).rolling(20).mean())
        plt.plot(pd.Series(epsilon_arr).rolling(20).mean())
        plt.savefig("results.png")

        self.policy_map = np.argmax(Q, axis=1)
        print(self.policy_map)
 
    def Sarsa_Lambda(self, episodes, lamb=0.7):
        # Int Q(S,A)
        # repeat(eps):
            # Int E(S,A)
            # Int S
            # Int A from ep-greedy w.r.t. Q
            # repeat(steps):
                # A @ S ---> R, S'
                # use policy get A'
                # E(S,A) += 1
                # for_all(S,A):
                    # Q = Q + a(R + g*Q(S',A') - Q)*E
                    # E(S,A) = lam*gE(S,A)
                # A = A'
                # S = S'

        env = gym.make("FrozenLake-v1", map_name="4x4", is_slippery=False, render_mode="human" if self.render else None)
        Q = np.zeros(shape=(env.observation_space.n, env.action_space.n))
        alpha, gamma, epsilon = 0.05, 0.9, 1

        reward_arr = np.zeros(episodes)
        epsilon_arr = np.zeros(episodes)

        for i in range(episodes):
            if i % 500 == 0:
                print(i)
            E = np.zeros(shape=(env.observation_space.n, env.action_space.n))
            state = env.reset()[0]
            action = np.argmax(Q[state]) if np.random.uniform() > epsilon else env.action_space.sample() 
            terminated, truncated = False, False
            
            while (not terminated and not truncated):
                next_state, reward, terminated, truncated, _ = env.step(action)
                next_action = np.argmax(Q[next_state]) if np.random.uniform() > epsilon else env.action_space.sample() 
                E[state, action] = E[state, action] + 1

                # Sarsa update on Q.
                for s in range(env.observation_space.n):
                    for a in range(env.action_space.n):
                        Q[s, a] = Q[s, a] + alpha*E[s, a]*(
                            reward + gamma*Q[next_state, next_action] - Q[s, a]
                        )
                        E[s, a] = lamb*gamma*E[s, a]
                
                action = next_action
                state = next_state
            
            epsilon = max(epsilon - 1/(episodes*1.1), 0.1)
   
            reward_arr[i] = reward
            epsilon_arr[i] = epsilon

        print(reward_arr.sum()) 
        plt.plot(pd.Series(reward_arr).rolling(20).mean())
        plt.plot(pd.Series(epsilon_arr).rolling(20).mean())
        plt.savefig("results.png")

        self.policy_map = np.argmax(Q, axis=1)

    # Deep Q-Learning i.e. AVF Approximation for Control.
    def DeepQ(self):
        pass

    def test(self, iters, render):
        env = gym.make("FrozenLake-v1", map_name="4x4", is_slippery=False, render_mode="human" if render else None) 
        if render:
            env.reset()
            env.render()
            time.sleep(3)

        reward_hist = np.zeros(shape=iters)

        for i in range(iters):
            terminated, truncated = False, False
            state = env.reset()[0]
            while not terminated and not truncated:
                action = int(self.policy_map[state])
                new_state, reward, terminated, truncated, _ = env.step(action)

                state = new_state
            reward_hist[i] = reward

        print(f"The success rate is: {np.mean(reward_hist)}")


model = SilverLake(render=False)
res = model.Sarsa_Lambda(episodes=8000)
model.test(iters=10, render=True)


