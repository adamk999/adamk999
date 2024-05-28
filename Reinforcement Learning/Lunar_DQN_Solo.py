import torch
import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
import random
import pandas as pd
from collections import deque
from torch import nn, FloatTensor
import torch.nn.functional as F


# Create NN model/class.
# Define replay class.
# Create training class:
    # ––––– TRAIN –––––
    # Int Q/target & policy DQN, replay, espsilon.
    # Iterate(episodes):
        # Int S
        # Iterate(steps):
            # A <-- epsilon-g w.r.t. policy DQN.
            # A --> S', R
            # Add (S, A, S', R) to memory.
            # S = S'
        
        # If len(mem) > x & total reward > y:
            # Sample mini_batch
            # Optimise policy DQN

            # If net_sync_rate > z:
                # Copy policy DQN --> target DQN.
            
            # Decay epsilon

    # –––––– Optimise –––––––
    # Int policy_Q_arr = [], target_Q_arr = []

    # for S, A, R, S', term in mini_batch:
        # if term:
            # target = R
        # else:
            # target = R + gam*max_A'Q_hat(S',A')

        # current_q = policy_DQN(S)
        # apppend --> policy_Q_arr

        # target_q = target_DQN(S)
        # target_q[A] = target
        # append --> target_Q_arr
    
    # loss = MSE(target_Q_arr, policy_Q_arr)
    # Optimise model.


# Define function approximator.
class DQN(nn.Module):
    def __init__(self, input_states_n, fc1_nodes, fc2_nodes, output_states_n):
        super().__init__()

        # Define network layers.
        self.fc1 = nn.Linear(input_states_n, fc1_nodes)
        self.fc2 = nn.Linear(fc1_nodes, fc2_nodes)
        self.output = nn.Linear(fc2_nodes, output_states_n)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.output(x)

        return x

# Define experience replay class.
class ReplayMemory:
    def __init__(self, maxlen):
        self.memory = deque([], maxlen=maxlen)

    def append(self, observation):
        self.memory.append(observation)

    def sample(self, n):
        return random.sample(self.memory, n)

    def __len__(self):
        return len(self.memory)



# Define environment.
class LunarLander:
    # Hyperparameters:
    alpha = 7e-3             # Learaning rate.
    gamma = 0.99               # Discount factor.
    tau = 1e-3
    ep_decay = 0.9991 
    #network_sync_rate = 100      # Num steps agent takes before syncing policy/target networks.
    update_every = 4
    replay_memory_size = int(1e5) # Size of replay memory.
    mini_batch_size = 2**6     # Size of training dataset sampled from replay memory.

    # Neural Network.
    fc1_nodes = 64
    fc2_nodes = 64


    def train(self, episodes):
        # Instantiate environment.
        env = gym.make("LunarLander-v2")
        n_input = 8
        n_output = 4 
        
        memory = ReplayMemory(maxlen=self.replay_memory_size)

        # Instantiate policy and target DQNs.
        policy_DQN = DQN(input_states_n=n_input, fc1_nodes=self.fc1_nodes, 
                         fc2_nodes=self.fc2_nodes, output_states_n=n_output)
        target_DQN = DQN(input_states_n=n_input, fc1_nodes=self.fc1_nodes, 
                         fc2_nodes=self.fc2_nodes, output_states_n=n_output)
        
        # Align target and policy networks.
        target_DQN.load_state_dict(policy_DQN.state_dict())
        
        # Set optimiser; note we are optimising the policy DQN.
        self.optimiser = torch.optim.Adam(policy_DQN.parameters(), lr=self.alpha)

        epsilon = 1
        step_count = 0
        
        epsilon_arr = np.zeros(episodes)
        reward_arr = np.zeros(episodes)
        actions = []
        
        for i in range(episodes):
            truncated, terminated = False, False
            state = env.reset()[0]
            total_reward = 0
            counter = 0
            while (not terminated and not truncated):
                if np.random.uniform() > epsilon:
                    with torch.no_grad():
                        action = policy_DQN(FloatTensor(state)).argmax().item()  
                        actions.append(action)
                else:
                    action = np.random.randint(0,4)
                
                new_state, reward, terminated, truncated, _ = env.step(action)
                
                memory.append((state, action, reward, new_state, terminated))
    
                state = new_state
                total_reward += reward
                step_count += 1
                counter += 1
                       
                if len(memory) > self.mini_batch_size and step_count > self.update_every:
                    mini_batch = memory.sample(self.mini_batch_size)
                    policy_DQN, target_DQN = self.optimise(mini_batch, policy_DQN, target_DQN)

                    step_count = 0

            epsilon = max(epsilon*self.ep_decay, 0.01)
            epsilon_arr[i] = epsilon
            reward_arr[i] = total_reward
            
            if i % 100 == 0:
                print(i)
                print(np.mean(reward_arr[i-50:i]))
                print(epsilon)
                print("–––––––––")

        env.close()
        torch.save(policy_DQN.state_dict(), "lunar_lander_policy.pt")

        # Create graph.
        fig, axes = plt.subplots(1, 2, dpi=200, figsize=(14,4))
        axes[0].plot(pd.Series(reward_arr).rolling(300).mean())
        axes[0].set_title(f"Reward History – $alpha$ = {self.alpha}")
        axes[1].plot(pd.Series(epsilon_arr).rolling(300).mean())
        axes[1].set_title(f"Epsilon History - $varepsilon = {self.ep_decay}$")
        plt.savefig("LunarLander_Res.png")
        #plt.hist(actions)
        #plt.show()


    def optimise(self, mini_batch, policy_DQN, target_DQN):
        target_Q_arr, policy_Q_arr = [], []

        for state, action, reward, new_state, terminated in mini_batch:
            if terminated:
                target = FloatTensor([reward])
            else:
                with torch.no_grad():
                    target = torch.FloatTensor(
                        reward + self.gamma*target_DQN(FloatTensor(new_state)).max()
                        )
            
            # Get the current set of Q values.
            current_Q = policy_DQN(FloatTensor(state))
            policy_Q_arr.append(current_Q)

            # Get the target set of Q values.
            target_Q = target_DQN(FloatTensor(state))

            # Adjust the specific action to the target that was just calculated.
            target_Q[action] = target
            target_Q_arr.append(target_Q)
    

        # Compute loss for whole minibatch.
        loss = F.mse_loss(torch.stack(policy_Q_arr), torch.stack(target_Q_arr))

        # Optimise weights over one step.
        self.optimiser.zero_grad()
        loss.backward()
        self.optimiser.step()

        policy_DQN, target_DQN = self.soft_update(policy_DQN, target_DQN, self.tau)

        return policy_DQN, target_DQN


    def soft_update(self, local_model, target_model, tau):
        """Soft update model parameters.
        θ_target = τ*θ_local + (1 - τ)*θ_target

        Params
        ======
            local_model (PyTorch model): weights will be copied from
            target_model (PyTorch model): weights will be copied to
            tau (float): interpolation parameter 
        """
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau*local_param.data + (1.0-tau)*target_param.data)

        return local_model, target_model

model = LunarLander()
model.train(episodes=2_500)

