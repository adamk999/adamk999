import torch
import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
import random
from collections import deque
from torch import nn
import torch.nn.functional as F


# Create NN model/class.
# Define replay class.
# Create training class:
    # ––––– TRAIN –––––
    # Int Q/target & policy DQN, replay.
    # Iterate(episodes):
        # Int S
        # Iterate(steps):
            # A <-- epsilon-g w.r.t. policy DQN.
            # A --> S', R
            # Add (S, A, S', R) to memory.
        
        # If len(mem) > x & total reward > y:
            # Sample mini_batch
            # Optimise policy DQN

            # If net_sync_rate > z:
                # Copy policy DQN --> target DQN.

    # –––––– Optimise –––––––
    # for S, A, R, S', term in mini_batch:
        # Int policy_Q_arr = [], target_Q_arr = []

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

# Define model.
class DQN(nn.Module):
    def __init__(self, in_states, h1_nodes, out_actions):
        super().__init__()

        # Define network layers.
        self.fc1 = nn.Linear(in_states, h1_nodes) # first fully connected layer.
        self.out = nn.Linear(h1_nodes, out_actions) # output layer.

    def forward(self, x):
        x = F.relu(self.fc1(x)) # Apply ReLU activation.
        x = self.out(x) # Calculate output.
        
        return x

# Define memory for Experience Replay.
class ReplayMemory():
    def __init__(self, maxlen):
        self.memory = deque([], maxlen=maxlen)

    def append(self, transition):
        self.memory.append(transition)

    def sample(self, sample_size):
        return random.sample(self.memory, sample_size)

    def __len__(self):
        return len(self.memory)
    
class FrozenLakeDQL():
    # Hyperparameters:
    alpha = 0.001
    gamma = 0.9
    network_sync_rate = 10     # Num steps agent takes before syncing policy/target networks.
    replay_memory_size = 1000  # Size of replay memory.
    mini_batch_size = 32       # Size of training dataset sampled from replay memory.

    # Neural Network.
    loss_fn = nn.MSELoss()
    optimiser = None

    actions = ["L", "D", "R", "U"]
    
    # Train the FrozenLake Environment.
    def train(self, episodes, render=False, is_slippery=False):
        # Create FrozenLake instance.
        env = gym.make("FrozenLake-v1", map_name="4x4", is_slippery=is_slippery, render_mode="human" if render else None)
        num_states = env.observation_space.n
        num_actions = env.action_space.n

        epsilon = 1
        memory = ReplayMemory(self.replay_memory_size)

        # Create policy and target network. Number of nodes in the h-layer can be adjusted.
        policy_dqn = DQN(in_states=num_states, h1_nodes=num_states, out_actions=num_actions)
        target_dqn = DQN(in_states=num_states, h1_nodes=num_states, out_actions=num_actions) 

        # Make target and policy networks the same.
        target_dqn.load_state_dict(policy_dqn.state_dict())
        
        print("Policy (random, before training)")
        self.print_dqn(policy_dqn)

        # Policy network optimiser.
        self.optimiser = torch.optim.Adam(policy_dqn.parameters(), lr=self.alpha)

        # List to keep track of rewards per episodes and epsilon.
        rewards_per_episode = np.zeros(episodes)
        epsilon_history = []

        # Track num steps taken, used for syncing policy/target networks.
        step_count = 0

        for i in range(episodes):
            state = env.reset()[0]
            if render:
                env.render()
            terminated = False
            truncated = False

            # Agent navigates map until it falls into hole/reaches goal/200 actions.
            while (not terminated and not truncated):
                # Selection action based on epsilon-greedy.
                if random.random() < epsilon:
                    # Select random action.
                    action = env.action_space.sample()
                else:
                    # Select best action 
                    with torch.no_grad():
                        action = policy_dqn(self.state_to_dqn_input(state, num_states)).argmax().item()

                # Execute action.
                new_state, reward, terminated, truncated, _ = env.step(action)
                
                # Save experience into memory.
                memory.append((state, action, new_state, reward, terminated))

                # Move to next state.
                state = new_state
                step_count += 1

            # Keep track of rewards collected per episode.
            if reward == 1:
                rewards_per_episode[i] = 1

            # Check if enough experience has been collected and if at least 1 reward collected.
            if len(memory) > self.mini_batch_size and np.sum(rewards_per_episode) > 0:
                mini_batch = memory.sample(self.mini_batch_size)
                self.optimise(mini_batch, policy_dqn, target_dqn)

                # Decay epsilon.
                epsilon = max(epsilon - 1/episodes, 0)
                epsilon_history.append(epsilon)

                # Copy policy network to target network after a certain number of steps.
                if step_count > self.network_sync_rate:
                    target_dqn.load_state_dict(policy_dqn.state_dict())
                    step_count = 0

        # Close environment.
        env.close()

        # Save policy.
        torch.save(policy_dqn.state_dict(), "frozen_lake_dqn.pt")

        # Create graph.
        plt.figure(dpi=200)
        sum_rewards = np.zeros(episodes)

        for x in range(episodes):
            sum_rewards[x] = np.sum(rewards_per_episode[max(0, x-100):(x+1)])
        
        plt.subplot(121)
        plt.plot(sum_rewards)

        plt.subplot(122)
        plt.plot(epsilon_history)

        plt.savefig("frozen_lake_dqn.png")

    def optimise(self, mini_batch, policy_dqn, target_dqn):
        # Get number of input nodes.
        num_states = policy_dqn.fc1.in_features

        current_q_list = []
        target_q_list = []

        for state, action, new_state, reward, terminated in mini_batch:
            if terminated:
                # Agent at goal (rew=1) or fell hole (rew=0).
                # when in terminated state, target q val should be set to reward.
                target = torch.FloatTensor([reward])
            
            else:
                # Calculate target q value.
                with torch.no_grad():
                    target = torch.FloatTensor(
                        reward + self.gamma*target_dqn(self.state_to_dqn_input(new_state, num_states)).max()
                    )
            
            # Get the current set of Q values
            current_q = policy_dqn(self.state_to_dqn_input(state, num_states))
            current_q_list.append(current_q)

            # Get the target set of Q values
            target_q = target_dqn(self.state_to_dqn_input(state, num_states))
            
            # Adjust the specific action to the target that was just calculated
            target_q[action] = target
            target_q_list.append(target_q)           

        # Compute loss for whole minibatch.
        loss = self.loss_fn(torch.stack(current_q_list), torch.stack(target_q_list))

        # Optimise model.
        self.optimiser.zero_grad()
        loss.backward()
        self.optimiser.step()

    def state_to_dqn_input(self, state:int, num_states:int)->torch.Tensor:
        # Converts an integer state to one-hot-encoded tensor representation.
        input_tensor = torch.zeros(num_states)
        input_tensor[state] = 1

        return input_tensor

    def test(self, episodes, is_slippery=False):
        # Create FrozenLake environment.
        env = gym.make("FrozenLake-v1", map_name="4x4", is_slippery=is_slippery, render_mode="human")
        num_states = env.observation_space.n
        num_actions = env.action_space.n

        # Load learned policy.
        policy_dqn = DQN(in_states=num_states, h1_nodes=num_states, out_actions=num_actions)
        policy_dqn.load_state_dict(torch.load("frozen_lake_dqn.pt"))
        policy_dqn.eval()

        print("Policy (trained):")
        self.print_dqn(policy_dqn)

        for i in range(episodes):
            state = env.reset()[0]
            env.render()
            terminated = False
            truncated = False

            while (not terminated and not truncated):
                # Select best action.
                with torch.no_grad():
                    action = policy_dqn(self.state_to_dqn_input(state, num_states)).argmax().item()

                # Execute action.
                state, reward, terminated, truncated, _ = env.step(action)

        env.close()

    def print_dqn(self, dqn):
        # Print DQN: state, best action, q values
        
        # Get number of input nodes
        num_states = dqn.fc1.in_features

        # Loop each state and print policy to console
        for s in range(num_states):
            #  Format q values for printing
            q_values = ''
            for q in dqn(self.state_to_dqn_input(s, num_states)).tolist():
                q_values += "{:+.2f}".format(q)+' '  # Concatenate q values, format to 2 decimals
            q_values=q_values.rstrip()              # Remove space at the end

            # Map the best action to L D R U
            best_action = self.actions[dqn(self.state_to_dqn_input(s, num_states)).argmax()]

            # Print policy in the format of: state, action, q values
            # The printed layout matches the FrozenLake map.
            print(f'{s:02},{best_action},[{q_values}]', end=' ')         
            if (s+1)%4==0:
                print() # Print a newline every 4 states


frozen_lake = FrozenLakeDQL()
is_slippery = False
#frozen_lake.train(1000, render=False, is_slippery=is_slippery)
frozen_lake.test(4, is_slippery=is_slippery)
