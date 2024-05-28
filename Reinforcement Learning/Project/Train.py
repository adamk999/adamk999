import numpy as np
import torch
import sys
from math import log
from DeepQ import *
from Trading_Env import *


# TD(λ) --> intermediate rewards. assign reward for 


class Agent_DQN():
    # Hyperparameters:
    alpha = 0.01
    gamma = 0.9999
    eps_target = 0.1
    network_sync_rate = 2000   # Num steps agent takes before syncing policy/target networks.
    replay_memory_size = 1000  # Size of replay memory.
    mini_batch_size = 100       # Size of training dataset sampled from replay memory.
    nodes = [32, 64, 64] # 32 64 64 

    num_states = 18 
    num_actions = 24


    strategies = ["call", "call", "call", "put", "put", "put", "butterfly", "straddle", "strangle", "risk_rev", "bull_spread", "bear_spread"]
    strategy_map = {"call_0": "c_0", "call_1": "c_1", "call_2": "c_2", 
                    "put_0": "p_0", "put_1": "p_1", "put_2": "p_2",
                    "butteryfly": ["c_0","c_1","c_1", "c_2"], 
                    "straddle": ["p_0", "c_0"], "strangle": ["p_1", "c_1"],
                    "risk_rev": ["p_1", "c_1"], "bull_spread": ["c_1", "c_2"], "bear_spread": ["p_2", "p_1"]}

    loss_fn = nn.MSELoss()

    def parse_action(self, action: int):
        if action >= self.num_actions/2:
            strat = self.strategies[int(action - (self.num_actions/2))]
            key = list(self.strategy_map.keys())[int(action - (self.num_actions/2))]
            direc = -1
        else:
            strat = self.strategies[action]
            key = list(self.strategy_map.keys())[action]
            direc = 1

        return strat, direc, self.strategy_map[key]



    def train(self, n_episodes):
        # Create trading environment instance.
        epsilon = 1
        eps_decay = np.exp(log(self.eps_target)/n_episodes)
        memory = ReplayMemory(self.replay_memory_size)

        # Create policy and target network. Number of nodes in the h-layer can be adjusted.
        policy_dqn = DQN(in_states=self.num_states, nodes=self.nodes, out_actions=self.num_actions)
        target_dqn = DQN(in_states=self.num_states, nodes=self.nodes, out_actions=self.num_actions) 

        # Make target and policy networks the same.
        target_dqn.load_state_dict(policy_dqn.state_dict())
 
        self.optimiser = optim.Adam(policy_dqn.parameters(), lr=self.alpha)
 
        # List to keep track of rewards per episodes and epsilon.
        reward_arr = np.zeros(n_episodes)
        epsilon_arr = []

        # Track num steps taken, used for syncing policy/target networks.
        step_count = 0

        for i in range(n_episodes):
            episode_reward = 0 

            current_ep = np.random.randint(0,23) 
            env = Trading_Env(current_ep)

            state, len_episode = env.reset()

            # Agent navigates map until it falls into hole/reaches goal/200 actions.
            for steps in range(len_episode):
                # Selection action based on epsilon-greedy.
                if random.random() < epsilon:
                    # Select random action.
                    action = np.random.randint(0,self.num_actions)
                else:
                    # Select best action 
                    with torch.no_grad():
                        action = policy_dqn(state).argmax().item()

                # Execute action.
                strat, direction, strike = self.parse_action(action)
                new_state, reward = env.step(strat, direction, strike)
                
                # Save experience into memory.
                memory.append((state, action, new_state, reward))

                # Move to next state.
                state = new_state
                step_count += 1
                episode_reward += reward

            # Keep track of rewards collected per episode.
            reward_arr[i] = episode_reward 

            if i % 500 == 0 and i != 0:
                print("Episode:", i, "Avg Reward:", reward_arr[i-500:i].mean())
                save_pickle("reward_arr.pkl", reward_arr)

            # Check if enough experience has been collected and if at least 1 reward collected.
            if len(memory) > self.mini_batch_size:
                mini_batch = memory.sample(self.mini_batch_size)
                self.optimise(mini_batch, policy_dqn, target_dqn)

                # Decay epsilon.
                epsilon = max(epsilon*eps_decay, 0.01)                
                epsilon_arr.append(epsilon)

                # Copy policy network to target network after a certain number of steps.
                if step_count > self.network_sync_rate:
                    target_dqn.load_state_dict(policy_dqn.state_dict())
                    step_count = 0


        # Save policy.
        torch.save(policy_dqn.state_dict(), "Trading_Policy_DQN.pt")
        save_pickle("reward_arr.pkl", reward_arr)

        fig, axes = plt.subplots(1, 2, dpi=200, figsize=(14,4))
        axes[0].plot(pd.Series(reward_arr).rolling(200).mean())
        axes[0].set_title(f"Reward History – $alpha$ = {self.alpha}")
        axes[1].plot(pd.Series(epsilon_arr).rolling(200).mean())
        axes[1].set_title(f"Epsilon History - $varepsilon = {eps_decay}$")
        plt.savefig("Trading_DQN.png")
        
    def optimise(self, mini_batch, policy_dqn, target_dqn):
        # Get number of input nodes.
        num_states = self.num_states

        current_q_list = []
        target_q_list = []

        for state, action, new_state, reward in mini_batch:
            # Calculate target q value.
            with torch.no_grad():
                target = torch.FloatTensor(
                    reward + self.gamma*target_dqn(new_state).max()
                )
            
            # Get the current set of Q values
            current_q = policy_dqn(state)
            current_q_list.append(current_q)

            # Get the target set of Q values
            target_q = target_dqn(state)
        
            # Adjust the specific action to the target that was just calculated
            target_q[action] = target
            target_q_list.append(target_q)           

        # Compute loss for whole minibatch.
        loss = self.loss_fn(torch.stack(current_q_list), torch.stack(target_q_list))

        # Optimise model.
        self.optimiser.zero_grad()
        loss.backward()
        self.optimiser.step()


class Agent_TD():
    # Hyperparameters:
    alpha = 0.01
    lam = 0.95
    eps_target = 0.1
    replay_memory_size = 3000  # Size of replay memory.
    mini_batch_size = 100       # Size of training dataset sampled from replay memory.
    nodes = [32, 64, 64] # 32 64 64 

    num_states = 18 
    num_actions = 24

    strategies = ["call", "call", "call", "put", "put", "put", "butterfly", "straddle", "strangle", "risk_rev", "bull_spread", "bear_spread"]
    strategy_map = {"call_0": "c_0", "call_1": "c_1", "call_2": "c_2", 
                    "put_0": "p_0", "put_1": "p_1", "put_2": "p_2",
                    "butteryfly": ["c_0","c_1","c_1", "c_2"], 
                    "straddle": ["p_0", "c_0"], "strangle": ["p_1", "c_1"],
                    "risk_rev": ["p_1", "c_1"], "bull_spread": ["c_1", "c_2"], "bear_spread": ["p_2", "p_1"]}

    loss_fn = nn.MSELoss()

    def parse_action(self, action: int):
        if action >= self.num_actions/2:
            strat = self.strategies[int(action - (self.num_actions/2))]
            key = list(self.strategy_map.keys())[int(action - (self.num_actions/2))]
            direc = -1
        else:
            strat = self.strategies[action]
            key = list(self.strategy_map.keys())[action]
            direc = 1

        return strat, direc, self.strategy_map[key]


    def train(self, n_episodes):
        # Create trading environment instance.
        epsilon = 1
        eps_decay = np.exp(log(self.eps_target)/n_episodes)
        memory = ReplayMemory(self.replay_memory_size)

        # Create policy network. Number of nodes in the h-layer can be adjusted.
        policy_dqn = DQN(in_states=self.num_states, nodes=self.nodes, out_actions=self.num_actions)

        # Define policy network optimiser.
        self.optimiser = optim.Adam(policy_dqn.parameters(), lr=self.alpha)
 
        # List to keep track of rewards per episodes and epsilon.
        reward_arr = np.zeros(n_episodes)
        epsilon_arr = []


        for i in range(n_episodes):
            current_ep = np.random.randint(0,23) 
            env = Trading_Env(current_ep)

            state, len_episode = env.reset()

            episode_rets = np.zeros(len_episode)
            episode_memory = []

            # Agent navigates map until it falls into hole/reaches goal/200 actions.
            for steps in range(len_episode):
                # Selection action based on epsilon-greedy.
                if random.random() < epsilon:
                    # Select random action.
                    action = np.random.randint(0,self.num_actions)
                else:
                    # Select best action 
                    with torch.no_grad():
                        action = policy_dqn(state).argmax().item()

                # Execute action.
                strat, direction, strike = self.parse_action(action)
                new_state, reward = env.step(strat, direction, strike)
                
                episode_memory.append([state, action])
                episode_rets[steps] = reward

                # Move to next state.
                state = new_state

            # Keep track of rewards collected per episode.
            reward_arr[i] = episode_rets.cumsum()[-1]

            if i % 500 == 0 and i != 0:
                print("Episode:", i, "Avg Reward:", reward_arr[i-500:i].mean())
                save_pickle("/40k_lower_LR/reward_arr.pkl", reward_arr)
                torch.save(policy_dqn.state_dict(), "/40k_lower_LR/Trading_Policy_DQN.pt")
            
            if i % 250 == 0 and i != 0:
                save_pickle(f"/40k_lower_LR/trading_hist_{i}.pkl", env.)


            
            episode_hist = env.asset_history[:-1,:] 
            episode_memory = self.calc_lambda_rets(episode_hist, episode_memory, lam=self.lam)
            [memory.append(i) for i in episode_memory]


            # Check if enough experience has been collected and if at least 1 reward collected.
            if len(memory) > self.mini_batch_size:
                mini_batch = memory.sample(self.mini_batch_size)
                self.optimise(mini_batch, policy_dqn)

                # Decay epsilon.
                epsilon = max(epsilon*eps_decay, 0.01)                
                epsilon_arr.append(epsilon)


        # Save policy.
        torch.save(policy_dqn.state_dict(), "Trading_Policy_DQN.pt")
        save_pickle("reward_arr.pkl", reward_arr)

        fig, axes = plt.subplots(1, 2, dpi=200, figsize=(14,4))
        axes[0].plot(pd.Series(reward_arr).rolling(200).mean())
        axes[0].set_title(f"Reward History – $alpha$ = {self.alpha}")
        axes[1].plot(pd.Series(epsilon_arr).rolling(200).mean())
        axes[1].set_title(f"Epsilon History - $varepsilon = {eps_decay}$")
        plt.savefig("Trading_DQN.png")

    def calc_lambda_rets(self, asset_hist, memory, lam):
        n_data = len(asset_hist)
        lam_arr = np.full(n_data, lam)
        n = np.arange(n_data)
        lam_arr = lam_arr**n

        for i in range(n_data):
            row = pd.Series(asset_hist[i,i:])
            if row.iloc[-1] == 0:
                row = row[row != 0].diff(1).dropna()
            else:
                row = row.diff(1).dropna()
            
            ret = (1-lam)*np.sum(lam_arr[:len(row)]*row)
            
            memory[i].append(ret)
    
        return memory 
        
    def optimise(self, mini_batch, policy_dqn):
        # Get number of input nodes.
        num_states = self.num_states

        current_q_list = []
        target_q_list = []

        for state, action, lam_ret in mini_batch:
            if lam_ret != 0:
                # Calculate target q value.
                target = torch.FloatTensor([lam_ret])
                
                # Get the current set of Q values.
                current_q = policy_dqn(state)
                current_q_list.append(current_q)

                # Get the target set of Q values.
                target_q = policy_dqn(state)
        
                # Adjust the specific action to the target that was just calculated.
                target_q[action] = target
                target_q_list.append(target_q)        

        # Compute loss for whole minibatch.
        loss = self.loss_fn(torch.stack(current_q_list), torch.stack(target_q_list))

        # Optimise model.
        self.optimiser.zero_grad()
        loss.backward()
        self.optimiser.step()


agent = Agent_TD()
#agent.train(sys.argv[1])
agent.train(1)
