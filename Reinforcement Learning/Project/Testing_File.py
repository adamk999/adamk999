import pandas as pd
import matplotlib.pyplot as plt
import pickle
from DeepQ import *
from Trading_Env import *
import numpy as np

def load_pickle(file_name):
    # Importing the dictionary back from the pickle file
    with open(file_name, 'rb') as file:
        loaded_dict = pickle.load(file)
        
    return loaded_dict


class Agent_Testing():
    # Hyperparameters:
    alpha = 0.01
    nodes = [32, 64, 64] # 32 64 64 
    num_states = 19
    num_actions = 25

    strategies = ["call", "call", "call", "put", "put", "put", "butterfly", "straddle", "strangle", "risk_rev", "bull_spread", "bear_spread"]
    strategy_map = {"call_0": "c_0", "call_1": "c_1", "call_2": "c_2", 
                    "put_0": "p_0", "put_1": "p_1", "put_2": "p_2",
                    "butteryfly": ["c_0","c_1","c_1", "c_2"], 
                    "straddle": ["p_0", "c_0"], "strangle": ["p_1", "c_1"],
                    "risk_rev": ["p_1", "c_1"], "bull_spread": ["c_1", "c_2"], "bear_spread": ["p_2", "p_1"]}


    def parse_action(self, action: int):
        if action == 24:
            return "wait", "_", "_"
        elif action >= 12:
            strat = self.strategies[int(action - 12)]
            key = list(self.strategy_map.keys())[int(action - 12)]
            direc = -1
        else:
            strat = self.strategies[action]
            key = list(self.strategy_map.keys())[action]
            direc = 1

        return strat, direc, self.strategy_map[key]


    def train(self, n_episodes):
        # Create policy network. Number of nodes in the h-layer can be adjusted.
        policy_dqn = DQN(in_states=self.num_states, nodes=self.nodes, out_actions=self.num_actions)
        policy_dqn.load_state_dict(torch.load("/Users/adamkeys/Documents/Computational_Finance/Advanced_Financial_ML/RL_Project/Trading_Policy_DQN.pt"))

        policy_dqn.eval()

  
        # List to keep track of rewards per episodes and epsilon.
        reward_arr = np.zeros(n_episodes)

        for i in range(n_episodes):
            env = Trading_Env(episode=i)

            state, len_episode = env.reset()

            episode_rets = np.zeros(len_episode)
            episode_memory = []
            action_arr = []

            # Agent navigates map until it falls into hole/reaches goal/200 actions.
            for steps in range(len_episode):
                # Selection action based on epsilon-greedy.
                action = policy_dqn(state).argmax().item()

                # Execute action.
                strat, direction, strike = self.parse_action(action)
                action_arr.append([strat, direction, strike])
                new_state, reward = env.step(strat, direction, strike)
                
                episode_memory.append([state, action])
                episode_rets[steps] = reward

                # Move to next state.
                state = new_state

            # Keep track of rewards collected per episode.
            reward_arr[i] = episode_rets.cumsum()[-1]
            print(reward_arr[i])

            save_pickle(f"episode_rets{i}.pkl", episode_rets)
            save_pickle(f"asset_hist_{i}.pkl", env.asset_history)
            save_pickle(f"port_val_hist_{i}.pkl", env.port_val_hist)
            save_pickle(f"episode_actions_{i}.pkl", action_arr)

        # Save policy.
        #save_pickle("reward_arr.pkl", reward_arr)


agent = Agent_Testing()
agent.train(n_episodes=25)







