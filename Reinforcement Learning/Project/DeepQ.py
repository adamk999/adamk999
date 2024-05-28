import torch as T
from torch import optim, nn
import numpy as np          
import torch.nn.functional as F
import pandas as pd             
import matplotlib.pyplot as plt 
import datetime as dt           
import time                     
import pickle    
import random
from collections import deque

class DQN(nn.Module):
    def __init__(self, in_states, nodes, out_actions):
        super().__init__()
    
        h1_nodes, h2_nodes, h3_nodes = nodes

        # Define network layers.
        self.fc1 = nn.Linear(in_states, h1_nodes) # first fully connected layer.
        self.fc2 = nn.Linear(h1_nodes, h2_nodes)
        self.fc3 = nn.Linear(h2_nodes, h3_nodes)
        self.V = nn.Linear(h3_nodes, 1)
        self.A = nn.Linear(h3_nodes, out_actions)
        
        self.device = T.device("cuda:0" if T.cuda.is_available() else "cpu")
        self.to(self.device)


    def forward(self, x):
        x = F.relu(self.fc1(x)) # Apply ReLU activation.
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        V = self.V(x)
        A = self.A(x)
       
        Q = V + A - A.mean()
        return Q


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


def save_pickle(file_name, object):
    # Open the file in binary write mode
    with open(file_name, 'wb') as file:
        # Use pickle.dump() to export the dictionary to the file
        pickle.dump(object, file)

    print("Dictionary exported successfully.")

def load_pickle(file_name):
    # Importing the dictionary back from the pickle file
    with open(file_name, 'rb') as file:
        loaded_dict = pickle.load(file)
        
    return loaded_dict

